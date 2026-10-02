import json
import time

import requests

from inventory import decimal_value, read_inventory
from tools import TOOLS, confirm_pending_order


MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"

MAX_MODEL_RETRIES = 2
MAX_PLAN_ATTEMPTS = 4
MEMORY_MESSAGES = 12

ORDER_STEPS = [
    "check_stock",
    "calculate_total",
    "prepare_invoice",
]


class FruitAgent:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.memory = []
        self.pending_order = None
        self.last_state = None

    def log(self, label, value):
        if self.verbose:
            if not isinstance(value, str):
                value = json.dumps(value, indent=2)

            print(f"\n[{label}]\n{value}")

    def remember(self, user_message, assistant_message):
        self.memory.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ])

        self.memory = self.memory[-MEMORY_MESSAGES:]

    def ask_model(self, system_prompt, payload):
        last_error = None

        for attempt in range(MAX_MODEL_RETRIES):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL,
                        "stream": False,
                        "format": "json",
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": json.dumps(payload),
                            },
                        ],
                        "options": {"temperature": 0},
                    },
                    timeout=120,
                )

                response.raise_for_status()

                return json.loads(
                    response.json()["message"]["content"]
                )

            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError,
            ) as error:
                last_error = error
                self.log("RETRY", str(error))

                if attempt + 1 < MAX_MODEL_RETRIES:
                    time.sleep(1)

        raise RuntimeError(
            f"Model request failed after retries: {last_error}"
        )

    def interpret_request(self, message, catalog):
        prompt = """
You interpret requests for a fruit shop.
Return only one JSON object.

Choose one intent:
- inventory: list products, stock, or all prices
- product: information about one product
- order: create or modify an invoice
- clarify: the request is ambiguous or unsupported

JSON structure:
{
  "intent": "order",
  "product": null,
  "items": [
    {"name": "apple", "quantity_kg": "3"}
  ],
  "budget": null,
  "allow_reduction": false,
  "question": null
}

Rules:
- Use exact singular product names from the catalog.
- For order requests, include the complete proposed basket.
- budget is a decimal string or null.
- allow_reduction is true only if the user explicitly permits
  reducing quantities, including permission retained from context.
- For a modification, use the pending order as context.
- Do not invent missing quantities.
- Do not substitute or remove requested fruits.
- If important details are missing, choose clarify and ask a question.
- Do not confirm orders or update inventory.
"""

        request = self.ask_model(
            prompt,
            {
                "message": message,
                "catalog": catalog,
                "conversation": self.memory,
                "pending_order": self.pending_order,
            },
        )

        self.validate_request(request)
        return request

    def validate_request(self, request):
        if not isinstance(request, dict):
            raise ValueError("The interpreted request must be an object.")

        intent = request.get("intent")

        if intent not in {"inventory", "product", "order", "clarify"}:
            raise ValueError("Invalid request intent.")

        if intent == "product":
            if not isinstance(request.get("product"), str):
                raise ValueError("A product name is required.")

        if intent == "clarify":
            if not isinstance(request.get("question"), str):
                raise ValueError("A clarification question is required.")

        if intent != "order":
            return

        items = request.get("items")

        if not isinstance(items, list) or not items:
            raise ValueError("An order must contain at least one item.")

        names = set()

        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each order item must be an object.")

            name = item.get("name")

            if not isinstance(name, str) or name in names:
                raise ValueError("Product names must be valid and unique.")

            names.add(name)

            if decimal_value(item.get("quantity_kg")) <= 0:
                raise ValueError("Quantities must be positive.")

        budget = request.get("budget")

        if budget is not None and decimal_value(budget) < 0:
            raise ValueError("Budget cannot be negative.")

        if not isinstance(request.get("allow_reduction"), bool):
            raise ValueError("allow_reduction must be a boolean.")

    def create_plan(self, state):
        request = state["request"]

        # Direct questions use simple, deterministic routing.
        if request["intent"] == "inventory":
            return {
                "items": [],
                "steps": ["list_inventory"],
            }

        if request["intent"] == "product":
            return {
                "items": [],
                "steps": ["get_product"],
            }

        prompt = """
You are a fruit-order planner.
Return only JSON:
{
  "items": [
    {"name": "apple", "quantity_kg": "3"}
  ],
  "steps": [
    "check_stock",
    "calculate_total",
    "prepare_invoice"
  ]
}

Rules:
- Use exactly the three steps shown, in that order.
- Include every product in the interpreted request exactly once.
- Do not add or remove products.
- Every quantity must be positive.
- Never exceed the quantity in the interpreted request.
- On the first attempt, use the interpreted quantities unchanged.
- On later attempts, use observations to revise quantities.
- Reduce quantities only if allow_reduction is true.
- Revised quantities should fit both stock and budget.
- Do not change the user's constraints.
- Never update inventory or claim the order is confirmed.
"""

        return self.ask_model(
            prompt,
            {
                "request": request,
                "catalog": state["catalog"],
                "previous_attempts": state["history"],
            },
        )

    def validate_plan(self, plan, state):
        if not isinstance(plan, dict):
            raise ValueError("The plan must be an object.")

        intent = state["request"]["intent"]

        expected_steps = {
            "inventory": ["list_inventory"],
            "product": ["get_product"],
            "order": ORDER_STEPS,
        }[intent]

        if plan.get("steps") != expected_steps:
            raise ValueError("Invalid tools or execution order.")

        if intent != "order":
            return

        items = plan.get("items")

        if not isinstance(items, list):
            raise ValueError("Plan items must be a list.")

        original = {
            item["name"]: decimal_value(item["quantity_kg"])
            for item in state["request"]["items"]
        }

        seen = set()

        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each planned item must be an object.")

            name = item.get("name")

            if name not in original or name in seen:
                raise ValueError("Unexpected or duplicate product.")

            seen.add(name)
            quantity = decimal_value(item.get("quantity_kg"))

            if quantity <= 0 or quantity > original[name]:
                raise ValueError("Invalid planned quantity.")

            if (
                not state["request"]["allow_reduction"]
                and quantity != original[name]
            ):
                raise ValueError(
                    "The user did not authorize quantity reductions."
                )

        if seen != set(original):
            raise ValueError("The plan must retain every requested fruit.")

    def evaluate(self, step, result, state):
        request = state["request"]

        if not result["ok"]:
            if (
                request["intent"] == "order"
                and request["allow_reduction"]
                and step == "check_stock"
            ):
                return {
                    "decision": "REPLAN",
                    "reason": result["reason"],
                }

            return {
                "decision": "ASK_USER",
                "reason": result["reason"],
            }

        if step == "calculate_total":
            budget = request.get("budget")
            total = decimal_value(state["invoice"]["total"])

            if budget is not None and total > decimal_value(budget):
                return {
                    "decision": (
                        "REPLAN"
                        if request["allow_reduction"]
                        else "ASK_USER"
                    ),
                    "reason": (
                        f"Total ${total} exceeds budget ${budget}."
                    ),
                }

        if step == "prepare_invoice":
            return {
                "decision": "AWAIT_CONFIRMATION",
                "reason": "Stock and budget checks passed.",
            }

        if step in {"list_inventory", "get_product"}:
            return {
                "decision": "FINISH",
                "reason": "Requested information retrieved.",
            }

        return {
            "decision": "CONTINUE",
            "reason": "Step passed evaluation.",
        }

    def format_invoice(self, invoice):
        lines = ["Invoice quote:"]

        for item in invoice["items"]:
            lines.append(
                f"- {item['name']}: {item['quantity_kg']} kg "
                f"x ${item['price_per_kg']} "
                f"= ${item['line_total']}"
            )

        lines.append(f"Total: ${invoice['total']}")
        lines.append(
            "Type 'confirm order' to purchase, "
            "or 'cancel order' to discard this quote."
        )

        return "\n".join(lines)

    def format_information(self, result):
        products = (
            result["products"]
            if "products" in result
            else [result["product"]]
        )

        return "\n".join(
            f"- {product['name']}: "
            f"${product['price_per_kg']}/kg; "
            f"stock: {product['stock_kg']} kg"
            for product in products
        )

    def execute_request(self, request, catalog):
        state = {
            "request": request,
            "catalog": catalog,
            "plan": None,
            "invoice": None,
            "history": [],
            "status": "PLANNING",
        }

        self.last_state = state

        for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            state["invoice"] = None
            state["status"] = "PLANNING"

            self.log("PLANNING ATTEMPT", str(attempt))

            try:
                plan = self.create_plan(state)
                self.validate_plan(plan, state)

            except (
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
            ) as error:
                state["history"].append({
                    "feedback": f"Invalid plan: {error}",
                })
                self.log("INVALID PLAN", str(error))
                continue

            state["plan"] = plan
            self.log("PLAN", plan)

            record = {
                "plan": plan,
                "observations": [],
            }

            state["history"].append(record)

            for step in plan["steps"]:
                state["status"] = "EXECUTING"
                self.log("ACT", step)

                result = TOOLS[step](state)

                self.log("OBSERVE", result)

                state["status"] = "EVALUATING"
                evaluation = self.evaluate(step, result, state)

                record["observations"].append({
                    "tool": step,
                    "result": result,
                    "evaluation": evaluation,
                })

                self.log("EVALUATE", evaluation)
                decision = evaluation["decision"]

                if decision == "REPLAN":
                    break

                if decision == "ASK_USER":
                    state["status"] = "ASK_USER"

                    return (
                        evaluation["reason"]
                        + "\nPlease revise the request or clarify "
                        "which constraints may change. "
                        "No new order was prepared."
                    )

                if decision == "FINISH":
                    state["status"] = "FINISHED"
                    return self.format_information(result)

                if decision == "AWAIT_CONFIRMATION":
                    state["status"] = "AWAITING_CONFIRMATION"

                    self.pending_order = {
                        "items": plan["items"],
                        "invoice": state["invoice"],
                        "budget": request.get("budget"),
                        "allow_reduction": request["allow_reduction"],
                    }

                    return self.format_invoice(state["invoice"])

        state["status"] = "FAILED"

        return (
            "I could not produce a valid plan within the attempt limit. "
            "No new order was prepared and inventory is unchanged. "
            "Please revise the quantities or budget."
        )

    def handle_command(self, command):
        if command == "reset":
            self.memory.clear()
            self.pending_order = None
            self.last_state = None
            return "Conversation memory and pending order cleared."

        if command == "cancel order":
            self.pending_order = None
            return "Pending order cancelled. Inventory is unchanged."

        if command == "confirm order":
            if self.pending_order is None:
                return "There is no pending order to confirm."

            result = confirm_pending_order(self.pending_order)

            # Clear stale quotes as well as completed orders.
            self.pending_order = None

            return result["reason"]

        return None

    def chat(self, message):
        command = message.strip().lower()

        try:
            command_response = self.handle_command(command)

            if command_response is not None:
                return command_response

            catalog = read_inventory()

            request = self.interpret_request(message, catalog)
            self.log("ROUTE / CONSTRAINTS", request)

            if request["intent"] == "clarify":
                answer = request["question"]
            else:
                answer = self.execute_request(request, catalog)

            self.remember(message, answer)
            return answer

        except (
            OSError,
            ValueError,
            RuntimeError,
            TypeError,
            KeyError,
        ) as error:
            return (
                f"Operation failed: {error}\n"
                "No successful order confirmation was completed."
            )