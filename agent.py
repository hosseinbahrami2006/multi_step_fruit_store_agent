import json
from decimal import Decimal
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from inventory import decimal_value, find_product_name, read_inventory
from tools import (
    TOOLS,
    TOOL_SCHEMAS,
    PlanningConstraintError,
    adjust_order_items,
    confirm_pending_order,
    parse_tool_call_text,
    tool_call_to_request,
    validate_tool_arguments,
)


MODEL = "qwen2.5:3b"
MAX_MODEL_RETRIES = 2
MAX_PLAN_ATTEMPTS = 4
MEMORY_MESSAGES = 12

ORDER_STEPS = ["check_stock", "calculate_total", "prepare_invoice"]

STEPS_BY_INTENT = {
    "inventory": ["list_inventory"],
    "product": ["get_product"],
    "order": ORDER_STEPS,
}


# The model must choose exactly one tool.
SYSTEM_PROMPT = """
You are the decision maker of a fruit shop agent.
You receive one JSON object with: message, catalog, conversation, pending_order.
Your only job is to call exactly ONE tool. Never answer with plain text.

Choose the tool:
- list_inventory: the user wants the product list, prices, or stock
- get_product: the user asks about one product
- quote_order: the user wants a new or changed invoice
- ask_clarification: the message is unclear or not supported

Rules for quote_order:
- Quantities and budget are decimal strings. Quantities are in kg.
- If the user gives pounds, convert to kg (1 lb = 0.45359237 kg).
- Always send the complete basket in one call.
- For changes, use pending_order and conversation as context.
- Keep an earlier budget unless the user changes or removes it.
- Never calculate totals and never reduce quantities yourself.
- A low budget alone is NOT permission to reduce quantities.
- allow_reduction is true only if the user clearly allows smaller quantities.
- Never invent quantities. Never remove or replace fruits.

Rules for ask_clarification:
- Use it if the budget currency is different from the catalog currency.
- Use it if important details are missing.
- Always write questions in English.

Example: "I need 3 kg of apples. My budget is 5 USD. Reduce if needed."
-> quote_order with
   {"items": [{"name": "apple", "quantity_kg": "3"}],
    "budget": "5", "allow_reduction": true}
"""


# State passed between graph nodes.
# State keys must differ from graph node names.
class AgentState(TypedDict, total=False):
    message: str
    memory: list
    pending_order: Optional[dict]
    inventory: dict
    request: dict
    plan: dict
    step_index: int
    invoice: Optional[dict]
    result: dict
    attempts: int
    history: list
    decision: str
    reason: str
    answer: str
    new_pending: Optional[dict]


# Compare requested and planned quantities numerically.
def quantities_changed(planned_items, requested_items):
    requested = {
        item["name"]: decimal_value(item["quantity_kg"])
        for item in requested_items
    }

    for item in planned_items:
        if decimal_value(item["quantity_kg"]) != requested[item["name"]]:
            return True

    return False


# Format an invoice for display.
def format_invoice(invoice, title, note=None):
    currency = invoice["currency"]
    lines = [title]

    if note:
        lines.append(note)

    for item in invoice["items"]:
        lines.append(
            f"- {item['name']}: {item['quantity_kg']} kg "
            f"x {item['price_per_kg']} {currency} "
            f"= {item['line_total']} {currency}"
        )

    lines.append(f"Total: {invoice['total']} {currency}")
    return "\n".join(lines)


# Format products for display.
def format_products(products, currency):
    return "\n".join(
        f"- {product['name']}: {product['price_per_kg']} {currency}/kg, "
        f"stock: {product['stock_kg']} kg"
        for product in products
    )


# Manages conversation state and the workflow.
class FruitAgent:

    # Initialize the model and graph.
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.memory = []
        self.pending_order = None

        self.model = ChatOllama(
            model=MODEL,
            temperature=0,
        ).bind_tools(TOOL_SCHEMAS)

        self.graph = self.build_graph()

    # Print a log entry when verbose mode is enabled.
    def log(self, label, value):
        if self.verbose:
            if not isinstance(value, str):
                value = json.dumps(value, indent=2)
            print(f"\n[{label}]\n{value}")

    # Save the latest user message and answer.
    def remember(self, user_message, answer):
        self.memory.append({"role": "user", "content": user_message})
        self.memory.append({"role": "assistant", "content": answer})
        self.memory = self.memory[-MEMORY_MESSAGES:]

    # Read exactly one tool call from the model response.
    def read_tool_call(self, response):
        tool_calls = getattr(response, "tool_calls", None) or []

        if len(tool_calls) > 1:
            raise ValueError(
                "Call exactly one tool. "
                "Put the complete basket in one quote_order call."
            )

        if tool_calls:
            return tool_calls[0]["name"], tool_calls[0].get("args", {})

        return parse_tool_call_text(response.content)

    # Ask the model for a validated tool call, retrying on errors.
    def ask_model(self, payload):
        last_error = None

        for attempt in range(MAX_MODEL_RETRIES):
            try:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(payload)),
                ]

                if last_error is not None:
                    messages.append(HumanMessage(content=(
                        f"Your previous answer was rejected: {last_error} "
                        "Call exactly one tool with valid arguments."
                    )))

                response = self.model.invoke(messages)

                name, arguments = self.read_tool_call(response)
                arguments = validate_tool_arguments(name, arguments)
                self.log("TOOL CALL", {"tool": name, "arguments": arguments})

                return tool_call_to_request(name, arguments)

            except Exception as error:
                last_error = error
                self.log("MODEL RETRY", str(error))

        raise RuntimeError(f"The model failed: {last_error}")

    # Validate and normalize the model's request.
    def check_request(self, request, products):
        if not isinstance(request, dict):
            raise ValueError("The model answer must be a JSON object.")

        intent = request.get("intent")
        available = ", ".join(product["name"] for product in products)

        if intent == "clarify":
            question = request.get("question")
            if not isinstance(question, str) or not question.strip():
                question = "Could you explain your request in more detail?"
            return {"intent": "clarify", "question": question}

        if intent == "inventory":
            return {"intent": "inventory"}

        if intent == "product":
            name = find_product_name(request.get("product"), products)
            if name is None:
                return {
                    "intent": "clarify",
                    "question": f"We do not sell that product. "
                                f"Available products: {available}.",
                }
            return {"intent": "product", "product": name}

        if intent != "order":
            raise ValueError("The model returned an unknown intent.")

        items = request.get("items")

        if not isinstance(items, list) or not items:
            raise ValueError("An order needs at least one item.")

        basket = {}

        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each order item must be an object.")

            name = find_product_name(item.get("name"), products)

            if name is None:
                return {
                    "intent": "clarify",
                    "question": f"We do not sell '{item.get('name')}'. "
                                f"Available products: {available}.",
                }

            quantity = decimal_value(item.get("quantity_kg"))

            if quantity <= 0:
                raise ValueError("Quantities must be positive.")

            basket[name] = basket.get(name, Decimal("0")) + quantity

        budget = request.get("budget")

        if budget is not None:
            budget = decimal_value(budget)
            if budget < 0:
                raise ValueError("The budget cannot be negative.")
            budget = str(budget)

        return {
            "intent": "order",
            "items": [
                {"name": name, "quantity_kg": str(quantity)}
                for name, quantity in basket.items()
            ],
            "budget": budget,
            "allow_reduction": request.get("allow_reduction") is True,
        }

    # ---------------- Graph nodes ----------------

    # Load inventory and interpret the request.
    def interpret_node(self, state):
        inventory = read_inventory()

        raw_request = self.ask_model({
            "message": state["message"],
            "catalog": inventory,
            "conversation": state["memory"],
            "pending_order": state["pending_order"],
        })

        request = self.check_request(raw_request, inventory["products"])
        self.log("ROUTE / CONSTRAINTS", request)

        if request["intent"] == "clarify":
            return {
                "inventory": inventory,
                "request": request,
                "decision": "STOP",
                "answer": request["question"],
            }

        return {
            "inventory": inventory,
            "request": request,
            "decision": "PLAN",
            "attempts": 0,
            "history": [],
        }

    # Create or revise a plan.
    def plan_node(self, state):
        attempts = state["attempts"] + 1
        request = state["request"]
        self.log("PLANNING ATTEMPT", str(attempts))

        if attempts > MAX_PLAN_ATTEMPTS:
            return {
                "attempts": attempts,
                "decision": "FAILED",
                "reason": "I could not make a valid plan in time.",
            }

        items = [dict(item) for item in request.get("items", [])]

        if state["history"] and request.get("allow_reduction"):
            self.log("REPLAN TOOL", "adjust_order_items")

            try:
                items = adjust_order_items(
                    items,
                    state["inventory"]["products"],
                    request.get("budget"),
                )
            except PlanningConstraintError as error:
                return {
                    "attempts": attempts,
                    "decision": "ASK_USER",
                    "reason": str(error),
                }

        plan = {
            "items": items,
            "steps": STEPS_BY_INTENT[request["intent"]],
        }

        self.log("PLAN", plan)

        return {
            "attempts": attempts,
            "plan": plan,
            "step_index": 0,
            "invoice": None,
            "decision": "ACT",
        }

    # Run the current tool and save its result.
    def act_node(self, state):
        step = state["plan"]["steps"][state["step_index"]]
        self.log("ACT", step)

        try:
            result = TOOLS[step](state)
        except (ValueError, KeyError) as error:
            result = {"ok": False, "reason": str(error)}

        self.log("OBSERVE", result)

        update = {"result": result}

        if step == "calculate_total" and result["ok"]:
            update["invoice"] = result["invoice"]

        return update

    # Evaluate the tool result and choose the next action.
    def evaluate_node(self, state):
        step = state["plan"]["steps"][state["step_index"]]
        result = state["result"]
        request = state["request"]
        can_reduce = request.get("allow_reduction", False)

        decision = "CONTINUE"
        reason = "Step passed."

        if not result["ok"]:
            reason = result["reason"]
            if request["intent"] == "order" and can_reduce and step == "check_stock":
                decision = "REPLAN"
            else:
                decision = "ASK_USER"

        elif step == "calculate_total":
            budget = request.get("budget")
            total = decimal_value(state["invoice"]["total"])

            if budget is not None and total > decimal_value(budget):
                decision = "REPLAN" if can_reduce else "ASK_USER"
                reason = f"Total {total} is above the budget of {budget}."

        elif step == "prepare_invoice":
            decision = "AWAIT_CONFIRMATION"
            reason = "Stock and budget checks passed."

        elif step in ("list_inventory", "get_product"):
            decision = "FINISH"
            reason = "Information is ready."

        evaluation = {"tool": step, "decision": decision, "reason": reason}
        self.log("EVALUATE", evaluation)

        return {
            "decision": decision,
            "reason": reason,
            "step_index": state["step_index"] + 1,
            "history": state["history"] + [evaluation],
        }

    # Format the final answer and any pending quote.
    def finish_node(self, state):
        decision = state["decision"]
        currency = state["inventory"]["currency"]

        if decision == "FINISH":
            return {
                "answer": format_products(state["result"]["products"], currency),
            }

        if decision == "AWAIT_CONFIRMATION":
            plan = state["plan"]
            request = state["request"]

            note = None
            if quantities_changed(plan["items"], request["items"]):
                note = "Quantities were reduced to fit the stock and budget."

            answer = format_invoice(state["invoice"], "Invoice quote:", note)
            answer += (
                "\nInventory is unchanged until confirmation."
                "\nType 'confirm order' to buy or 'cancel order' to discard."
            )

            return {
                "answer": answer,
                "new_pending": {
                    "items": [dict(item) for item in plan["items"]],
                    "invoice": state["invoice"],
                    "budget": request.get("budget"),
                },
            }

        answer = state["reason"] + "\nNo new quote was prepared."

        if state["request"]["intent"] == "order":
            answer += " You can change the request or allow smaller quantities."

        if state.get("pending_order"):
            answer += "\nYour previous quote is still waiting for confirmation."

        return {"answer": answer}

    # ---------------- Graph routes ----------------

    # Route clarification requests to the end.
    def route_after_interpret(self, state):
        return "plan_step" if state["decision"] == "PLAN" else END

    # Route valid plans to the tools.
    def route_after_plan(self, state):
        return "act_step" if state["decision"] == "ACT" else "finish_step"

    # Route evaluation results to the next workflow step.
    def route_after_evaluate(self, state):
        if state["decision"] == "CONTINUE":
            return "act_step"
        if state["decision"] == "REPLAN":
            return "plan_step"
        return "finish_step"

    # Build and compile the workflow graph.
    def build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("interpret_step", self.interpret_node)
        graph.add_node("plan_step", self.plan_node)
        graph.add_node("act_step", self.act_node)
        graph.add_node("evaluate_step", self.evaluate_node)
        graph.add_node("finish_step", self.finish_node)

        graph.add_edge(START, "interpret_step")
        graph.add_conditional_edges("interpret_step", self.route_after_interpret)
        graph.add_conditional_edges("plan_step", self.route_after_plan)
        graph.add_edge("act_step", "evaluate_step")
        graph.add_conditional_edges("evaluate_step", self.route_after_evaluate)
        graph.add_edge("finish_step", END)

        return graph.compile()

    # ---------------- Commands and chat ----------------

    # Handle built-in commands without calling the model.
    def handle_command(self, command):
        if command == "reset":
            self.memory.clear()
            self.pending_order = None
            return "Conversation memory and pending order cleared."

        if command == "cancel order":
            if self.pending_order is None:
                return "There is no pending order to cancel."
            self.pending_order = None
            return "Pending order cancelled. Inventory is unchanged."

        if command == "confirm order":
            if self.pending_order is None:
                return "There is no pending order to confirm."

            pending = self.pending_order
            self.pending_order = None
            result = confirm_pending_order(pending)

            if not result["ok"]:
                return result["reason"]

            return format_invoice(result["invoice"], "Final invoice:") + \
                "\n" + result["reason"]

        return None

    # Handle a command or run the workflow for a chat message.
    def chat(self, message):
        command = " ".join(message.split()).casefold()

        try:
            answer = self.handle_command(command)

            if answer is not None:
                if command != "reset":
                    self.remember(message, answer)
                return answer

            final_state = self.graph.invoke(
                {
                    "message": message,
                    "memory": list(self.memory),
                    "pending_order": self.pending_order,
                },
                {"recursion_limit": 50},
            )

            if final_state.get("new_pending"):
                self.pending_order = final_state["new_pending"]

            answer = final_state["answer"]
            self.remember(message, answer)
            return answer

        except Exception as error:
            return (
                f"Operation failed: {error}\n"
                "Inventory is unchanged."
            )
