import json
from decimal import Decimal
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from inventory import decimal_value, find_product_name, read_inventory
from tools import (
    TOOLS,
    PlanningConstraintError,
    adjust_order_items,
    confirm_pending_order,
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


SYSTEM_PROMPT = """
You interpret messages for a fruit shop.
Return only one JSON object.

Choose one intent:
- inventory: the user wants the product list, prices, or stock
- product: the user asks about one product
- order: the user wants a new or changed invoice
- clarify: the message is unclear or not supported

JSON structure:
{
  "intent": "order",
  "product": null,
  "items": [{"name": "apple", "quantity_kg": "3"}],
  "budget": null,
  "allow_reduction": false,
  "question": null
}

Rules:
- Quantities and budget are decimal strings. Quantities are in kg.
- If the user gives pounds, convert to kg (1 lb = 0.45359237 kg).
- For orders, write the complete basket.
- For changes, use pending_order and conversation as context.
- Keep an earlier budget unless the user changes or removes it.
- Never calculate totals and never reduce quantities yourself.
- A low budget alone is NOT permission to reduce quantities.
- allow_reduction is true only if the user clearly allows smaller quantities.
- Never invent quantities. Never remove or replace fruits.
- If the budget currency is different from the catalog currency, choose clarify.
- If important details are missing, choose clarify and write a question.
- Always write questions in English.
"""


# AgentState
# All data that moves between the graph nodes.
# Each node reads some fields and returns only the fields it changes.
# Important: state keys must NOT have the same name as a graph node.
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


# quantities_changed
# Compares the planned basket with the requested basket as numbers.
# "3" and "3.00" count as the same amount.
def quantities_changed(planned_items, requested_items):
    requested = {
        item["name"]: decimal_value(item["quantity_kg"])
        for item in requested_items
    }

    for item in planned_items:
        if decimal_value(item["quantity_kg"]) != requested[item["name"]]:
            return True

    return False


# format_invoice
# Turns an invoice into readable text for the user.
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


# format_products
# Turns a list of products into readable text for the user.
def format_products(products, currency):
    return "\n".join(
        f"- {product['name']}: {product['price_per_kg']} {currency}/kg, "
        f"stock: {product['stock_kg']} kg"
        for product in products
    )


# FruitAgent
# Keeps the conversation memory and the pending order,
# and runs the LangGraph workflow for every normal message.
class FruitAgent:

    # __init__
    # Creates the LangChain model and builds the LangGraph graph once.
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.memory = []
        self.pending_order = None

        self.model = ChatOllama(
            model=MODEL,
            temperature=0,
            format="json",
        )

        self.graph = self.build_graph()

    # log
    # Prints one step of the agent when verbose mode is on.
    def log(self, label, value):
        if self.verbose:
            if not isinstance(value, str):
                value = json.dumps(value, indent=2)
            print(f"\n[{label}]\n{value}")

    # remember
    # Saves the last user message and answer.
    # Only the newest MEMORY_MESSAGES messages are kept.
    def remember(self, user_message, answer):
        self.memory.append({"role": "user", "content": user_message})
        self.memory.append({"role": "assistant", "content": answer})
        self.memory = self.memory[-MEMORY_MESSAGES:]

    # ask_model
    # Sends the system prompt and data to the model with LangChain.
    # It tries again if the model fails or returns broken JSON.
    def ask_model(self, payload):
        last_error = None

        for attempt in range(MAX_MODEL_RETRIES):
            try:
                response = self.model.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(payload)),
                ])
                return json.loads(response.content)

            except Exception as error:
                last_error = error
                self.log("MODEL RETRY", str(error))

        raise RuntimeError(f"The model failed: {last_error}")

    # check_request
    # Checks the model answer and cleans it.
    # - Fruit names are matched to the catalog right away.
    # - An unknown fruit becomes a friendly question.
    # - Duplicate fruits are added together.
    # - Numbers are checked with Decimal.
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

            # Duplicate fruits are added together.
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

    # interpret_node
    # Node 1: reads the inventory and asks the model what the user wants.
    # If the request is unclear, it writes a question and the graph stops.
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

    # plan_node
    # Node 2: creates a plan (which items and which tools).
    # First attempt: the original quantities.
    # Later attempts (REPLAN): smaller quantities from adjust_order_items,
    # but only if the user allowed reductions.
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

    # act_node
    # Node 3 (ACT + OBSERVE): runs the current tool and saves its result.
    # A tool error becomes a failed result instead of a crash.
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

    # evaluate_node
    # Node 4 (EVALUATE): looks at the tool result and chooses the next move:
    #   CONTINUE            -> run the next tool
    #   REPLAN              -> go back to plan_step with smaller quantities
    #   ASK_USER            -> stop and explain the problem
    #   FINISH              -> information is ready
    #   AWAIT_CONFIRMATION  -> the quote is ready
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

    # finish_node
    # Node 5: writes the final answer for the user.
    # A new pending order is created only when a quote succeeds.
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

        # ASK_USER or FAILED: the old pending order is kept.
        answer = state["reason"] + "\nNo new quote was prepared."

        if state["request"]["intent"] == "order":
            answer += " You can change the request or allow smaller quantities."

        if state.get("pending_order"):
            answer += "\nYour previous quote is still waiting for confirmation."

        return {"answer": answer}

    # ---------------- Graph routes ----------------

    # route_after_interpret
    # Stops the graph for clarify questions, otherwise goes to planning.
    def route_after_interpret(self, state):
        return "plan_step" if state["decision"] == "PLAN" else END

    # route_after_plan
    # Goes to the tools if the plan is valid, otherwise to the final answer.
    def route_after_plan(self, state):
        return "act_step" if state["decision"] == "ACT" else "finish_step"

    # route_after_evaluate
    # This is the ReAct / Replan loop:
    # CONTINUE goes back to act_step, REPLAN goes back to plan_step.
    def route_after_evaluate(self, state):
        if state["decision"] == "CONTINUE":
            return "act_step"
        if state["decision"] == "REPLAN":
            return "plan_step"
        return "finish_step"

    # build_graph
    # Connects all nodes and routes into one LangGraph workflow.
    # Node names end with "_step" so they never match a state key.
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

    # handle_command
    # Handles fixed commands without the model.
    # Returns None when the message is not a command.
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

    # chat
    # The entry point for every user message.
    # 1. Commands are handled directly.
    # 2. Other messages go through the LangGraph workflow.
    # 3. The pending order is replaced only by a successful new quote.
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
