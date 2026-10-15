import json
from decimal import Decimal

from calculator import calculate_invoice
from inventory import decimal_value, read_inventory, save_inventory


# Reduced quantities change in steps of 0.01 kg.
QUANTITY_STEP = Decimal("0.01")


# PlanningConstraintError
# A special error for orders that cannot be fixed,
# for example when the budget is too small for even 0.01 kg of each fruit.
class PlanningConstraintError(ValueError):
    pass


# list_inventory
# Tool: returns all products with prices and stock.
def list_inventory(state):
    return {
        "ok": True,
        "products": state["inventory"]["products"],
    }


# get_product
# Tool: returns one product.
# The name was already matched to the catalog in check_request.
def get_product(state):
    name = state["request"]["product"]

    for product in state["inventory"]["products"]:
        if product["name"] == name:
            return {"ok": True, "products": [product]}

    return {"ok": False, "reason": f"Unknown product: {name}."}


# find_stock_problems
# Helper: compares each basket item with the stock.
# It returns a list of problems. An empty list means everything is fine.
# Both check_stock and confirm_pending_order use this helper.
def find_stock_problems(items, products):
    stock = {
        product["name"]: decimal_value(product["stock_kg"])
        for product in products
    }

    problems = []

    for item in items:
        name = item["name"]
        quantity = decimal_value(item["quantity_kg"])

        if name not in stock:
            problems.append(f"Unknown product: {name}.")
        elif quantity > stock[name]:
            problems.append(
                f"{name}: requested {quantity} kg, "
                f"available {stock[name]} kg."
            )

    return problems


# check_stock
# Tool: checks that the store has enough of every fruit in the plan.
def check_stock(state):
    problems = find_stock_problems(
        state["plan"]["items"],
        state["inventory"]["products"],
    )

    if problems:
        return {"ok": False, "reason": " ".join(problems)}

    return {"ok": True, "reason": "Stock check passed."}


# calculate_total
# Tool: calculates the invoice for the planned basket.
def calculate_total(state):
    inventory = state["inventory"]

    invoice = calculate_invoice(
        state["plan"]["items"],
        inventory["products"],
        inventory["currency"],
    )

    return {"ok": True, "invoice": invoice}


# prepare_invoice
# Tool: makes sure an invoice exists before it is shown to the user.
def prepare_invoice(state):
    if state.get("invoice") is None:
        return {"ok": False, "reason": "The invoice must be calculated first."}

    return {"ok": True, "invoice": state["invoice"]}


# adjust_order_items
# Replanning helper: makes the basket smaller so it fits the stock and budget.
# Rules:
#   1. Every requested fruit stays in the basket (at least 0.01 kg).
#   2. No fruit gets more than requested or more than the stock.
#   3. All fruits shrink together, so the basket keeps its shape.
#   4. A binary search finds the largest basket that fits the budget.
#   5. Every total is checked with the real calculator.
def adjust_order_items(items, products, budget):
    stock = {
        product["name"]: decimal_value(product["stock_kg"])
        for product in products
    }

    # Find the largest allowed amount of each fruit, counted in 0.01 kg units.
    limits = []

    for item in items:
        name = item["name"]
        largest = min(decimal_value(item["quantity_kg"]), stock[name])
        units = int(largest / QUANTITY_STEP)

        if units < 1:
            raise PlanningConstraintError(
                f"Cannot keep {name}: there is less than 0.01 kg available."
            )

        limits.append((name, units))

    top_level = max(units for _, units in limits)

    # basket_at
    # Builds a basket for a level between 0 and top_level.
    # Level 0 gives 0.01 kg of each fruit.
    # Level top_level gives the largest allowed amount of each fruit.
    def basket_at(level):
        basket = []

        for name, units in limits:
            amount = 1 + (units - 1) * level // top_level
            basket.append({
                "name": name,
                "quantity_kg": str(amount * QUANTITY_STEP),
            })

        return basket

    # total_of
    # Returns the real invoice total of a basket.
    def total_of(basket):
        invoice = calculate_invoice(basket, products, "")
        return decimal_value(invoice["total"])

    largest_basket = basket_at(top_level)

    if budget is None:
        return largest_basket

    budget_value = decimal_value(budget)

    if total_of(largest_basket) <= budget_value:
        return largest_basket

    smallest_basket = basket_at(0)
    smallest_total = total_of(smallest_basket)

    if smallest_total > budget_value:
        raise PlanningConstraintError(
            f"Your budget of {budget_value} is too low. "
            f"Even 0.01 kg of each fruit costs {smallest_total}. "
            "Please increase the budget or choose fewer fruits."
        )

    # Binary search for the largest basket inside the budget.
    low = 0
    high = top_level
    best = smallest_basket

    while low <= high:
        middle = (low + high) // 2
        candidate = basket_at(middle)

        if total_of(candidate) <= budget_value:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1

    return best


# The tools the agent is allowed to run.
TOOLS = {
    "list_inventory": list_inventory,
    "get_product": get_product,
    "check_stock": check_stock,
    "calculate_total": calculate_total,
    "prepare_invoice": prepare_invoice,
}


# confirm_pending_order
# Runs only after the user types "confirm order".
# The model can never call this function.
# It reads the newest inventory and checks the stock, prices, currency,
# and budget again. Only then does it reduce the stock and save the file.
def confirm_pending_order(pending):
    inventory = read_inventory()
    products = inventory["products"]

    problems = find_stock_problems(pending["items"], products)

    if problems:
        return {
            "ok": False,
            "reason": "Inventory changed. " + " ".join(problems)
                      + " Please request a new quote.",
        }

    current_invoice = calculate_invoice(
        pending["items"],
        products,
        inventory["currency"],
    )

    if current_invoice != pending["invoice"]:
        return {
            "ok": False,
            "reason": "Prices or currency changed. Please request a new quote.",
        }

    budget = pending.get("budget")

    if budget is not None and decimal_value(current_invoice["total"]) > decimal_value(budget):
        return {
            "ok": False,
            "reason": "The invoice is above your budget. Please request a new quote.",
        }

    for product in products:
        for item in pending["items"]:
            if item["name"] == product["name"]:
                remaining = (
                    decimal_value(product["stock_kg"])
                    - decimal_value(item["quantity_kg"])
                )
                product["stock_kg"] = str(remaining)

    save_inventory(inventory)

    return {
        "ok": True,
        "invoice": current_invoice,
        "reason": "Order confirmed. Inventory updated.",
    }


# ============================================================
# TOOL SCHEMAS
# ------------------------------------------------------------
# The language model reads these schemas and decides which tool
# to call and which arguments to send.
#
# Safety rules:
# - The model only CHOOSES a tool and fills its arguments.
# - The model never calculates prices, totals, or stock.
# - There is no schema for confirm_pending_order, so the model
#   can never confirm an order. Only "confirm order" can do that.
# - There is no schema for adjust_order_items, so the model can
#   never reduce quantities. Replanning is done by Python code only.
# ============================================================


# TOOL_SCHEMAS
# The tools the model can choose from, in the Ollama / OpenAI
# "function calling" format.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_inventory",
            "description": (
                "Show the full product list with the price per kg and the "
                "stock in kg. Use this when the user asks what is available, "
                "or asks for all prices or all stock."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": (
                "Show the price and stock of ONE product. Use this when the "
                "user asks about a single fruit without ordering it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The fruit name, for example \"apple\".",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quote_order",
            "description": (
                "Create a new or changed invoice quote for a basket of fruits. "
                "This starts the order workflow: check_stock -> "
                "calculate_total -> prepare_invoice. "
                "Always send the COMPLETE basket in one call. "
                "Never calculate totals and never reduce quantities yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "The complete basket. Each fruit appears once.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The fruit name, for example \"banana\".",
                                },
                                "quantity_kg": {
                                    "type": "string",
                                    "description": (
                                        "Quantity in kg as a decimal string, "
                                        "for example \"2.5\". Convert pounds to kg "
                                        "(1 lb = 0.45359237 kg)."
                                    ),
                                },
                            },
                            "required": ["name", "quantity_kg"],
                        },
                    },
                    "budget": {
                        "type": "string",
                        "description": (
                            "Optional budget in the catalog currency as a decimal "
                            "string, for example \"5\". Leave it out if there is "
                            "no budget."
                        ),
                    },
                    "allow_reduction": {
                        "type": "boolean",
                        "description": (
                            "true ONLY if the user clearly allows smaller "
                            "quantities. A low budget alone is NOT permission."
                        ),
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": (
                "Ask the user a question when the message is unclear, "
                "important details are missing, the request is not supported, "
                "or the budget currency is different from the catalog currency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A short and friendly question in English.",
                    },
                },
                "required": ["question"],
            },
        },
    },
]


# TOOL_INTENTS
# Connects each model tool to an intent that agent.py already understands.
# agent.py then runs the existing steps from STEPS_BY_INTENT.
TOOL_INTENTS = {
    "list_inventory": "inventory",
    "get_product": "product",
    "quote_order": "order",
    "ask_clarification": "clarify",
}


# SCHEMAS_BY_NAME
# A quick lookup: tool name -> tool schema.
SCHEMAS_BY_NAME = {
    schema["function"]["name"]: schema
    for schema in TOOL_SCHEMAS
}


# load_json_text
# Small models sometimes send a list or an object as a JSON string.
# This helper turns that string back into real data.
def load_json_text(text, path):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"{path} must be valid JSON data.")


# coerce_value
# Checks one value against its schema and fixes small, safe mistakes:
# - a number where a string is expected becomes a string (3 -> "3")
# - "true" / "false" text becomes a real boolean
# - a JSON string where a list or object is expected is decoded
# - a single object where a list is expected becomes a list
# Anything else that does not match raises a ValueError.
def coerce_value(value, schema, path):
    expected = schema.get("type")

    if expected == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise ValueError(f"{path} must be a string.")

    if expected == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().casefold() in ("true", "false"):
            return value.strip().casefold() == "true"
        raise ValueError(f"{path} must be true or false.")

    if expected == "array":
        if isinstance(value, str):
            value = load_json_text(value, path)
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list.")

        item_schema = schema.get("items", {})
        return [
            coerce_value(item, item_schema, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if expected == "object":
        if isinstance(value, str):
            value = load_json_text(value, path)
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object.")
        return coerce_object(value, schema, path)

    return value


# coerce_object
# Checks an object against its schema:
# - every required key must exist and must not be null
# - every known key is checked with coerce_value
# - unknown keys are dropped
def coerce_object(value, schema, path):
    properties = schema.get("properties", {})

    for key in schema.get("required", []):
        if value.get(key) is None:
            raise ValueError(f"{path} needs the argument '{key}'.")

    cleaned = {}

    for key, key_schema in properties.items():
        if value.get(key) is not None:
            cleaned[key] = coerce_value(value[key], key_schema, f"{path}.{key}")

    return cleaned


# validate_tool_arguments
# Checks the tool name and the arguments that the model sent.
# Returns clean arguments, or raises a ValueError that is sent back
# to the model so it can try again.
def validate_tool_arguments(name, arguments):
    schema = SCHEMAS_BY_NAME.get(name)

    if schema is None:
        allowed = ", ".join(SCHEMAS_BY_NAME)
        raise ValueError(f"Unknown tool: {name}. Allowed tools: {allowed}.")

    if arguments is None:
        arguments = {}

    if isinstance(arguments, str):
        arguments = load_json_text(arguments, name)

    if not isinstance(arguments, dict):
        raise ValueError(f"The arguments of {name} must be an object.")

    return coerce_object(arguments, schema["function"]["parameters"], name)


# parse_tool_call_text
# Backup plan: small models sometimes write the tool call as text
# instead of a real tool call, for example:
#   <tool_call>{"name": "list_inventory", "arguments": {}}</tool_call>
# This helper reads that text and returns (name, arguments).
def parse_tool_call_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The model did not call a tool.")

    cleaned = text.strip()

    for tag in ("<tool_call>", "</tool_call>", "```json", "```"):
        cleaned = cleaned.replace(tag, "")

    try:
        data = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        raise ValueError("The model answered with text instead of calling a tool.")

    if not isinstance(data, dict) or "name" not in data:
        raise ValueError("The model answered with text instead of calling a tool.")

    arguments = data.get("arguments", data.get("parameters", {}))
    return data["name"], arguments


# tool_call_to_request
# Turns a checked tool call into the same request dictionary that
# check_request in agent.py already expects.
# Because of this, planning, tools, and replanning work exactly as before.
def tool_call_to_request(name, arguments):
    return {
        "intent": TOOL_INTENTS[name],
        "product": arguments.get("product"),
        "items": arguments.get("items"),
        "budget": arguments.get("budget"),
        "allow_reduction": arguments.get("allow_reduction", False),
        "question": arguments.get("question"),
    }
