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
