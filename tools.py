from calculator import calculate_invoice
from inventory import decimal_value, read_inventory, save_inventory


def list_inventory(state):
    return {
        "ok": True,
        "products": state["catalog"],
    }


def get_product(state):
    name = state["request"]["product"]

    for product in state["catalog"]:
        if product["name"] == name:
            return {
                "ok": True,
                "product": product,
            }

    return {
        "ok": False,
        "reason": f"Unknown product: {name}",
    }


def check_stock(state):
    catalog = {
        product["name"]: product
        for product in state["catalog"]
    }

    issues = []

    for item in state["plan"]["items"]:
        name = item["name"]
        quantity = decimal_value(item["quantity_kg"])

        if name not in catalog:
            issues.append(f"Unknown product: {name}")
            continue

        available = decimal_value(catalog[name]["stock_kg"])

        if quantity > available:
            issues.append(
                f"{name}: requested {quantity} kg; "
                f"available {available} kg."
            )

    return {
        "ok": not issues,
        "reason": "; ".join(issues) if issues else "Stock check passed.",
    }


def calculate_total(state):
    state["invoice"] = calculate_invoice(
        state["plan"]["items"],
        state["catalog"],
    )

    return {
        "ok": True,
        "invoice": state["invoice"],
    }


def prepare_invoice(state):
    if state["invoice"] is None:
        return {
            "ok": False,
            "reason": "An invoice must be calculated first.",
        }

    return {
        "ok": True,
        "invoice": state["invoice"],
    }


TOOLS = {
    "list_inventory": list_inventory,
    "get_product": get_product,
    "check_stock": check_stock,
    "calculate_total": calculate_total,
    "prepare_invoice": prepare_invoice,
}


def confirm_pending_order(pending):
    """
    Called by application code after an explicit confirmation command.
    Not available to the LLM.
    """
    products = read_inventory()

    temporary_state = {
        "catalog": products,
        "plan": {"items": pending["items"]},
    }

    stock_result = check_stock(temporary_state)

    if not stock_result["ok"]:
        return {
            "ok": False,
            "reason": (
                "Inventory changed. Please request a new invoice. "
                + stock_result["reason"]
            ),
        }

    current_invoice = calculate_invoice(pending["items"], products)

    # Compare the entire invoice, not just the total.
    if current_invoice != pending["invoice"]:
        return {
            "ok": False,
            "reason": (
                "Prices changed. Please request a new invoice "
                "and confirm it again."
            ),
        }

    catalog = {product["name"]: product for product in products}

    for item in pending["items"]:
        product = catalog[item["name"]]

        remaining = (
            decimal_value(product["stock_kg"])
            - decimal_value(item["quantity_kg"])
        )

        product["stock_kg"] = str(remaining)

    save_inventory(products)

    return {
        "ok": True,
        "reason": "Order confirmed. Inventory updated.",
    }