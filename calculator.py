from decimal import Decimal, ROUND_HALF_UP

from inventory import decimal_value


# Round money to cents.
MONEY_STEP = Decimal("0.01")


# Round a number to two decimal places.
def money_value(value):
    return decimal_value(value).quantize(MONEY_STEP, rounding=ROUND_HALF_UP)


# Build an invoice and reject invalid items.
def calculate_invoice(items, products, currency):
    prices = {
        product["name"]: decimal_value(product["price_per_kg"])
        for product in products
    }

    if not isinstance(items, list) or not items:
        raise ValueError("An invoice needs at least one item.")

    lines = []
    total = Decimal("0")
    seen = set()

    for item in items:
        name = item["name"]

        if name not in prices:
            raise ValueError(f"Unknown product: {name}.")

        if name in seen:
            raise ValueError(f"Duplicate product: {name}.")

        seen.add(name)

        quantity = decimal_value(item["quantity_kg"])

        if quantity <= 0:
            raise ValueError("Quantities must be positive.")

        line_total = money_value(quantity * prices[name])
        total += line_total

        lines.append({
            "name": name,
            "quantity_kg": str(quantity),
            "price_per_kg": str(prices[name]),
            "line_total": str(line_total),
        })

    return {
        "currency": currency,
        "items": lines,
        "total": str(money_value(total)),
    }
