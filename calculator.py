from decimal import Decimal, ROUND_HALF_UP

from inventory import decimal_value


def calculate_invoice(items, products):
    catalog = {product["name"]: product for product in products}

    lines = []
    total = Decimal("0")

    for item in items:
        name = item["name"]

        if name not in catalog:
            raise ValueError(f"Unknown product: {name}")

        quantity = decimal_value(item["quantity_kg"])
        price = decimal_value(catalog[name]["price_per_kg"])

        if quantity <= 0:
            raise ValueError("Order quantities must be positive.")

        line_total = (quantity * price).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        total += line_total

        lines.append({
            "name": name,
            "quantity_kg": str(quantity),
            "price_per_kg": str(price),
            "line_total": str(line_total),
        })

    return {
        "items": lines,
        "total": str(total),
    }