import json
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


INVENTORY_PATH = Path(__file__).with_name("fruits.json")


def decimal_value(value):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Invalid decimal value: {value}")

    if not number.is_finite():
        raise ValueError("Decimal values must be finite.")

    return number


def read_inventory():
    with INVENTORY_PATH.open("r", encoding="utf-8") as file:
        products = json.load(file)

    if not isinstance(products, list):
        raise ValueError("Inventory must be a JSON list.")

    names = set()

    for product in products:
        if not isinstance(product, dict):
            raise ValueError("Each product must be an object.")

        name = product.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each product needs a valid name.")

        if name in names:
            raise ValueError(f"Duplicate product: {name}")

        names.add(name)

        price = decimal_value(product.get("price_per_kg"))
        stock = decimal_value(product.get("stock_kg"))

        if price < 0 or stock < 0:
            raise ValueError("Prices and stock cannot be negative.")

    return products


def save_inventory(products):
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=INVENTORY_PATH.parent,
            prefix="inventory_",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(products, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, INVENTORY_PATH)

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()