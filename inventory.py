import json
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


# Store inventory beside this file.
INVENTORY_FILE = Path(__file__).resolve().parent / "fruits.json"


# Convert a value to a finite Decimal.
def decimal_value(value):
    if value is None or isinstance(value, bool):
        raise ValueError("A valid number is required.")

    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Invalid number: {value}")

    if not number.is_finite():
        raise ValueError("Numbers must be finite.")

    return number


# Normalize a name for comparison.
def normalize_name(name):
    if not isinstance(name, str):
        return ""

    return " ".join(name.split()).casefold()


# Match a product name, including simple plural forms.
def find_product_name(name, products):
    wanted = normalize_name(name)

    catalog_names = {
        normalize_name(product["name"]): product["name"]
        for product in products
    }

    candidates = [wanted]

    if wanted.endswith("ies"):
        candidates.append(wanted[:-3] + "y")

    if wanted.endswith("es"):
        candidates.append(wanted[:-2])

    if wanted.endswith("s"):
        candidates.append(wanted[:-1])

    for candidate in candidates:
        if candidate in catalog_names:
            return catalog_names[candidate]

    return None


# Validate inventory fields and values.
def validate_inventory(data):
    if not isinstance(data, dict):
        raise ValueError("The inventory must be a JSON object.")

    currency = data.get("currency")

    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("The inventory needs a currency.")

    if data.get("quantity_unit") != "kg":
        raise ValueError("The quantity unit must be kg.")

    products = data.get("products")

    if not isinstance(products, list) or not products:
        raise ValueError("The inventory needs a list of products.")

    names = set()

    for product in products:
        if not isinstance(product, dict):
            raise ValueError("Each product must be an object.")

        name = normalize_name(product.get("name"))

        if not name or name in names:
            raise ValueError("Product names must be valid and unique.")

        names.add(name)

        if decimal_value(product.get("price_per_kg")) <= 0:
            raise ValueError(f"The price of {name} must be positive.")

        if decimal_value(product.get("stock_kg")) < 0:
            raise ValueError(f"The stock of {name} cannot be negative.")


# Read and validate the inventory file.
def read_inventory():
    with INVENTORY_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    validate_inventory(data)
    return data


# Save inventory through a temporary file.
def save_inventory(data):
    validate_inventory(data)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=INVENTORY_FILE.parent,
            prefix="inventory_",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, INVENTORY_FILE)

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
