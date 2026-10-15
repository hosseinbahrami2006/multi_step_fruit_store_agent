import json
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


# The inventory file is stored next to this Python file.
INVENTORY_FILE = Path(__file__).resolve().parent / "fruits.json"


# decimal_value
# Turns a value such as "2.50" or 3 into an exact Decimal number.
# It rejects empty values, True/False, words, and infinity.
# We use Decimal because normal floats make small money mistakes.
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


# normalize_name
# Cleans a name so two spellings can be compared.
# It removes extra spaces and ignores upper and lower case.
# Example: "  Green   Apple " becomes "green apple".
def normalize_name(name):
    if not isinstance(name, str):
        return ""

    return " ".join(name.split()).casefold()


# find_product_name
# Finds the real catalog name for a name written by the user or model.
# It also understands simple plural forms:
#   "apples" -> "apple"
#   "peaches" -> "peach"
#   "strawberries" -> "strawberry"
# It returns None when the fruit is not in the catalog.
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


# validate_inventory
# Checks that the inventory data is safe to use.
# It checks the currency, the unit, unique names,
# positive prices, and stock that is not negative.
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


# read_inventory
# Opens fruits.json, checks it, and returns the data.
def read_inventory():
    with INVENTORY_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    validate_inventory(data)
    return data


# save_inventory
# Saves the inventory safely.
# It first writes a temporary file and then replaces the real file.
# If the program crashes while writing, the real file is not broken.
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
