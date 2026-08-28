import json
from pathlib import Path


# Find the project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Location of our product catalog
PRODUCT_FILE = BASE_DIR / "data" / "products.json"


def load_products():
    """Load all products from the JSON catalog."""

    with open(PRODUCT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def search_products(category=None, max_price=None, preferences=None):
    """Find products matching category and budget, then score preferences."""

    products = load_products()

    if preferences is None:
        preferences = []

    preferences = [
        preference.lower().strip()
        for preference in preferences
    ]

    results = []

    for product in products:

        # 1. Category is a hard filter
        if category:
            if category.lower() not in product["category"].lower():
                continue

        # 2. Budget is a hard filter
        if max_price is not None:
            if product["price"] > max_price:
                continue

        # 3. Preferences are soft matching
        product_features = [
            feature.lower()
            for feature in product["features"]
        ]

        matched_preferences = []

        for preference in preferences:
            for feature in product_features:
                if preference in feature or feature in preference:
                    matched_preferences.append(preference)
                    break

        # Calculate a simple recommendation score
        preference_score = len(matched_preferences)

        product_result = product.copy()

        product_result["preference_score"] = preference_score
        product_result["matched_preferences"] = matched_preferences

        results.append(product_result)

    # Highest preference match first
    # Then highest rating
    results.sort(
        key=lambda product: (
            product["preference_score"],
            product["rating"]
        ),
        reverse=True
    )

    return results

if __name__ == "__main__":

    print("PAYPILOT PRODUCT SEARCH")
    print("-----------------------")

    products = search_products(
        category="headphones",
        max_price=5000
    )

    for product in products:
        print(
            f'{product["name"]} - '
            f'₹{product["price"]} - '
            f'Rating: {product["rating"]}'
        )