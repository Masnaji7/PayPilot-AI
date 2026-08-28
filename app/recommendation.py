def rank_products(products, preferences=None, max_price=None):
    """
    Rank products using rating, budget fit, and preference matches.
    """

    if not products:
        return []

    if preferences is None:
        preferences = []

    preferences = [
        preference.lower().strip()
        for preference in preferences
    ]

    ranked_products = []

    for product in products:

        # -----------------------------
        # 1. Rating score
        # -----------------------------
        rating_score = (product["rating"] / 5) * 60

        # -----------------------------
        # 2. Preference score
        # -----------------------------
        features = [
            feature.lower()
            for feature in product["features"]
        ]

        matched_preferences = []

        for preference in preferences:
            for feature in features:
                if preference in feature or feature in preference:
                    matched_preferences.append(preference)
                    break

        if preferences:
            preference_score = (
                len(matched_preferences) / len(preferences)
            ) * 25
        else:
            preference_score = 0

        # -----------------------------
        # 3. Budget/value score
        # -----------------------------
        budget_score = 0

        if max_price is not None and max_price > 0:
            price_ratio = product["price"] / max_price

            if price_ratio <= 0.6:
                budget_score = 15
            elif price_ratio <= 0.8:
                budget_score = 12
            elif price_ratio <= 1.0:
                budget_score = 9

        # -----------------------------
        # Final score
        # -----------------------------
        final_score = (
            rating_score
            + preference_score
            + budget_score
        )

        product_copy = product.copy()

        product_copy["recommendation_score"] = round(
            final_score, 2
        )

        product_copy["matched_preferences"] = matched_preferences

        ranked_products.append(product_copy)

    # Highest score first
    ranked_products.sort(
        key=lambda product: product["recommendation_score"],
        reverse=True
    )

    return ranked_products