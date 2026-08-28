import json
import requests

from app.product_search import search_products
from app.recommendation import rank_products
from app.response_generator import generate_customer_response


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"


def extract_customer_requirements(user_message):
    """
    Use the local Qwen model to understand the customer's
    shopping requirements.
    """

    prompt = f"""
You are the requirement extraction component of PayPilot AI.

Read the customer's shopping request and return ONLY valid JSON.

The JSON must contain exactly these fields:

{{
    "category": "product category or null",
    "max_price": number or null,
    "preferences": ["preference1", "preference2"]
}}

Available product categories are:
- headphones
- laptop accessories
- computer accessories
- chargers
- running shoes
- sports accessories

Important rules:

1. Use "headphones" when the customer asks for:
   - headphones
   - earphones
   - headsets

2. Never use broad categories such as:
   - electronics
   - technology
   - gadgets

3. Choose only one category from the available categories.

4. If no category matches, use null.

5. Extract the customer's actual preferences.

Examples of preferences:
- wireless
- comfortable
- noise cancellation
- long battery
- lightweight
- portable
- fast charging
- ergonomic
- breathable
- studying
- gaming
- running
- travel

6. If the customer mentions "wireless headphones",
   include "wireless" in preferences.

7. If the customer mentions "for studying",
   include "studying" in preferences.

8. If the customer does not mention a budget,
   use null for max_price.

9. Do not invent products.

10. Return JSON only.
Do not include explanations.
Do not include markdown.
Do not include code fences.

Customer request:
{user_message}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    result = response.json()["response"].strip()

    # Remove markdown code fences if the model adds them
    result = result.replace("```json", "")
    result = result.replace("```", "")
    result = result.strip()

    return json.loads(result)


def create_recommendation(user_message):
    """
    Complete PayPilot AI recommendation workflow:

    1. Understand customer request
    2. Extract requirements
    3. Search product catalog
    4. Rank products
    5. Generate human-friendly recommendation
    """

    # -----------------------------------------
    # STEP 1: Extract customer requirements
    # -----------------------------------------

    requirements = extract_customer_requirements(
        user_message
    )

    # -----------------------------------------
    # STEP 2: Search product catalog
    # -----------------------------------------

    products = search_products(
        category=requirements["category"],
        max_price=requirements["max_price"],
        preferences=requirements["preferences"]
    )

    # -----------------------------------------
    # STEP 3: Rank matching products
    # -----------------------------------------

    products = rank_products(
        products,
        preferences=requirements["preferences"],
        max_price=requirements["max_price"]
    )

    # -----------------------------------------
    # STEP 4: Generate customer-facing response
    # -----------------------------------------

    customer_response = generate_customer_response(
        user_message,
        requirements,
        products
    )

    # -----------------------------------------
    # STEP 5: Return final result
    # -----------------------------------------

    return {
        "requirements": requirements,
        "products": products,
        "message": customer_response
    }


def print_requirements(requirements):
    """
    Display the extracted customer requirements.
    """

    print("\nCUSTOMER REQUIREMENTS")
    print("---------------------")

    print(
        json.dumps(
            requirements,
            indent=2,
            ensure_ascii=False
        )
    )


def print_products(products):
    """
    Display matching products.
    """

    print("\nMATCHING PRODUCTS")
    print("-----------------")

    if not products:
        print("No matching products found.")
        return

    for index, product in enumerate(
        products,
        start=1
    ):

        print(
            f"{index}. "
            f'{product["name"]} - '
            f'₹{product["price"]} - '
            f'Rating: {product["rating"]} - '
            f'Score: {product["recommendation_score"]}'
        )

        matched_preferences = product.get(
            "matched_preferences",
            []
        )

        if matched_preferences:

            print(
                "   Matched preferences: "
                + ", ".join(
                    matched_preferences
                )
            )


def print_best_recommendation(products):
    """
    Display the highest-ranked product.
    """

    if not products:
        return

    best_product = products[0]

    print("\nBEST RECOMMENDATION")
    print("-------------------")

    print(
        f'🏆 {best_product["name"]}'
    )

    print(
        f'Price: ₹{best_product["price"]}'
    )

    print(
        f'Rating: ⭐ {best_product["rating"]}'
    )

    print(
        f'Score: {best_product["recommendation_score"]}'
    )


def print_recommendation_reasons(
    requirements,
    products
):
    """
    Explain why the highest-ranked product
    is a good recommendation.
    """

    if not products:
        return

    best_product = products[0]

    print("\nWHY THIS PRODUCT?")
    print("-----------------")

    reasons = []

    # Budget reason
    max_price = requirements.get(
        "max_price"
    )

    if max_price is not None:

        if best_product["price"] <= max_price:

            reasons.append(
                "It fits within your budget."
            )

    # Rating reason
    if best_product["rating"] >= 4.5:

        reasons.append(
            "It has a strong customer rating."
        )

    # Preference reason
    matched_preferences = best_product.get(
        "matched_preferences",
        []
    )

    if matched_preferences:

        reasons.append(
            "It matches your preferences: "
            + ", ".join(
                matched_preferences
            )
            + "."
        )

    # Fallback
    if not reasons:

        reasons.append(
            "It has the highest recommendation "
            "score among the matching products."
        )

    for reason in reasons:

        print(
            f"- {reason}"
        )


def main():

    print("\n==============================")
    print("          PAYPILOT AI")
    print("==============================")

    user_message = input(
        "\nWhat are you looking for?\n> "
    ).strip()

    if not user_message:

        print(
            "\nPlease enter a shopping request."
        )

        return

    try:

        # -----------------------------------------
        # RUN PAYPILOT AI
        # -----------------------------------------

        result = create_recommendation(
            user_message
        )

        # -----------------------------------------
        # SHOW CUSTOMER REQUIREMENTS
        # -----------------------------------------

        print_requirements(
            result["requirements"]
        )

        # -----------------------------------------
        # SHOW PRODUCTS
        # -----------------------------------------

        print_products(
            result["products"]
        )

        # -----------------------------------------
        # SHOW BEST PRODUCT
        # -----------------------------------------

        print_best_recommendation(
            result["products"]
        )

        # -----------------------------------------
        # SHOW REASONS
        # -----------------------------------------

        print_recommendation_reasons(
            result["requirements"],
            result["products"]
        )

        # -----------------------------------------
        # SHOW HUMAN-FRIENDLY AI RESPONSE
        # -----------------------------------------

        print("\nPAYPILOT AI RESPONSE")
        print("--------------------")

        print(
            result["message"]
        )

    # -----------------------------------------
    # JSON ERROR
    # -----------------------------------------

    except json.JSONDecodeError:

        print("\nERROR")
        print("-----")

        print(
            "The AI returned an invalid JSON response."
        )

        print(
            "Please run the program again."
        )

    # -----------------------------------------
    # OLLAMA CONNECTION ERROR
    # -----------------------------------------

    except requests.exceptions.ConnectionError:

        print("\nERROR")
        print("-----")

        print(
            "Could not connect to Ollama."
        )

        print(
            "Please make sure Ollama is running."
        )

    # -----------------------------------------
    # TIMEOUT ERROR
    # -----------------------------------------

    except requests.exceptions.Timeout:

        print("\nERROR")
        print("-----")

        print(
            "Ollama took too long to respond."
        )

        print(
            "Please try again."
        )

    # -----------------------------------------
    # OTHER ERRORS
    # -----------------------------------------

    except Exception as error:

        print("\nERROR")
        print("-----")

        print(
            f"{type(error).__name__}: {error}"
        )


# ---------------------------------------------
# PROGRAM START
# ---------------------------------------------

if __name__ == "__main__":
    main()