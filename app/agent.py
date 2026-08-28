import json
import requests
import os

from dotenv import load_dotenv

from app.product_search import search_products
from app.recommendation import rank_products
from app.response_generator import generate_customer_response


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Use Render environment variable if available.
# Otherwise use a stable default model.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)


# ============================================================
# GEMINI REQUEST FUNCTION
# ============================================================

def call_gemini(prompt, temperature=0):
    """
    Send a prompt to Gemini and return the generated text.
    """

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not configured."
        )

    response = requests.post(
        GEMINI_URL,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        },
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature
            }
        },
        timeout=120
    )

    # Show useful Gemini error instead of only "500 Internal Server Error"
    if not response.ok:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    try:
        return (
            data["candidates"][0]
            ["content"]
            ["parts"][0]
            ["text"]
            .strip()
        )
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected Gemini response: {data}"
        )


# ============================================================
# EXTRACT CUSTOMER REQUIREMENTS
# ============================================================

def extract_customer_requirements(user_message):
    """
    Use Gemini to understand the customer's shopping requirements.
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

    # Try the configured model first.
    # If Gemini temporarily returns 503, try a fallback model.
    models_to_try = [
        GEMINI_MODEL,
        "gemini-2.5-flash"
    ]

    # Remove duplicate model names
    models_to_try = list(dict.fromkeys(models_to_try))

    last_error = None

    for model in models_to_try:

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent"
        )

        try:

            response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY
                },
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0,
                        "responseMimeType": "application/json"
                    }
                },
                timeout=120
            )

            if response.ok:

                data = response.json()

                try:
                    result = (
                        data["candidates"][0]
                        ["content"]
                        ["parts"][0]
                        ["text"]
                    )

                    result = result.strip()

                    return json.loads(result)

                except (KeyError, IndexError, TypeError) as error:

                    last_error = RuntimeError(
                        f"Unexpected Gemini response: {data}"
                    )

                except json.JSONDecodeError as error:

                    last_error = RuntimeError(
                        f"Gemini returned invalid JSON: {result}"
                    )

            else:

                last_error = RuntimeError(
                    f"Gemini model {model} returned "
                    f"{response.status_code}: {response.text}"
                )

                # Try fallback model for server errors
                if response.status_code in [429, 500, 502, 503, 504]:
                    continue

                raise last_error

        except requests.exceptions.Timeout as error:

            last_error = RuntimeError(
                f"Gemini request timed out for model {model}."
            )

            continue

        except requests.exceptions.ConnectionError as error:

            last_error = RuntimeError(
                "Could not connect to Gemini API."
            )

            continue

    raise last_error or RuntimeError(
        "Gemini API request failed."
    )


# ============================================================
# CREATE RECOMMENDATION
# ============================================================

def create_recommendation(user_message):
    """
    Complete PayPilot AI recommendation workflow:

    1. Understand customer request
    2. Extract requirements
    3. Search product catalog
    4. Rank products
    5. Generate human-friendly recommendation
    """

    # --------------------------------------------------------
    # STEP 1: Extract customer requirements
    # --------------------------------------------------------

    requirements = extract_customer_requirements(
        user_message
    )

    # --------------------------------------------------------
    # STEP 2: Search product catalog
    # --------------------------------------------------------

    products = search_products(
        category=requirements["category"],
        max_price=requirements["max_price"],
        preferences=requirements["preferences"]
    )

    # --------------------------------------------------------
    # STEP 3: Rank matching products
    # --------------------------------------------------------

    products = rank_products(
        products,
        preferences=requirements["preferences"],
        max_price=requirements["max_price"]
    )

    # --------------------------------------------------------
    # STEP 4: Generate customer-facing response
    # --------------------------------------------------------

    customer_response = generate_customer_response(
        user_message,
        requirements,
        products
    )

    # --------------------------------------------------------
    # STEP 5: Return final result
    # --------------------------------------------------------

    return {
        "requirements": requirements,
        "products": products,
        "message": customer_response
    }


# ============================================================
# PRINT REQUIREMENTS
# ============================================================

def print_requirements(requirements):

    print("\nCUSTOMER REQUIREMENTS")
    print("---------------------")

    print(
        json.dumps(
            requirements,
            indent=2,
            ensure_ascii=False
        )
    )


# ============================================================
# PRINT PRODUCTS
# ============================================================

def print_products(products):

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


# ============================================================
# PRINT BEST RECOMMENDATION
# ============================================================

def print_best_recommendation(products):

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


# ============================================================
# PRINT RECOMMENDATION REASONS
# ============================================================

def print_recommendation_reasons(
    requirements,
    products
):

    if not products:
        return

    best_product = products[0]

    print("\nWHY THIS PRODUCT?")
    print("-----------------")

    reasons = []

    # --------------------------------------------------------
    # Budget reason
    # --------------------------------------------------------

    max_price = requirements.get(
        "max_price"
    )

    if max_price is not None:

        if best_product["price"] <= max_price:

            reasons.append(
                "It fits within your budget."
            )

    # --------------------------------------------------------
    # Rating reason
    # --------------------------------------------------------

    if best_product["rating"] >= 4.5:

        reasons.append(
            "It has a strong customer rating."
        )

    # --------------------------------------------------------
    # Preference reason
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not reasons:

        reasons.append(
            "It has the highest recommendation "
            "score among the matching products."
        )

    for reason in reasons:

        print(
            f"- {reason}"
        )


# ============================================================
# MAIN
# ============================================================

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

        result = create_recommendation(
            user_message
        )

        print_requirements(
            result["requirements"]
        )

        print_products(
            result["products"]
        )

        print_best_recommendation(
            result["products"]
        )

        print_recommendation_reasons(
            result["requirements"],
            result["products"]
        )

        print("\nPAYPILOT AI RESPONSE")
        print("--------------------")

        print(
            result["message"]
        )

    except json.JSONDecodeError:

        print("\nERROR")
        print("-----")

        print(
            "The AI returned an invalid JSON response."
        )

    except requests.exceptions.ConnectionError:

        print("\nERROR")
        print("-----")

        print(
            "Could not connect to Gemini API."
        )

        print(
            "Please check GEMINI_API_KEY."
        )

    except requests.exceptions.Timeout:

        print("\nERROR")
        print("-----")

        print(
            "Gemini API took too long to respond."
        )

    except requests.exceptions.HTTPError as error:

        print("\nGEMINI API ERROR")
        print("----------------")

        print(error)

        if error.response is not None:
            print(error.response.text)

    except Exception as error:

        print("\nERROR")
        print("-----")

        print(
            f"{type(error).__name__}: {error}"
        )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()