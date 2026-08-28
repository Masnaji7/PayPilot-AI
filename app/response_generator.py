import os
import requests
from dotenv import load_dotenv


load_dotenv()


# ============================================================
# GEMINI SETTINGS
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)


# ============================================================
# CUSTOMER RESPONSE
# ============================================================

def generate_customer_response(
    user_message,
    requirements,
    products
):

    # --------------------------------------------------------
    # Check Gemini API key
    # --------------------------------------------------------

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing."
        )


    # --------------------------------------------------------
    # No products
    # --------------------------------------------------------

    if not products:

        return (
            "I couldn't find a suitable product in our "
            "current catalog. Try increasing your budget "
            "or changing your preferences."
        )


    # --------------------------------------------------------
    # Best product
    # --------------------------------------------------------

    best_product = products[0]


    # --------------------------------------------------------
    # Product information
    # --------------------------------------------------------

    product_information = []

    for product in products:

        product_information.append(
            {
                "name": product.get("name"),
                "price": product.get("price"),
                "rating": product.get("rating"),
                "features": product.get(
                    "features",
                    []
                ),
                "description": product.get(
                    "description",
                    ""
                )
            }
        )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are PayPilot AI, a friendly shopping assistant.

Customer request:
{user_message}

Customer requirements:
{requirements}

Available products:
{product_information}

Best recommended product:
{best_product.get("name")}

Rules:

- Recommend only products from the available products.
- Do not invent products.
- Do not invent prices.
- Do not invent features.
- Mention the recommended product.
- Mention its price.
- Explain briefly why it is suitable.
- Keep the response friendly.
- Keep the response simple.
- Do not mention recommendation scores.
- Do not mention APIs.
- Do not mention technical details.

Return only the customer-facing recommendation.
"""


    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

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
                "temperature": 0.3,
                "maxOutputTokens": 300
            }
        },

        timeout=60
    )


    # --------------------------------------------------------
    # Check response
    # --------------------------------------------------------

    response.raise_for_status()


    data = response.json()


    # --------------------------------------------------------
    # Get Gemini text
    # --------------------------------------------------------

    try:

        answer = (
            data["candidates"][0]
            ["content"]["parts"][0]
            ["text"]
        )

    except (KeyError, IndexError, TypeError):

        raise RuntimeError(
            f"Unexpected Gemini response: {data}"
        )


    return answer.strip()