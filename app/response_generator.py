import json
import os
import requests

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b"
)


# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(prompt):
    """
    Generate text using local Ollama.
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        },
        timeout=180
    )

    if not response.ok:

        raise RuntimeError(
            f"Ollama API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    if "response" not in data:

        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    return str(
        data["response"]
    ).strip()


# ============================================================
# GENERATE CUSTOMER RESPONSE
# ============================================================

def generate_customer_response(
    user_message,
    requirements,
    products
):
    """
    Generate a human-friendly shopping recommendation
    using Ollama/Qwen.
    """

    # --------------------------------------------------------
    # No products
    # --------------------------------------------------------

    if not products:

        return (
            "I couldn't find a matching product in "
            "the current catalog. "
            "Try increasing your budget or changing "
            "your preferences."
        )

    # --------------------------------------------------------
    # Limit products sent to AI
    # --------------------------------------------------------

    product_data = []

    for product in products[:5]:

        product_data.append({
            "name": product.get(
                "name",
                "Unknown"
            ),
            "price": product.get(
                "price"
            ),
            "rating": product.get(
                "rating"
            ),
            "description": product.get(
                "description",
                ""
            ),
            "features": product.get(
                "features",
                []
            ),
            "recommendation_score": product.get(
                "recommendation_score"
            ),
            "matched_preferences": product.get(
                "matched_preferences",
                []
            )
        })

    products_json = json.dumps(
        product_data,
        ensure_ascii=False,
        indent=2
    )

    requirements_json = json.dumps(
        requirements,
        ensure_ascii=False,
        indent=2
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are PayPilot AI, a helpful shopping assistant.

The customer said:

"{user_message}"

PayPilot understood these requirements:

{requirements_json}

Available matching products:

{products_json}

Create a short, natural shopping recommendation.

Rules:

1. Recommend the best product first.
2. Explain why it matches the customer's request.
3. Mention price and rating when available.
4. Mention relevant preferences.
5. Do not invent specifications.
6. Do not mention internal AI processing.
7. Do not mention Ollama.
8. Do not mention Gemini.
9. Do not use markdown tables.
10. Keep the answer friendly and easy to understand.
11. Keep the answer under 150 words.

Return only the customer-facing response.
"""

    return call_ollama(prompt)