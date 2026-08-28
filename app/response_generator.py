import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"


def generate_customer_response(
    user_message,
    requirements,
    products
):
    """
    Generate a natural-language shopping recommendation.
    """

    if not products:
        return (
            "I couldn't find a suitable product in our "
            "current catalog. Try increasing your budget "
            "or changing your product preference."
        )

    best_product = products[0]

    product_information = []

    for product in products:
        product_information.append(
            {
                "name": product["name"],
                "price": product["price"],
                "rating": product["rating"],
                "features": product["features"],
                "score": product["recommendation_score"]
            }
        )

    prompt = f"""
You are PayPilot AI, a friendly shopping assistant.

Write a short, natural recommendation for the customer.

Customer request:
{user_message}

Customer requirements:
{requirements}

Available products:
{product_information}

The highest-ranked product is:
{best_product["name"]}

Rules:
- Recommend only products from the provided list.
- Do not invent products.
- Do not invent prices or features.
- Clearly mention the recommended product and price.
- Explain briefly why it is suitable.
- Keep the answer friendly and easy to understand.
- Do not mention recommendation scores.
- Do not mention that you are an AI model.
- Do not use complicated technical language.

Return only the customer-facing response.
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

    return response.json()["response"].strip()