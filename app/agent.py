import json
import os
import re
import requests

from dotenv import load_dotenv

from app.product_search import search_products
from app.recommendation import rank_products
from app.response_generator import generate_customer_response


# ============================================================
# LOAD ENVIRONMENT VARIABLES
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
# CALL OLLAMA
# ============================================================

def call_ollama(prompt, temperature=0):
    """
    Send a prompt to local Ollama and return generated text.
    """

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            },
            timeout=180
        )

    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running and "
            f"{OLLAMA_MODEL} is installed."
        ) from error

    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "Ollama took too long to respond."
        ) from error

    if not response.ok:
        raise RuntimeError(
            f"Ollama API error {response.status_code}: "
            f"{response.text}"
        )

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Ollama returned invalid JSON: {response.text}"
        ) from error

    if "response" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    return str(data["response"]).strip()


# ============================================================
# CLEAN JSON FROM OLLAMA
# ============================================================

def clean_json_response(text):
    """
    Remove markdown/code fences and extract JSON.
    """

    text = text.strip()

    # Remove markdown fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # Find JSON object if Ollama added extra text
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


# ============================================================
# FALLBACK REQUIREMENT EXTRACTION
# ============================================================

def fallback_requirements(user_message):
    """
    Extract basic requirements without AI.
    This prevents the entire API from failing
    if Ollama returns invalid JSON.
    """

    text = user_message.lower()

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    category = None

    if any(word in text for word in [
        "headphone",
        "headphones",
        "earphone",
        "earphones",
        "headset"
    ]):
        category = "headphones"

    elif any(word in text for word in [
        "running shoe",
        "running shoes",
        "sports shoe",
        "sports shoes"
    ]):
        category = "running shoes"

    elif any(word in text for word in [
        "charger",
        "charging"
    ]):
        category = "chargers"

    elif any(word in text for word in [
        "laptop accessory",
        "laptop accessories"
    ]):
        category = "laptop accessories"

    elif any(word in text for word in [
        "computer accessory",
        "computer accessories",
        "mouse",
        "keyboard"
    ]):
        category = "computer accessories"

    elif any(word in text for word in [
        "sports accessory",
        "sports accessories"
    ]):
        category = "sports accessories"

    # --------------------------------------------------------
    # Budget
    # --------------------------------------------------------

    max_price = None

    budget_patterns = [
        r"under\s*[₹rs.]?\s*([\d,]+)",
        r"below\s*[₹rs.]?\s*([\d,]+)",
        r"less than\s*[₹rs.]?\s*([\d,]+)",
        r"within\s*[₹rs.]?\s*([\d,]+)",
        r"budget\s*(?:of|is)?\s*[₹rs.]?\s*([\d,]+)"
    ]

    for pattern in budget_patterns:
        match = re.search(pattern, text)

        if match:
            try:
                max_price = int(
                    match.group(1).replace(",", "")
                )
            except ValueError:
                max_price = None

            break

    # --------------------------------------------------------
    # Preferences
    # --------------------------------------------------------

    preferences = []

    preference_words = [
        "wireless",
        "comfortable",
        "noise cancellation",
        "long battery",
        "lightweight",
        "portable",
        "fast charging",
        "ergonomic",
        "breathable",
        "studying",
        "gaming",
        "running",
        "travel"
    ]

    for preference in preference_words:
        if preference in text:
            preferences.append(preference)

    return {
        "category": category,
        "max_price": max_price,
        "preferences": preferences
    }


# ============================================================
# EXTRACT CUSTOMER REQUIREMENTS
# ============================================================

def extract_customer_requirements(user_message):
    """
    Use Ollama/Qwen to understand the customer's
    shopping requirements.

    If Ollama returns invalid JSON, use a local
    fallback parser instead of crashing.
    """

    prompt = f"""
You are the requirement extraction component of PayPilot AI.

Read the customer's shopping request.

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{{
    "category": "product category or null",
    "max_price": number or null,
    "preferences": ["preference1", "preference2"]
}}

Available product categories:

- headphones
- laptop accessories
- computer accessories
- chargers
- running shoes
- sports accessories

Rules:

1. Use "headphones" for headphones, earphones, or headsets.

2. Never use:
   - electronics
   - technology
   - gadgets

3. Choose only ONE category.

4. If no category matches, use null.

5. Extract only preferences actually mentioned.

6. Examples of preferences:
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

7. If the customer says wireless headphones,
   include "wireless".

8. If the customer says for studying,
   include "studying".

9. If there is no budget, use null.

10. Never invent information.

11. Return JSON only.

Customer request:

{user_message}
"""

    try:
        result = call_ollama(
            prompt,
            temperature=0
        )

        result = clean_json_response(result)

        requirements = json.loads(result)

        if not isinstance(requirements, dict):
            raise ValueError("Ollama response is not an object")

    except Exception as error:

        print(
            f"[WARNING] Ollama requirement extraction failed: "
            f"{error}"
        )

        print(
            "[INFO] Using fallback requirement extraction."
        )

        requirements = fallback_requirements(
            user_message
        )

    # --------------------------------------------------------
    # Normalize fields
    # --------------------------------------------------------

    category = requirements.get(
        "category"
    )

    max_price = requirements.get(
        "max_price"
    )

    preferences = requirements.get(
        "preferences",
        []
    )

    # --------------------------------------------------------
    # Validate category
    # --------------------------------------------------------

    valid_categories = {
        "headphones",
        "laptop accessories",
        "computer accessories",
        "chargers",
        "running shoes",
        "sports accessories"
    }

    if category not in valid_categories:
        category = None

    # --------------------------------------------------------
    # Validate price
    # --------------------------------------------------------

    if max_price is not None:

        try:
            max_price = float(max_price)

        except (TypeError, ValueError):
            max_price = None

    # --------------------------------------------------------
    # Validate preferences
    # --------------------------------------------------------

    if not isinstance(preferences, list):
        preferences = []

    preferences = [
        str(item).strip()
        for item in preferences
        if str(item).strip()
    ]

    return {
        "category": category,
        "max_price": max_price,
        "preferences": preferences
    }


# ============================================================
# CREATE RECOMMENDATION
# ============================================================

def create_recommendation(user_message):
    """
    Complete PayPilot AI workflow:

    1. Understand customer request using Ollama
    2. Search product catalog
    3. Rank products
    4. Generate final response using Ollama
    """

    if not user_message or not user_message.strip():
        raise ValueError(
            "Shopping request cannot be empty."
        )

    user_message = user_message.strip()

    # --------------------------------------------------------
    # STEP 1 - Extract requirements
    # --------------------------------------------------------

    requirements = extract_customer_requirements(
        user_message
    )

    print("\nCUSTOMER REQUIREMENTS")
    print(
        json.dumps(
            requirements,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # STEP 2 - Search products
    # --------------------------------------------------------

    products = search_products(
        category=requirements["category"],
        max_price=requirements["max_price"],
        preferences=requirements["preferences"]
    )

    if products is None:
        products = []

    # --------------------------------------------------------
    # STEP 3 - Rank products
    # --------------------------------------------------------

    products = rank_products(
        products,
        preferences=requirements["preferences"],
        max_price=requirements["max_price"]
    )

    if products is None:
        products = []

    # --------------------------------------------------------
    # STEP 4 - Generate customer response
    # --------------------------------------------------------

    try:

        customer_response = generate_customer_response(
            user_message,
            requirements,
            products
        )

    except Exception as error:

        print(
            f"[WARNING] Ollama response generation failed: "
            f"{error}"
        )

        # Safe fallback response
        if products:

            best = products[0]

            customer_response = (
                f"Based on your requirements, "
                f"I recommend {best.get('name', 'this product')} "
                f"at ₹{best.get('price', 'N/A')}. "
                f"It is one of the best matches in our catalog."
            )

        else:

            customer_response = (
                "I could not find a matching product "
                "in the current catalog. "
                "Try changing your budget or preferences."
            )

    # --------------------------------------------------------
    # STEP 5 - Return API response
    # --------------------------------------------------------

    return {
        "requirements": requirements,
        "products": products,
        "message": customer_response
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

def main():

    print("\n================================")
    print("         PAYPILOT AI")
    print("================================")

    print(
        f"\nAI Engine: Ollama"
    )

    print(
        f"Model: {OLLAMA_MODEL}"
    )

    print(
        f"URL: {OLLAMA_URL}"
    )

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

        print("\n")
        print("==============================")
        print("PAYPILOT AI RESULT")
        print("==============================")

        print("\nRequirements:")

        print(
            json.dumps(
                result["requirements"],
                indent=2,
                ensure_ascii=False
            )
        )

        print("\nProducts:")

        if not result["products"]:

            print(
                "No matching products found."
            )

        else:

            for index, product in enumerate(
                result["products"],
                start=1
            ):

                print(
                    f"{index}. "
                    f"{product.get('name', 'Unknown')} "
                    f"- ₹{product.get('price', 'N/A')} "
                    f"- ⭐ {product.get('rating', 'N/A')}"
                )

        print("\nAI Response:")
        print("------------------------------")
        print(result["message"])

    except Exception as error:

        print("\nERROR")
        print("------------------------------")
        print(
            f"{type(error).__name__}: {error}"
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()