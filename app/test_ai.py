import os
import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)


def ask_ai(question):
    """
    Send a question to Google Gemini
    and return the generated response.
    """

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
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
                            "text": question
                        }
                    ]
                }
            ]
        },

        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return (
        data["candidates"][0]
        ["content"]
        ["parts"][0]
        ["text"]
    ).strip()


if __name__ == "__main__":

    question = (
        "I need wireless headphones under ₹5000 "
        "for studying."
    )

    answer = ask_ai(question)

    print("\nPAYPILOT AI RESPONSE")
    print("--------------------")
    print(answer)