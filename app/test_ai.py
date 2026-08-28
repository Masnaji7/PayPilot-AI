import os
import requests
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b"
)


# ============================================================
# ASK OLLAMA
# ============================================================

def ask_ai(question):
    """
    Send a question to Ollama and return
    the generated response.
    """

    try:
        response = requests.post(
            OLLAMA_URL,

            json={
                "model": OLLAMA_MODEL,
                "prompt": question,
                "stream": False
            },

            timeout=120
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not connect to Ollama: {e}"
        )

    data = response.json()

    if "response" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    return data["response"].strip()


# ============================================================
# TEST OLLAMA
# ============================================================

if __name__ == "__main__":

    question = (
        "I need wireless headphones under ₹5000 "
        "for studying."
    )

    answer = ask_ai(question)

    print("\nPAYPILOT AI RESPONSE")
    print("--------------------")
    print(answer)