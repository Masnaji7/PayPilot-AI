import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"


def ask_ai(question):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": question,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


if __name__ == "__main__":
    question = "I need headphones under ₹5000 for studying."

    answer = ask_ai(question)

    print("\nPAYPILOT AI RESPONSE")
    print("--------------------")
    print(answer)