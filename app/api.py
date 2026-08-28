from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent import create_recommendation


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PayPilot AI",
    description="AI-powered shopping recommendation API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ShoppingRequest(BaseModel):

    message: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to PayPilot AI API",
        "status": "running",
        "ai_engine": "Ollama",
        "model": "qwen2.5:7b"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "ai_engine": "Ollama",
        "model": "qwen2.5:7b"
    }


# ============================================================
# RECOMMEND
# ============================================================

@app.post("/recommend")
def recommend(request: ShoppingRequest):

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Shopping message cannot be empty."
        )

    try:

        result = create_recommendation(
            message
        )

        return result

    except Exception as error:

        print(
            f"[ERROR] /recommend failed: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )