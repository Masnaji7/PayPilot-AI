from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent import create_recommendation


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="PayPilot AI",
    description="AI-powered shopping recommendation API",
    version="1.0.0"
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,

    # Allow frontend running through Live Server
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ==========================================
# REQUEST MODEL
# ==========================================

class ShoppingRequest(BaseModel):

    message: str


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to PayPilot AI API",
        "status": "running"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==========================================
# RECOMMENDATION API
# ==========================================

@app.post("/recommend")
def recommend(request: ShoppingRequest):

    result = create_recommendation(
        request.message
    )

    return result