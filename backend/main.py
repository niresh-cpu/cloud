"""
FastAPI Application Entry Point for Real-Time Multi-Crop Plant Leaf Disease Detection
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.routes.api import router as api_router
from backend.services.inference_service import InferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pre-warm model and load database
    print("Initializing Agricultural AI Backend Service...")
    InferenceService.initialize()
    yield
    print("Shutting down Agricultural AI Backend Service...")


app = FastAPI(
    title="Real-Time Multi-Crop Plant Leaf Disease Detection API",
    description="ResNet-50 Transfer Learning System with Agricultural Advisory & Leaf Validation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)


@app.get("/")
def root():
    return {
        "message": "Real-Time Multi-Crop Plant Leaf Disease Detection API is running.",
        "documentation": "/docs",
        "health_check": "/api/health"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
