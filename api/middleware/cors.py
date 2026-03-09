"""
CORS middleware configuration for TwinVision API.

The React frontend runs on http://localhost:5173 (Vite default).
All origins, methods, and headers are permitted so that the browser
can POST to /api/generate and stream back video/image URLs without
pre-flight failures.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5173",   # Vite dev server (React frontend)
    "http://127.0.0.1:5173",   # Same host, numeric form
]


def setup_cors(app: FastAPI) -> None:
    """Attach CORSMiddleware to the FastAPI application.

    Args:
        app: The FastAPI application instance to configure.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
