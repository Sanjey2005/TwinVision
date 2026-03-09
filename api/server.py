"""
TwinVision FastAPI application entry point.

Start with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints
─────────
  GET  /api/health                          — liveness + device info
  POST /api/generate                        — start a pipeline job
  GET  /api/status/{job_id}                 — poll job progress
  GET  /api/results/{job_id}                — retrieve full results
  GET  /api/images/{job_id}/{path}          — serve generated images / charts
  GET  /api/videos/{job_id}/{filename}      — serve generated videos
  GET  /output/{path}                       — raw static file access
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.middleware.cors import setup_cors
from api.models.request_models import HealthResponse
from api.routes import generate, results, status
from pipeline.config import API_HOST, API_PORT, OUTPUT_BASE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Log startup information once the server is ready."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "TwinVision API started — host=%s  port=%d  device=%s",
        API_HOST, API_PORT, device,
    )
    logger.info("Output directory: %s", OUTPUT_BASE.resolve())
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TwinVision API",
    description="Prompt-to-video AI model comparison: FLUX.1 vs SD3.5",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite frontend running on localhost:5173.
setup_cors(app)

# Static file mount — exposes output/ so charts and images can also be
# reached directly at /output/{job_id}/... (useful for debugging).
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
app.mount(
    "/output",
    StaticFiles(directory=str(OUTPUT_BASE), html=False),
    name="output",
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(generate.router, tags=["pipeline"])
app.include_router(status.router,   tags=["pipeline"])
app.include_router(results.router,  tags=["pipeline"])

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Liveness check with GPU device info",
)
def health() -> HealthResponse:
    """Return server liveness status and the active torch device.

    Returns:
        HealthResponse with status='ok' and device='cuda' or 'cpu'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug("Health check — device: %s", device)
    return HealthResponse(status="ok", device=device)


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------


@app.get(
    "/api/images/{job_id}/{path:path}",
    tags=["assets"],
    summary="Serve a generated image or chart PNG",
)
def serve_image(job_id: str, path: str) -> FileResponse:
    """Serve a PNG file from the job's output directory.

    URL pattern covers three sub-directories:
        /api/images/{job_id}/flux/{filename}        — FLUX.1 generated images
        /api/images/{job_id}/sd35/{filename}        — SD3.5 generated images
        /api/images/{job_id}/results/{filename}     — chart PNGs

    Args:
        job_id: The 8-character hex job identifier.
        path: Relative path within output/{job_id}/ (e.g. 'flux/prompt_0_img_0.png').

    Returns:
        FileResponse with the PNG file contents.

    Raises:
        HTTPException 404: If the file does not exist on disk.
    """
    file_path = OUTPUT_BASE / job_id / path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {path}",
        )
    return FileResponse(str(file_path), media_type="image/png")


# ---------------------------------------------------------------------------
# Video serving
# ---------------------------------------------------------------------------


@app.get(
    "/api/videos/{job_id}/{filename}",
    tags=["assets"],
    summary="Serve a generated MP4 video",
)
def serve_video(job_id: str, filename: str) -> FileResponse:
    """Serve an MP4 video file from the job's videos/ subdirectory.

    URL pattern:
        /api/videos/{job_id}/flux_prompt_0.mp4
        /api/videos/{job_id}/sd35_prompt_0.mp4
        /api/videos/{job_id}/comparison_prompt_0.mp4

    FileResponse supports HTTP range requests so the browser's <video>
    element can seek without downloading the entire file first.

    Args:
        job_id: The 8-character hex job identifier.
        filename: MP4 filename (e.g. 'flux_prompt_0.mp4').

    Returns:
        FileResponse with video/mp4 content-type and range-request support.

    Raises:
        HTTPException 404: If the file does not exist on disk.
    """
    file_path = OUTPUT_BASE / job_id / "videos" / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {filename}",
        )
    return FileResponse(str(file_path), media_type="video/mp4")

