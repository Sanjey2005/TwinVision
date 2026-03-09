"""
POST /api/generate — start a new pipeline job.

Creates a job entry, spawns a daemon thread running run_full_pipeline(),
and immediately returns HTTP 202 with the job_id. The caller then polls
GET /api/status/{job_id} for progress.
"""

from __future__ import annotations

import logging
import threading
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.models.job_store import create_job
from api.models.request_models import GenerateRequest, GenerateResponse
from pipeline.orchestrator import run_full_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/api/generate",
    response_model=GenerateResponse,
    status_code=202,
    summary="Start a new prompt-to-video pipeline job",
)
def start_generation(request: GenerateRequest) -> JSONResponse:
    """Accept a text prompt and kick off the full TwinVision pipeline.

    Generates a short unique job_id, registers the job in the store with
    status 'queued', then starts the pipeline in a background daemon thread.
    Returns immediately with HTTP 202 so the client can begin polling.

    Args:
        request: Validated POST body containing the user prompt.

    Returns:
        JSONResponse (HTTP 202) with job_id and initial status 'queued'.
    """
    job_id: str = uuid.uuid4().hex[:8]
    create_job(job_id, request.prompt)

    thread = threading.Thread(
        target=run_full_pipeline,
        kwargs={"job_id": job_id, "prompt": request.prompt},
        daemon=True,
        name=f"pipeline-{job_id}",
    )
    thread.start()

    logger.info("Job %s queued for prompt: %.60s...", job_id, request.prompt)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued"},
    )
