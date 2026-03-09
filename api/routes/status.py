"""
GET /api/status/{job_id} — poll pipeline progress.

Returns the current status, stage label, progress percentage (0–100),
and a human-readable message. The frontend polls this endpoint every
two seconds while status is 'queued' or 'running'.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.models.job_store import get_job
from api.models.request_models import JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/api/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll the progress of a pipeline job",
)
def get_status(job_id: str) -> JobStatusResponse:
    """Return the current status of a pipeline job.

    Args:
        job_id: The 8-character hex job identifier returned by POST /api/generate.

    Returns:
        JobStatusResponse with status, stage, progress, and message fields.

    Raises:
        HTTPException 404: If no job with the given job_id exists.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        stage=job.get("stage", ""),
        progress=job.get("progress", 0),
        message=job.get("message", ""),
    )
