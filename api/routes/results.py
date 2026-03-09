"""
GET /api/results/{job_id} — retrieve the full ComparisonPayload.

While the job is still running this endpoint returns the job_id, current
status, and prompt with results: null.  Once status is 'done' the full
nested payload (images, videos, metrics, charts) is returned.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.models.job_store import get_job, get_result

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/api/results/{job_id}",
    summary="Retrieve the full comparison results for a completed job",
)
def get_results(job_id: str) -> JSONResponse:
    """Return the ComparisonPayload for a pipeline job.

    If the job has not yet completed, results is null and status reflects
    the current pipeline state.  If the job failed, status is 'failed' and
    results is null.

    Args:
        job_id: The 8-character hex job identifier.

    Returns:
        JSON object matching the ComparisonPayload schema from the PRD.
        Shape is always:
            { job_id, status, prompt, results: <payload | null> }

    Raises:
        HTTPException 404: If no job with the given job_id exists.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )

    if job["status"] != "done":
        # Return partial payload — results not yet available.
        return JSONResponse(content={
            "job_id":  job["job_id"],
            "status":  job["status"],
            "prompt":  job.get("prompt", ""),
            "results": None,
        })

    # Job is done — return the full stored payload.
    payload = get_result(job_id)
    if payload is None:
        # Shouldn't happen (store_result sets status=done), but guard anyway.
        logger.warning("Job %s status is 'done' but result payload is missing.", job_id)
        raise HTTPException(
            status_code=500,
            detail="Result payload is missing for a completed job.",
        )

    return JSONResponse(content=payload)
