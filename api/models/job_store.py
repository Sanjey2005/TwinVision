"""
Thread-safe in-memory job store for TwinVision.

All pipeline jobs are tracked here as plain dicts. Every public function
acquires _lock before reading or writing so it is safe to call from both
FastAPI route handlers and pipeline threads concurrently.

Public API
──────────
  create_job(job_id, prompt)       → dict        create and return new job entry
  update_job(job_id, **fields)     → None        update any job fields in-place
  set_error(job_id, message)       → None        mark job as 'failed'
  get_job(job_id)                  → dict | None  return a snapshot copy
  store_result(job_id, payload)    → None        store result and mark 'done'
  get_result(job_id)               → dict | None  return stored result payload

Aliases provided for orchestrator compatibility:
  mark_job_failed  = set_error
  set_job_result   = store_result
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Store state
# ---------------------------------------------------------------------------

# The single source of truth for all in-flight and completed jobs.
_jobs: dict[str, dict[str, Any]] = {}

# All reads and writes must hold this lock to prevent race conditions
# between FastAPI route handlers and background pipeline threads.
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def create_job(job_id: str, prompt: str) -> dict:
    """Create a new job entry in the store and return a snapshot.

    The job is initialised with status 'queued' and zero progress. Any
    pre-existing entry with the same job_id is overwritten.

    Args:
        job_id: Unique 8-character hex identifier (uuid4().hex[:8]).
        prompt: Original user-supplied text prompt for this job.

    Returns:
        A copy of the newly created job dict.
    """
    entry: dict[str, Any] = {
        "job_id":   job_id,
        "status":   "queued",
        "stage":    "",
        "progress": 0,
        "message":  "Job queued, waiting to start.",
        "prompt":   prompt,
        "result":   None,
        "error":    None,
    }
    with _lock:
        _jobs[job_id] = entry
    logger.debug("Job created: %s", job_id)
    return dict(entry)


def update_job(job_id: str, **fields: Any) -> None:
    """Thread-safely update one or more fields on an existing job.

    Unknown field names are accepted and stored as-is, which allows the
    orchestrator to pass status, stage, progress, and message in a single
    call without needing a separate signature per combination.

    Args:
        job_id: Job identifier to update.
        **fields: Arbitrary keyword arguments to merge into the job dict.
            Common keys: status, stage, progress, message.

    Raises:
        KeyError: If job_id does not exist in the store.
    """
    with _lock:
        if job_id not in _jobs:
            raise KeyError(f"Job '{job_id}' not found in store.")
        _jobs[job_id].update(fields)
    logger.debug("Job updated: %s  fields=%s", job_id, list(fields.keys()))


def set_error(job_id: str, message: str) -> None:
    """Mark a job as failed and record the error message.

    Progress is not reset so the UI can show where the pipeline stopped.

    Args:
        job_id: Job identifier.
        message: Human-readable description of the failure.
    """
    with _lock:
        if job_id not in _jobs:
            logger.warning("set_error called for unknown job '%s'.", job_id)
            return
        _jobs[job_id].update({
            "status":  "failed",
            "stage":   "failed",
            "message": message,
            "error":   message,
        })
    logger.error("Job failed: %s — %s", job_id, message)


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return a snapshot copy of the job dict, or None if not found.

    Returns a copy (not a reference) so callers cannot accidentally mutate
    store state without going through the lock.

    Args:
        job_id: Job identifier.

    Returns:
        Shallow copy of the job dict, or None if the job does not exist.
    """
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def store_result(job_id: str, payload: dict[str, Any]) -> None:
    """Store the completed ComparisonPayload and mark the job as done.

    Args:
        job_id: Job identifier.
        payload: Full ComparisonPayload dict produced by the orchestrator.
    """
    with _lock:
        if job_id not in _jobs:
            logger.warning("store_result called for unknown job '%s'.", job_id)
            return
        _jobs[job_id].update({
            "status":   "done",
            "stage":    "done",
            "progress": 100,
            "message":  "Pipeline complete.",
            "result":   payload,
        })
    logger.info("Job complete: %s", job_id)


def get_result(job_id: str) -> dict[str, Any] | None:
    """Return the stored ComparisonPayload for a completed job.

    Args:
        job_id: Job identifier.

    Returns:
        The result dict stored by store_result(), or None if the job does
        not exist or has not yet completed.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return job.get("result")


# ---------------------------------------------------------------------------
# Aliases for orchestrator compatibility
# (orchestrator.py imports these names; keep in sync if renamed above)
# ---------------------------------------------------------------------------

mark_job_failed = set_error
set_job_result  = store_result
