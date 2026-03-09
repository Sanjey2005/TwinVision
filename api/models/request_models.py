"""
Pydantic v2 request and response models for the TwinVision API.

All field names and JSON shapes match the API examples in twinvision_prd.md
Section 5.6 exactly so that FastAPI's automatic serialisation produces the
correct wire format without any extra aliases or transformers.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Body for POST /api/generate.

    Attributes:
        prompt: Text prompt used to drive both image-generation models.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text prompt for image generation.",
    )


# ---------------------------------------------------------------------------
# Lightweight response for POST /api/generate
# ---------------------------------------------------------------------------


class GenerateResponse(BaseModel):
    """Immediate response body for POST /api/generate (HTTP 202).

    Attributes:
        job_id: Short hex identifier for the queued pipeline job.
        status: Always 'queued' at creation time.
    """

    job_id: str
    status: str = "queued"


# ---------------------------------------------------------------------------
# Status polling  (GET /api/status/{job_id})
# ---------------------------------------------------------------------------


class JobStatusResponse(BaseModel):
    """Response body for GET /api/status/{job_id}.

    Attributes:
        job_id: Unique job identifier.
        status: One of 'queued', 'running', 'done', or 'failed'.
        stage: Current pipeline stage label (e.g. 'image_generation').
        progress: Completion percentage in [0, 100].
        message: Human-readable status message for the UI.
    """

    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    stage: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    message: str = ""


# ---------------------------------------------------------------------------
# Results payload  (GET /api/results/{job_id})
# ---------------------------------------------------------------------------


class MetricWinner(BaseModel):
    """Per-metric comparison result.

    Attributes:
        winner: Display name of the winning model ('FLUX', 'SD3.5', or 'Tie').
        flux: Mean average score for FLUX.1 on this metric.
        sd35: Mean average score for SD3.5 on this metric.
        diff_pct: Relative percentage difference between the two scores.
    """

    winner: str
    flux: float
    sd35: float
    diff_pct: float


class ImagesPayload(BaseModel):
    """Relative URLs to all generated images, split by model.

    Attributes:
        flux: Ordered list of /api/images/... URLs for FLUX.1 images.
        sd35: Ordered list of /api/images/... URLs for SD3.5 images.
    """

    flux: list[str] = Field(default_factory=list)
    sd35: list[str] = Field(default_factory=list)


class VideosPayload(BaseModel):
    """Relative URLs to the generated MP4 videos.

    All fields are optional because video creation may partially fail
    for individual prompts while other stages succeed.

    Attributes:
        flux: URL to the FLUX.1 model video.
        sd35: URL to the SD3.5 model video.
        comparison: URL to the side-by-side comparison video.
    """

    flux: Optional[str] = None
    sd35: Optional[str] = None
    comparison: Optional[str] = None


class ResultsPayload(BaseModel):
    """Inner results object returned by GET /api/results/{job_id}.

    Attributes:
        overall_winner: Display name of the overall winner.
        overall_score: Weighted normalised scores per model.
        metric_winners: Per-metric comparison details.
        images: URLs to all generated images.
        videos: URLs to all generated videos.
        generation_time: Total wall-clock generation seconds per model.
        charts: URLs to bar_chart.png, radar_chart.png, and image_grid.png.
    """

    overall_winner: str
    overall_score: dict[str, float]
    metric_winners: dict[str, MetricWinner]
    images: ImagesPayload
    videos: VideosPayload
    generation_time: dict[str, float]
    charts: dict[str, str]


class ComparisonPayload(BaseModel):
    """Full response body for GET /api/results/{job_id}.

    Attributes:
        job_id: Unique job identifier.
        status: Job status at time of retrieval.
        prompt: Original user-supplied text prompt.
        results: Complete comparison data; None while the job is in progress.
    """

    job_id: str
    status: str
    prompt: str
    results: Optional[ResultsPayload] = None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for GET /api/health.

    Attributes:
        status: Always 'ok' when the server is reachable.
        device: Torch device in use ('cuda' or 'cpu').
    """

    status: str = "ok"
    device: str = "unknown"
