"""
Full pipeline orchestrator for TwinVision.

run_full_pipeline() coordinates every stage in sequence, updating the job
store at defined progress milestones. It is designed to run inside a
threading.Thread spawned by the FastAPI server.

Stage map and progress milestones
──────────────────────────────────
  0 %  initialising
  5 %  FLUX.1 image generation begins
 30 %  FLUX.1 done  →  SD3.5 image generation begins
 55 %  SD3.5 done   →  FFmpeg video creation begins
 65 %  Videos done  →  Metric evaluation begins
 85 %  Evaluation done  →  Comparison report begins
 95 %  Comparison done  →  Building payload
100 %  Done
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

# Pipeline modules are imported lazily inside _run_stages() so that the
# FastAPI server can start without the heavy ML libraries (pyiqa, torchmetrics,
# diffusers, etc.) installed. They are only needed when a job actually runs.
from pipeline.config import OUTPUT_BASE, PIPELINE_TIMEOUT_SECONDS
from api.models.job_store import mark_job_failed, set_job_result, update_job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update(
    job_id: str,
    progress: int,
    message: str,
    stage: str,
) -> None:
    """Update the job store and emit a log line for the transition.

    Args:
        job_id: Unique job identifier.
        progress: Progress percentage in [0, 100].
        message: Human-readable status message shown in the UI.
        stage: Pipeline stage label for the status endpoint.
    """
    logger.info("[%s] %3d%%  (%s)  %s", job_id, progress, stage, message)
    update_job(
        job_id=job_id,
        status="running",
        stage=stage,
        progress=progress,
        message=message,
    )


def _group_images_by_prompt(images: list[Path]) -> dict[int, list[Path]]:
    """Parse prompt_{i}_img_{j}.png filenames and group by prompt index.

    Images that cannot be parsed are skipped with a warning.

    Args:
        images: Flat list of image Paths produced by generate_images functions.

    Returns:
        Dict mapping prompt_index (int) to an ordered list of image Paths
        sorted ascending by image index j.
    """
    grouped: dict[int, list[Path]] = {}
    for path in images:
        parts = path.stem.split("_")   # ['prompt', '{i}', 'img', '{j}']
        try:
            prompt_idx = int(parts[1])
        except (IndexError, ValueError):
            logger.warning(
                "Cannot parse prompt index from '%s' — skipping.", path.name
            )
            continue
        grouped.setdefault(prompt_idx, []).append(path)

    for idx in grouped:
        grouped[idx].sort(key=lambda p: int(p.stem.split("_")[3]))

    return grouped


def _build_payload(
    job_id: str,
    prompt: str,
    flux_paths: list[Path],
    sd35_paths: list[Path],
    video_paths: dict[str, Path],
    report: dict,
    flux_gen_time: float,
    sd35_gen_time: float,
) -> dict:
    """Assemble the final ComparisonPayload dict for storage and the API.

    Image, video, and chart paths are converted to relative /api/... URLs
    as specified in the PRD results schema.

    Args:
        job_id: Unique job identifier.
        prompt: User-supplied text prompt.
        flux_paths: Ordered list of FLUX.1 image Paths.
        sd35_paths: Ordered list of SD3.5 image Paths.
        video_paths: Dict from create_videos.run_all() mapping key → Path.
        report: Dict from compare.build_comparison_report().
        flux_gen_time: Total FLUX.1 generation time in seconds.
        sd35_gen_time: Total SD3.5 generation time in seconds.

    Returns:
        ComparisonPayload dict matching the GET /api/results/{job_id} schema.
    """
    # Image URLs: /api/images/{job_id}/{model}/{filename}
    flux_img_urls = [
        f"/api/images/{job_id}/flux/{p.name}" for p in flux_paths
    ]
    sd35_img_urls = [
        f"/api/images/{job_id}/sd35/{p.name}" for p in sd35_paths
    ]

    # Video URLs: /api/videos/{job_id}/{filename}
    videos: dict[str, str] = {}
    for key, path in video_paths.items():
        if key.startswith("flux_"):
            videos["flux"] = f"/api/videos/{job_id}/{path.name}"
        elif key.startswith("sd35_"):
            videos["sd35"] = f"/api/videos/{job_id}/{path.name}"
        elif key.startswith("comparison_"):
            videos["comparison"] = f"/api/videos/{job_id}/{path.name}"

    # Chart URLs: /api/images/{job_id}/results/{filename}
    charts: dict[str, str] = {
        chart_key: f"/api/images/{job_id}/results/{Path(chart_path).name}"
        for chart_key, chart_path in report["charts"].items()
    }

    return {
        "job_id": job_id,
        "status": "done",
        "prompt": prompt,
        "results": {
            "overall_winner":  report["overall_winner"],
            "overall_score":   report["overall_score"],
            "metric_winners":  report["metric_winners"],
            "images": {
                "flux": flux_img_urls,
                "sd35": sd35_img_urls,
            },
            "videos": videos,
            "generation_time": {
                "flux": round(flux_gen_time, 2),
                "sd35": round(sd35_gen_time, 2),
            },
            "charts": charts,
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_full_pipeline(
    job_id: str,
    prompt: str,
    n_images: int = 7,
    device: str | None = None,
) -> None:
    """Run the full TwinVision pipeline for a single user prompt.

    Intended to run inside a threading.Thread. Catches all exceptions and
    marks the job as failed rather than propagating to the thread boundary.

    Args:
        job_id: Unique identifier that scopes output paths and job state.
        prompt: Text prompt supplied by the user via the API.
        n_images: Number of images to generate per model (default 7,
            pass 2 for a fast test run).
        device: Torch device string ('cuda' / 'cpu'). Auto-detected if None.
    """
    wall_start = time.monotonic()

    def _check_timeout() -> None:
        """Raise TimeoutError if elapsed time exceeds PIPELINE_TIMEOUT_SECONDS."""
        elapsed = time.monotonic() - wall_start
        if elapsed > PIPELINE_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Pipeline exceeded the {PIPELINE_TIMEOUT_SECONDS}s limit "
                f"(elapsed {elapsed:.0f}s)."
            )

    try:
        _run_stages(job_id, prompt, n_images, device, _check_timeout, wall_start)
    except TimeoutError as exc:
        logger.error("[%s] Timeout: %s", job_id, exc)
        mark_job_failed(job_id, str(exc))
    except Exception as exc:
        logger.exception("[%s] Pipeline failed unexpectedly.", job_id)
        mark_job_failed(job_id, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Stage runner (separated to keep the top-level try/except clean)
# ---------------------------------------------------------------------------

def _run_stages(
    job_id: str,
    prompt: str,
    n_images: int,
    device: str | None,
    check_timeout,
    wall_start: float,
) -> None:
    """Execute all five pipeline stages in sequence.

    Args:
        job_id: Unique job identifier.
        prompt: Text prompt for generation.
        n_images: Images per model.
        device: Torch device string or None.
        check_timeout: Callable that raises TimeoutError when limit is hit.
        wall_start: monotonic clock value at pipeline start, for timing.
    """
    # Lazy imports — ML libraries are only loaded when a job actually runs.
    from pipeline import compare, create_videos, evaluate, generate_images  # noqa: PLC0415

    base_dir: Path = OUTPUT_BASE / job_id
    flux_dir: Path = base_dir / "flux"
    sd35_dir: Path = base_dir / "sd35"
    results_dir: Path = base_dir / "results"

    for d in (flux_dir, sd35_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Stage 1 — FLUX.1 image generation  (5 → 30 %)
    # -----------------------------------------------------------------------
    _update(
        job_id, progress=5, stage="image_generation",
        message=f"Generating {n_images} images with FLUX.1 [schnell]...",
    )
    check_timeout()

    flux_paths: list[Path] = generate_images.generate_images(
        prompt=prompt,
        model_name="flux",
        num_images=n_images,
        prompt_index=0,
        output_dir=flux_dir,
        device=device,
    )
    flux_gen_time = generate_images.get_generation_times().get("flux", 0.0)

    logger.info(
        "[%s] FLUX.1 complete: %d images in %.1fs (%.2fs/img avg).",
        job_id, len(flux_paths), flux_gen_time,
        flux_gen_time / max(len(flux_paths), 1),
    )

    # -----------------------------------------------------------------------
    # Stage 2 — SD3.5 Large image generation  (30 → 55 %)
    # -----------------------------------------------------------------------
    _update(
        job_id, progress=30, stage="image_generation",
        message=f"Generating {n_images} images with SD3.5 Large...",
    )
    check_timeout()

    sd35_paths: list[Path] = generate_images.generate_images(
        prompt=prompt,
        model_name="sd35",
        num_images=n_images,
        prompt_index=0,
        output_dir=sd35_dir,
        device=device,
    )
    sd35_gen_time = generate_images.get_generation_times().get("sd35", 0.0)

    logger.info(
        "[%s] SD3.5 complete: %d images in %.1fs (%.2fs/img avg).",
        job_id, len(sd35_paths), sd35_gen_time,
        sd35_gen_time / max(len(sd35_paths), 1),
    )

    # -----------------------------------------------------------------------
    # Stage 3 — FFmpeg video creation  (55 → 65 %)
    # -----------------------------------------------------------------------
    _update(
        job_id, progress=55, stage="video_creation",
        message="Assembling MP4 videos with Ken Burns zoom and crossfade...",
    )
    check_timeout()

    video_paths: dict[str, Path] = create_videos.run_all(
        job_id=job_id,
        flux_images=flux_paths,
        sd35_images=sd35_paths,
    )

    logger.info(
        "[%s] Videos created: %s",
        job_id, list(video_paths.keys()),
    )

    # -----------------------------------------------------------------------
    # Stage 4 — Metric evaluation  (65 → 85 %)
    # -----------------------------------------------------------------------
    _update(
        job_id, progress=65, stage="evaluation",
        message="Computing CLIP Score, BRISQUE, NIQE, SSIM, and LPIPS...",
    )
    check_timeout()

    flux_grouped: dict[int, list[Path]] = _group_images_by_prompt(flux_paths)
    sd35_grouped: dict[int, list[Path]] = _group_images_by_prompt(sd35_paths)

    metrics_df = evaluate.run_all_metrics(
        flux_images=flux_grouped,
        sd35_images=sd35_grouped,
        prompts=[prompt],
        generation_times={
            "flux": flux_gen_time,
            "sd35": sd35_gen_time,
        },
        output_dir=results_dir,
    )

    logger.info(
        "[%s] Evaluation complete: %d metric rows written to metrics.csv.",
        job_id, len(metrics_df),
    )

    # -----------------------------------------------------------------------
    # Stage 5 — Comparison report and charts  (85 → 95 %)
    # -----------------------------------------------------------------------
    _update(
        job_id, progress=85, stage="comparison",
        message="Computing weighted winner and generating charts...",
    )
    check_timeout()

    report: dict = compare.build_comparison_report(
        metrics_df=metrics_df,
        job_id=job_id,
        flux_images=flux_grouped,
        sd35_images=sd35_grouped,
        output_dir=results_dir,
    )

    logger.info(
        "[%s] Comparison: %s wins (FLUX %.4f vs SD3.5 %.4f).",
        job_id,
        report["overall_winner"],
        report["overall_score"]["flux"],
        report["overall_score"]["sd35"],
    )

    # -----------------------------------------------------------------------
    # Finalise — build payload and store result  (95 → 100 %)
    # -----------------------------------------------------------------------
    _update(
        job_id, progress=95, stage="finalising",
        message="Building results payload...",
    )

    payload = _build_payload(
        job_id=job_id,
        prompt=prompt,
        flux_paths=flux_paths,
        sd35_paths=sd35_paths,
        video_paths=video_paths,
        report=report,
        flux_gen_time=flux_gen_time,
        sd35_gen_time=sd35_gen_time,
    )

    set_job_result(job_id, payload)

    elapsed = time.monotonic() - wall_start
    logger.info(
        "[%s] Pipeline complete in %.1fs. Winner: %s.",
        job_id, elapsed, report["overall_winner"],
    )
