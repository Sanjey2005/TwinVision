"""
TwinVision CLI runner.

Runs the full prompt-to-video pipeline locally without starting the API
server. All five pipeline stages are called directly (image generation,
video creation, evaluation, comparison) with timestamped log output.

Usage
─────
  python run_pipeline_cli.py                          # all 5 config prompts
  python run_pipeline_cli.py --test                   # 2 images/model, quick run
  python run_pipeline_cli.py --prompt "custom text"   # one custom prompt
  python run_pipeline_cli.py --n-images 4             # 4 images per model
  python run_pipeline_cli.py --skip-generation        # reuse images from prior run
  python run_pipeline_cli.py --skip-video             # skip FFmpeg step
  python run_pipeline_cli.py --skip-generation --skip-video --job-id abc12345
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env (HF_TOKEN) before importing any pipeline modules.
load_dotenv()

from pipeline import compare, create_videos, evaluate, generate_images
from pipeline.config import OUTPUT_BASE, PROMPTS

# ---------------------------------------------------------------------------
# Logging — ISO timestamp + level + module name
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("twinvision.cli")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="run_pipeline_cli.py",
        description="TwinVision: prompt-to-video AI model comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        metavar="TEXT",
        help="Single custom prompt. Omit to run all prompts from config.",
    )
    parser.add_argument(
        "--n-images", "-n",
        type=int,
        default=7,
        dest="n_images",
        metavar="N",
        help="Number of images to generate per model (default: 7).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick-verify mode: 2 images per model instead of --n-images.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        dest="skip_generation",
        help="Skip image generation and load PNGs from a previous run. "
             "Requires --job-id.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        dest="skip_video",
        help="Skip FFmpeg video creation; still runs evaluation and comparison.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        dest="job_id",
        metavar="ID",
        help="Reuse output from a previous run (for --skip-generation). "
             "A new ID is auto-generated when omitted.",
    )
    return parser


# ---------------------------------------------------------------------------
# Image-grouping helper (same logic as orchestrator)
# ---------------------------------------------------------------------------

def _group_by_prompt(images: list[Path]) -> dict[int, list[Path]]:
    """Parse prompt_{i}_img_{j}.png filenames and group by prompt index.

    Args:
        images: Flat list of image Paths.

    Returns:
        Dict mapping prompt_index → sorted list of image Paths.
    """
    grouped: dict[int, list[Path]] = {}
    for path in images:
        parts = path.stem.split("_")
        try:
            idx = int(parts[1])
        except (IndexError, ValueError):
            logger.warning("Cannot parse prompt index from '%s' — skipping.", path.name)
            continue
        grouped.setdefault(idx, []).append(path)
    for idx in grouped:
        grouped[idx].sort(key=lambda p: int(p.stem.split("_")[3]))
    return grouped


# ---------------------------------------------------------------------------
# Stage banner
# ---------------------------------------------------------------------------

def _banner(stage_num: int, total: int, title: str) -> None:
    """Print a stage divider with timestamp.

    Args:
        stage_num: Current stage number (1-based).
        total: Total number of stages.
        title: Short stage description.
    """
    logger.info("─" * 60)
    logger.info("  Stage %d/%d — %s", stage_num, total, title)
    logger.info("─" * 60)


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Execute the pipeline according to parsed CLI flags.

    Args:
        args: Parsed namespace from _build_parser().
    """
    # ---- resolve parameters ------------------------------------------------
    job_id: str = args.job_id or uuid.uuid4().hex[:8]
    n_images: int = 2 if args.test else args.n_images
    prompts: list[str] = [args.prompt] if args.prompt else list(PROMPTS)
    total_stages: int = 4 - int(args.skip_generation) - int(args.skip_video)

    base_dir:    Path = OUTPUT_BASE / job_id
    flux_dir:    Path = base_dir / "flux"
    sd35_dir:    Path = base_dir / "sd35"
    results_dir: Path = base_dir / "results"

    for d in (flux_dir, sd35_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Job ID        : %s", job_id)
    logger.info("Prompts       : %d", len(prompts))
    logger.info("Images/model  : %d%s", n_images, "  (test mode)" if args.test else "")
    logger.info("Skip generation: %s", args.skip_generation)
    logger.info("Skip video     : %s", args.skip_video)
    logger.info("Output dir     : %s", base_dir.resolve())

    wall_start = time.monotonic()
    stage_idx  = 0
    flux_paths: list[Path] = []
    sd35_paths: list[Path] = []

    # ------------------------------------------------------------------ #
    # Stage A — Image generation (or load from disk)
    # ------------------------------------------------------------------ #
    if args.skip_generation:
        logger.info("Skipping image generation — loading existing PNGs from %s", base_dir)
        flux_paths = sorted(flux_dir.glob("prompt_*.png"))
        sd35_paths = sorted(sd35_dir.glob("prompt_*.png"))
        if not flux_paths or not sd35_paths:
            logger.error(
                "No PNG files found in %s or %s. "
                "Run without --skip-generation first.",
                flux_dir, sd35_dir,
            )
            sys.exit(1)
        logger.info("Loaded %d FLUX + %d SD3.5 images.", len(flux_paths), len(sd35_paths))
    else:
        stage_idx += 1
        _banner(stage_idx, total_stages, "Image Generation")

        for prompt_idx, prompt in enumerate(prompts):
            logger.info("Prompt %d/%d: %s", prompt_idx + 1, len(prompts), prompt[:80])

            paths = generate_images.generate_images(
                prompt=prompt,
                model_name="flux",
                num_images=n_images,
                prompt_index=prompt_idx,
                output_dir=flux_dir,
            )
            flux_paths.extend(paths)

            paths = generate_images.generate_images(
                prompt=prompt,
                model_name="sd35",
                num_images=n_images,
                prompt_index=prompt_idx,
                output_dir=sd35_dir,
            )
            sd35_paths.extend(paths)

        gen_times = generate_images.get_generation_times()
        logger.info(
            "Generation complete — FLUX: %.1fs  SD3.5: %.1fs",
            gen_times.get("flux", 0.0),
            gen_times.get("sd35", 0.0),
        )

    flux_grouped = _group_by_prompt(flux_paths)
    sd35_grouped = _group_by_prompt(sd35_paths)

    # ------------------------------------------------------------------ #
    # Stage B — Video creation
    # ------------------------------------------------------------------ #
    video_paths: dict[str, Path] = {}
    if args.skip_video:
        logger.info("Skipping video creation.")
    else:
        stage_idx += 1
        _banner(stage_idx, total_stages, "Video Creation (FFmpeg)")

        video_paths = create_videos.run_all(
            job_id=job_id,
            flux_images=flux_paths,
            sd35_images=sd35_paths,
        )
        logger.info("Videos created: %d files.", len(video_paths))

    # ------------------------------------------------------------------ #
    # Stage C — Metric evaluation
    # ------------------------------------------------------------------ #
    stage_idx += 1
    _banner(stage_idx, total_stages, "Metric Evaluation")

    gen_times = generate_images.get_generation_times()
    metrics_df = evaluate.run_all_metrics(
        flux_images=flux_grouped,
        sd35_images=sd35_grouped,
        prompts=prompts,
        generation_times=gen_times,
        output_dir=results_dir,
    )
    logger.info("Evaluation complete — %d metric rows written.", len(metrics_df))

    # ------------------------------------------------------------------ #
    # Stage D — Comparison report
    # ------------------------------------------------------------------ #
    stage_idx += 1
    _banner(stage_idx, total_stages, "Comparison Report & Charts")

    report = compare.build_comparison_report(
        metrics_df=metrics_df,
        job_id=job_id,
        flux_images=flux_grouped,
        sd35_images=sd35_grouped,
        output_dir=results_dir,
    )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    elapsed = time.monotonic() - wall_start
    minutes, seconds = divmod(int(elapsed), 60)

    logger.info("─" * 60)
    logger.info("  TwinVision Pipeline Complete")
    logger.info("─" * 60)
    logger.info("  Total time    : %dm %02ds", minutes, seconds)
    logger.info("  Winner        : %s", report["overall_winner"])
    logger.info(
        "  FLUX score    : %.4f",
        report["overall_score"]["flux"],
    )
    logger.info(
        "  SD3.5 score   : %.4f",
        report["overall_score"]["sd35"],
    )
    logger.info("  Results dir   : %s", results_dir.resolve())
    logger.info("─" * 60)

    # Per-metric winner table
    logger.info("  Per-metric winners:")
    for metric, info in report["metric_winners"].items():
        logger.info(
            "    %-12s  winner=%-6s  flux=%-8.4f  sd35=%-8.4f  diff=%.1f%%",
            metric,
            info["winner"],
            info["flux"],
            info["sd35"],
            info["diff_pct"],
        )
    logger.info("─" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the pipeline.

    Returns:
        Exits with code 0 on success, 1 on argument/validation errors.
    """
    parser = _build_parser()
    args   = parser.parse_args()

    if args.skip_generation and args.job_id is None:
        parser.error(
            "--skip-generation requires --job-id to identify which run's "
            "images to load."
        )

    try:
        run(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
