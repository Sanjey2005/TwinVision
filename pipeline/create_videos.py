"""
Video creation pipeline for TwinVision.

Assembles PNG images into MP4 videos using FFmpeg via subprocess.
Each model's images become a video with Ken Burns zoom and crossfade
transitions. A side-by-side comparison video is also produced.

FFmpeg must be installed and on PATH (Colab: apt-get install ffmpeg).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

from pipeline.config import (
    OUTPUT_BASE,
    VIDEO_CODEC,
    VIDEO_CROSSFADE_DURATION,
    VIDEO_FPS,
    VIDEO_IMG_DURATION,
    VIDEO_PIX_FMT,
    VIDEO_RESOLUTION,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

_FRAMES_PER_IMAGE: int = int(VIDEO_FPS * VIDEO_IMG_DURATION)   # 90 frames


# ---------------------------------------------------------------------------
# FFmpeg availability check
# ---------------------------------------------------------------------------

def _require_ffmpeg() -> None:
    """Raise a clear error if FFmpeg is not found on PATH.

    Raises:
        RuntimeError: If the ffmpeg binary cannot be located.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg not found on PATH. "
            "Install it with: apt-get install ffmpeg  (Colab/Ubuntu) "
            "or: choco install ffmpeg  (Windows)"
        )


# ---------------------------------------------------------------------------
# Filter-graph builders
# ---------------------------------------------------------------------------

def _build_zoompan_filter(input_label: str, output_label: str) -> str:
    """Return an FFmpeg filter string that applies Ken Burns zoom to one input.

    Zooms from 1.0× to 1.5× over _FRAMES_PER_IMAGE frames, keeping the
    subject centred.

    Args:
        input_label: FFmpeg stream label for this clip's input (e.g. '[0:v]').
        output_label: FFmpeg stream label to assign to the output (e.g. '[v0]').

    Returns:
        Filter string fragment suitable for use inside -filter_complex.
    """
    return (
        f"{input_label}"
        f"scale=1024:1024:force_original_aspect_ratio=increase,"
        f"crop=1024:1024,"
        f"zoompan="
        f"z='min(zoom+0.0015,1.5)':"
        f"d={_FRAMES_PER_IMAGE}:"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"s={VIDEO_RESOLUTION}:"
        f"fps={VIDEO_FPS},"
        f"format={VIDEO_PIX_FMT}"
        f"{output_label}"
    )


def _build_xfade_chain(n: int) -> list[str]:
    """Return a list of xfade filter strings that chain N clips together.

    For N clips each of _IMG_DURATION seconds with _CROSSFADE_DURATION
    overlap, the transition offset for the i-th xfade is:
        offset = i * (_IMG_DURATION - _CROSSFADE_DURATION)

    This accounts for the cumulative time compression introduced by each
    preceding crossfade.

    Args:
        n: Total number of video clips (must be >= 2).

    Returns:
        List of xfade filter strings. The final output label is [x{n-1}].
    """
    parts: list[str] = []
    step = VIDEO_IMG_DURATION - VIDEO_CROSSFADE_DURATION   # 2.5 s per transition step

    prev = "v0"
    for i in range(1, n):
        offset = i * step
        curr = f"x{i}"
        parts.append(
            f"[{prev}][v{i}]"
            f"xfade=transition=fade:"
            f"duration={VIDEO_CROSSFADE_DURATION}:"
            f"offset={offset:.3f}"
            f"[{curr}]"
        )
        prev = curr

    return parts


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def create_video(image_paths: list[Path], output_path: Path) -> Path:
    """Create an MP4 video from a list of images using FFmpeg.

    Each image is displayed for 3 seconds with a Ken Burns zoom effect.
    Consecutive images are joined with 0.5-second crossfade transitions.
    Missing images are skipped with a warning; if all are missing an error
    is raised.

    Args:
        image_paths: Ordered list of PNG image paths to include.
        output_path: Destination path for the output MP4 file.

    Returns:
        Path to the created MP4 file.

    Raises:
        RuntimeError: If FFmpeg is not installed or if FFmpeg returns a
            non-zero exit code.
        ValueError: If no valid (existing) images are provided.
    """
    _require_ffmpeg()

    # Filter missing images, warn for each.
    valid: list[Path] = []
    for p in image_paths:
        if p.exists():
            valid.append(p)
        else:
            logger.warning("Image not found, skipping: %s", p)

    if not valid:
        raise ValueError(
            f"No valid images to create video at '{output_path}'. "
            "All provided paths were missing."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(valid)

    # Build ffmpeg command.
    cmd: list[str] = ["ffmpeg", "-y"]

    # One input per image, held for _IMG_DURATION seconds.
    for p in valid:
        cmd.extend(["-loop", "1", "-t", str(VIDEO_IMG_DURATION), "-i", str(p)])

    # Build the filter_complex string.
    filter_parts: list[str] = []

    # 1. Apply Ken Burns zoom to every input.
    for i in range(n):
        filter_parts.append(_build_zoompan_filter(f"[{i}:v]", f"[v{i}]"))

    # 2. Chain xfade transitions (skip if only one image).
    if n > 1:
        filter_parts.extend(_build_xfade_chain(n))
        final_label = f"x{n - 1}"
    else:
        final_label = "v0"

    filter_str = ";".join(filter_parts)

    cmd.extend([
        "-filter_complex", filter_str,
        "-map", f"[{final_label}]",
        "-c:v", VIDEO_CODEC,
        "-pix_fmt", VIDEO_PIX_FMT,
        "-r", str(VIDEO_FPS),
        str(output_path),
    ])

    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("FFmpeg stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}) creating '{output_path}'. "
            f"See logs for stderr output."
        )

    logger.info("Created video: %s (%d images)", output_path, n)
    return output_path


def create_side_by_side_video(
    flux_video: Path,
    sd_video: Path,
    output_path: Path,
) -> Path:
    """Stack two videos side by side into a single comparison MP4.

    Both inputs must exist. If their durations differ FFmpeg will stop at
    the shorter one.

    Args:
        flux_video: Path to the FLUX.1 MP4 file.
        sd_video: Path to the SD3.5 MP4 file.
        output_path: Destination path for the comparison MP4.

    Returns:
        Path to the created comparison MP4.

    Raises:
        FileNotFoundError: If either input video does not exist.
        RuntimeError: If FFmpeg is not installed or returns a non-zero exit.
    """
    _require_ffmpeg()

    for video_path in (flux_video, sd_video):
        if not video_path.exists():
            raise FileNotFoundError(
                f"Cannot create side-by-side video: '{video_path}' not found."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-i", str(flux_video),
        "-i", str(sd_video),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", VIDEO_CODEC,
        "-pix_fmt", VIDEO_PIX_FMT,
        str(output_path),
    ]

    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("FFmpeg stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}) creating side-by-side "
            f"'{output_path}'. See logs for stderr output."
        )

    logger.info("Created comparison video: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _group_by_prompt(images: list[Path]) -> dict[int, list[Path]]:
    """Parse filenames and group image paths by prompt index.

    Expected filename format: prompt_{i}_img_{j}.png

    Missing or unparseable filenames are skipped with a warning.

    Args:
        images: Flat list of image Paths from generate_images.run_all().

    Returns:
        Dict mapping prompt index (int) to an ordered list of image Paths
        sorted by image index j.
    """
    grouped: dict[int, list[Path]] = {}
    for p in images:
        parts = p.stem.split("_")   # ['prompt', '{i}', 'img', '{j}']
        try:
            prompt_idx = int(parts[1])
        except (IndexError, ValueError):
            logger.warning("Cannot parse prompt index from filename '%s' — skipping.", p.name)
            continue
        grouped.setdefault(prompt_idx, []).append(p)

    # Sort each group by image index so the video is in generation order.
    for idx in grouped:
        grouped[idx].sort(key=lambda p: int(p.stem.split("_")[3]))

    return grouped


def run_all(
    job_id: str,
    flux_images: list[Path],
    sd35_images: list[Path],
) -> dict[str, Path]:
    """Create all model and comparison videos for a job.

    Groups the flat image lists from generate_images.run_all() by prompt
    index, creates one video per model per prompt, then stitches each pair
    into a side-by-side comparison video.

    Output files are written to OUTPUT_BASE/{job_id}/videos/:
        flux_prompt_{i}.mp4
        sd35_prompt_{i}.mp4
        comparison_prompt_{i}.mp4

    Args:
        job_id: Unique job identifier used to scope the output directory.
        flux_images: Flat list of all FLUX.1 image Paths (all prompts).
        sd35_images: Flat list of all SD3.5 image Paths (all prompts).

    Returns:
        Dict mapping descriptive keys (e.g. 'flux_prompt_0',
        'comparison_prompt_0') to the Path of each created MP4.
    """
    videos_dir: Path = OUTPUT_BASE / job_id / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    flux_by_prompt = _group_by_prompt(flux_images)
    sd35_by_prompt = _group_by_prompt(sd35_images)
    all_indices = sorted(set(flux_by_prompt) | set(sd35_by_prompt))

    output_paths: dict[str, Path] = {}

    for prompt_idx in tqdm(all_indices, desc="Creating videos", unit="prompt"):
        flux_imgs = flux_by_prompt.get(prompt_idx, [])
        sd35_imgs = sd35_by_prompt.get(prompt_idx, [])

        flux_out = videos_dir / f"flux_prompt_{prompt_idx}.mp4"
        sd35_out = videos_dir / f"sd35_prompt_{prompt_idx}.mp4"
        comp_out = videos_dir / f"comparison_prompt_{prompt_idx}.mp4"

        # FLUX video
        if flux_imgs:
            try:
                create_video(flux_imgs, flux_out)
                output_paths[f"flux_prompt_{prompt_idx}"] = flux_out
            except (RuntimeError, ValueError) as exc:
                logger.error("Failed to create FLUX video for prompt_%d: %s", prompt_idx, exc)

        # SD3.5 video
        if sd35_imgs:
            try:
                create_video(sd35_imgs, sd35_out)
                output_paths[f"sd35_prompt_{prompt_idx}"] = sd35_out
            except (RuntimeError, ValueError) as exc:
                logger.error("Failed to create SD3.5 video for prompt_%d: %s", prompt_idx, exc)

        # Side-by-side comparison (only if both individual videos were created)
        if flux_out.exists() and sd35_out.exists():
            try:
                create_side_by_side_video(flux_out, sd35_out, comp_out)
                output_paths[f"comparison_prompt_{prompt_idx}"] = comp_out
            except (RuntimeError, FileNotFoundError) as exc:
                logger.error(
                    "Failed to create comparison video for prompt_%d: %s",
                    prompt_idx,
                    exc,
                )

    logger.info(
        "Video creation complete: %d files in '%s'.",
        len(output_paths),
        videos_dir,
    )
    return output_paths
