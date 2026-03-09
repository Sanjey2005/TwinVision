"""
Evaluation metrics pipeline for TwinVision.

Computes five quality and consistency metrics for FLUX.1-schnell and
Stable Diffusion 3.5 Large generated images:

  - CLIP Score   : text-image alignment (higher = better)
  - BRISQUE      : no-reference quality (lower = better)
  - NIQE         : statistical naturalness (lower = better)
  - SSIM         : inter-image structural consistency (higher = better)
  - LPIPS        : inter-image perceptual consistency (lower = better)

Metric models are loaded once and cached at module level. Results are
serialised to a pandas DataFrame and written to metrics.csv.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyiqa
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore

from pipeline.config import OUTPUT_BASE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Scores for a single metric applied to one model's image set.

    Attributes:
        per_image_scores: Score for each image (or image pair for SSIM/LPIPS).
        average_score: Mean of per_image_scores.
    """

    per_image_scores: list[float]
    average_score: float


@dataclass
class ModelMetrics:
    """All five evaluation metrics for one (model, prompt) combination.

    Attributes:
        clip_score: Text-image alignment scores.
        brisque: BRISQUE no-reference quality scores.
        niqe: NIQE statistical naturalness scores.
        ssim: Consecutive-pair structural similarity scores.
        lpips: Consecutive-pair perceptual similarity scores.
    """

    clip_score: MetricResult
    brisque: MetricResult
    niqe: MetricResult
    ssim: MetricResult
    lpips: MetricResult


@dataclass
class EvaluationResults:
    """Aggregated evaluation results for both models across all prompts.

    Attributes:
        flux: Dict mapping prompt_index to ModelMetrics for FLUX.1.
        sd35: Dict mapping prompt_index to ModelMetrics for SD3.5.
    """

    flux: dict[int, ModelMetrics]
    sd35: dict[int, ModelMetrics]


# ---------------------------------------------------------------------------
# Module-level metric cache
# ---------------------------------------------------------------------------

_clip_metric: CLIPScore | None = None
_ssim_metric: StructuralSimilarityIndexMeasure | None = None
_lpips_metric: LearnedPerceptualImagePatchSimilarity | None = None
_brisque_metric: Any | None = None
_niqe_metric: Any | None = None


def _get_clip(device: str) -> CLIPScore:
    """Return (or initialise) the cached CLIPScore metric.

    Args:
        device: Torch device string.

    Returns:
        CLIPScore instance on the requested device.
    """
    global _clip_metric
    if _clip_metric is None:
        logger.info("Loading CLIPScore model (openai/clip-vit-large-patch14)...")
        _clip_metric = CLIPScore(
            model_name_or_path="openai/clip-vit-large-patch14"
        ).to(device)
    return _clip_metric


def _get_ssim(device: str) -> StructuralSimilarityIndexMeasure:
    """Return (or initialise) the cached SSIM metric.

    Args:
        device: Torch device string.

    Returns:
        StructuralSimilarityIndexMeasure instance on the requested device.
    """
    global _ssim_metric
    if _ssim_metric is None:
        _ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return _ssim_metric


def _get_lpips(device: str) -> LearnedPerceptualImagePatchSimilarity:
    """Return (or initialise) the cached LPIPS (AlexNet) metric.

    Args:
        device: Torch device string.

    Returns:
        LearnedPerceptualImagePatchSimilarity instance on the requested device.
    """
    global _lpips_metric
    if _lpips_metric is None:
        logger.info("Loading LPIPS metric (AlexNet)...")
        _lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex"
        ).to(device)
    return _lpips_metric


def _get_brisque(device: str) -> Any:
    """Return (or initialise) the cached pyiqa BRISQUE metric.

    Args:
        device: Torch device string.

    Returns:
        pyiqa BRISQUE metric callable.
    """
    global _brisque_metric
    if _brisque_metric is None:
        _brisque_metric = pyiqa.create_metric("brisque", device=device)
    return _brisque_metric


def _get_niqe(device: str) -> Any:
    """Return (or initialise) the cached pyiqa NIQE metric.

    Args:
        device: Torch device string.

    Returns:
        pyiqa NIQE metric callable.
    """
    global _niqe_metric
    if _niqe_metric is None:
        _niqe_metric = pyiqa.create_metric("niqe", device=device)
    return _niqe_metric


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------


def _load_float_tensor(path: Path, device: str) -> torch.Tensor:
    """Load an image as a [1, 3, H, W] float32 tensor normalised to [0, 1].

    Args:
        path: Path to the PNG image.
        device: Target torch device.

    Returns:
        Float32 tensor of shape [1, 3, H, W] with values in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(img).unsqueeze(0)   # [1, 3, H, W], float [0, 1]
    return tensor.to(device)


def _load_uint8_tensor(path: Path, device: str) -> torch.Tensor:
    """Load an image as a [1, 3, H, W] uint8 tensor in [0, 255] for CLIPScore.

    Args:
        path: Path to the PNG image.
        device: Target torch device.

    Returns:
        Uint8 tensor of shape [1, 3, H, W] with values in [0, 255].
    """
    img = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(img)                        # [3, H, W], float [0, 1]
    uint8 = (tensor * 255).to(torch.uint8).unsqueeze(0)  # [1, 3, H, W], uint8
    return uint8.to(device)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_clip_score(
    images: list[Path],
    prompt: str,
    device: str = "cuda",
) -> dict:
    """Compute CLIP Score for each image against the generation prompt.

    Each image is evaluated individually by resetting the metric between
    calls, giving a per-image alignment score.

    Args:
        images: List of image Paths to evaluate.
        prompt: The text prompt used to generate the images.
        device: Torch device string.

    Returns:
        Dict with keys:
            'per_image_scores' (list[float]): One score per image.
            'average_score' (float): Mean of per_image_scores.
    """
    metric = _get_clip(device)
    scores: list[float] = []

    for path in tqdm(images, desc="    CLIP", unit="img", leave=False):
        img_tensor = _load_uint8_tensor(path, device)
        metric.reset()
        metric.update(img_tensor, [prompt])
        scores.append(float(metric.compute().item()))

    avg = sum(scores) / len(scores) if scores else 0.0
    logger.debug("CLIP: %d scores, avg=%.4f", len(scores), avg)
    return {"per_image_scores": scores, "average_score": avg}


def compute_brisque(
    images: list[Path],
    device: str = "cuda",
) -> dict:
    """Compute BRISQUE no-reference quality score for each image.

    Lower scores indicate better perceptual quality.

    Args:
        images: List of image Paths to evaluate.
        device: Torch device string.

    Returns:
        Dict with keys:
            'per_image_scores' (list[float]): One score per image.
            'average_score' (float): Mean of per_image_scores.
    """
    metric = _get_brisque(device)
    scores: list[float] = []

    for path in tqdm(images, desc="    BRISQUE", unit="img", leave=False):
        img_tensor = _load_float_tensor(path, device)
        score_tensor = metric(img_tensor)
        scores.append(float(score_tensor.item()))

    avg = sum(scores) / len(scores) if scores else 0.0
    logger.debug("BRISQUE: %d scores, avg=%.4f", len(scores), avg)
    return {"per_image_scores": scores, "average_score": avg}


def compute_niqe(
    images: list[Path],
    device: str = "cuda",
) -> dict:
    """Compute NIQE statistical naturalness score for each image.

    Lower scores indicate better statistical naturalness.

    Args:
        images: List of image Paths to evaluate.
        device: Torch device string.

    Returns:
        Dict with keys:
            'per_image_scores' (list[float]): One score per image.
            'average_score' (float): Mean of per_image_scores.
    """
    metric = _get_niqe(device)
    scores: list[float] = []

    for path in tqdm(images, desc="    NIQE", unit="img", leave=False):
        img_tensor = _load_float_tensor(path, device)
        score_tensor = metric(img_tensor)
        scores.append(float(score_tensor.item()))

    avg = sum(scores) / len(scores) if scores else 0.0
    logger.debug("NIQE: %d scores, avg=%.4f", len(scores), avg)
    return {"per_image_scores": scores, "average_score": avg}


def compute_ssim_consistency(
    images: list[Path],
    device: str = "cuda",
) -> dict:
    """Compute SSIM between consecutive image pairs to measure consistency.

    For N images produces N-1 scores, one per consecutive pair.
    Returns [1.0] sentinel when fewer than 2 images are provided.

    Args:
        images: List of image Paths to evaluate (same prompt, same model).
        device: Torch device string.

    Returns:
        Dict with keys:
            'per_image_scores' (list[float]): N-1 SSIM values in [0, 1].
            'average_score' (float): Mean of per_image_scores.
    """
    if len(images) < 2:
        logger.debug("SSIM: fewer than 2 images — returning sentinel [1.0].")
        return {"per_image_scores": [1.0], "average_score": 1.0}

    metric = _get_ssim(device)
    scores: list[float] = []

    for path_a, path_b in tqdm(
        zip(images[:-1], images[1:]),
        desc="    SSIM",
        unit="pair",
        total=len(images) - 1,
        leave=False,
    ):
        t_a = _load_float_tensor(path_a, device)   # [1, 3, H, W] in [0, 1]
        t_b = _load_float_tensor(path_b, device)
        metric.reset()
        metric.update(t_a, t_b)
        scores.append(float(metric.compute().item()))

    avg = sum(scores) / len(scores) if scores else 0.0
    logger.debug("SSIM: %d pair scores, avg=%.4f", len(scores), avg)
    return {"per_image_scores": scores, "average_score": avg}


def compute_lpips_consistency(
    images: list[Path],
    device: str = "cuda",
) -> dict:
    """Compute LPIPS between consecutive image pairs to measure perceptual consistency.

    For N images produces N-1 scores, one per consecutive pair.
    Returns [0.0] sentinel when fewer than 2 images are provided.
    Images are normalised from [0, 1] to [-1, 1] as required by LPIPS.

    Args:
        images: List of image Paths to evaluate (same prompt, same model).
        device: Torch device string.

    Returns:
        Dict with keys:
            'per_image_scores' (list[float]): N-1 LPIPS values in [0, 1].
            'average_score' (float): Mean of per_image_scores.
    """
    if len(images) < 2:
        logger.debug("LPIPS: fewer than 2 images — returning sentinel [0.0].")
        return {"per_image_scores": [0.0], "average_score": 0.0}

    metric = _get_lpips(device)
    scores: list[float] = []

    for path_a, path_b in tqdm(
        zip(images[:-1], images[1:]),
        desc="    LPIPS",
        unit="pair",
        total=len(images) - 1,
        leave=False,
    ):
        # LPIPS expects float tensors in [-1, 1].
        t_a = _load_float_tensor(path_a, device) * 2.0 - 1.0
        t_b = _load_float_tensor(path_b, device) * 2.0 - 1.0
        metric.reset()
        metric.update(t_a, t_b)
        scores.append(float(metric.compute().item()))

    avg = sum(scores) / len(scores) if scores else 0.0
    logger.debug("LPIPS: %d pair scores, avg=%.4f", len(scores), avg)
    return {"per_image_scores": scores, "average_score": avg}


# ---------------------------------------------------------------------------
# Internal: evaluate one (model, prompt) combination
# ---------------------------------------------------------------------------


def _evaluate_images(
    images: list[Path],
    prompt: str,
    device: str,
) -> ModelMetrics:
    """Run all five metrics for one model's images on one prompt.

    Args:
        images: Ordered list of image Paths for one (model, prompt).
        prompt: The text prompt used to generate these images.
        device: Torch device string.

    Returns:
        ModelMetrics populated with MetricResult for each of the five metrics.
    """
    return ModelMetrics(
        clip_score=MetricResult(**compute_clip_score(images, prompt, device)),
        brisque=MetricResult(**compute_brisque(images, device)),
        niqe=MetricResult(**compute_niqe(images, device)),
        ssim=MetricResult(**compute_ssim_consistency(images, device)),
        lpips=MetricResult(**compute_lpips_consistency(images, device)),
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_all_metrics(
    flux_images: dict[int, list[Path]],
    sd35_images: dict[int, list[Path]],
    prompts: list[str],
    generation_times: dict[str, float] | None = None,
    output_dir: Path | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Evaluate both models across all prompts and write metrics.csv.

    Iterates over every (model, prompt_index) combination, runs all five
    metrics, and builds a DataFrame with one row per (prompt_index, model,
    metric). Writes the result to output_dir/metrics.csv.

    Args:
        flux_images: Dict mapping prompt_index → list of FLUX image Paths.
        sd35_images: Dict mapping prompt_index → list of SD3.5 image Paths.
        prompts: Full list of text prompts; indexed by prompt_index.
        generation_times: Optional dict with total generation seconds per model
            (keys 'flux', 'sd35') from generate_images.get_generation_times().
            Populates the generation_time column; defaults to 0.0 if absent.
        output_dir: Directory to write metrics.csv.  Defaults to
            OUTPUT_BASE/results/.
        device: Torch device string.  Auto-detected (cuda if available) if None.

    Returns:
        DataFrame with columns:
            prompt_index, prompt_text, model, metric,
            per_image_scores, average_score, generation_time.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if generation_times is None:
        generation_times = {}

    save_dir: Path = output_dir if output_dir is not None else OUTPUT_BASE / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_prompt_indices = sorted(set(flux_images) | set(sd35_images))
    model_sources: list[tuple[str, dict[int, list[Path]]]] = [
        ("flux", flux_images),
        ("sd35", sd35_images),
    ]

    rows: list[dict] = []

    for model_name, images_by_prompt in tqdm(
        model_sources, desc="Evaluating models", unit="model"
    ):
        gen_time = generation_times.get(model_name, 0.0)

        for prompt_idx in tqdm(
            all_prompt_indices,
            desc=f"  {model_name.upper()} prompts",
            unit="prompt",
            leave=False,
        ):
            prompt_text = prompts[prompt_idx] if prompt_idx < len(prompts) else ""
            images = images_by_prompt.get(prompt_idx, [])

            if not images:
                logger.warning(
                    "No images for model=%s prompt_idx=%d — skipping.",
                    model_name,
                    prompt_idx,
                )
                continue

            logger.info(
                "Evaluating %s prompt_%d (%d images)...",
                model_name.upper(),
                prompt_idx,
                len(images),
            )

            metrics = _evaluate_images(images, prompt_text, device)

            metric_map: dict[str, MetricResult] = {
                "clip_score": metrics.clip_score,
                "brisque": metrics.brisque,
                "niqe": metrics.niqe,
                "ssim": metrics.ssim,
                "lpips": metrics.lpips,
            }

            for metric_name, result in metric_map.items():
                rows.append(
                    {
                        "prompt_index": prompt_idx,
                        "prompt_text": prompt_text,
                        "model": model_name,
                        "metric": metric_name,
                        # json.dumps ensures valid JSON, not Python repr.
                        "per_image_scores": json.dumps(
                            [round(s, 6) for s in result.per_image_scores]
                        ),
                        "average_score": round(result.average_score, 6),
                        "generation_time": round(gen_time, 3),
                    }
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "prompt_index",
            "prompt_text",
            "model",
            "metric",
            "per_image_scores",
            "average_score",
            "generation_time",
        ],
    )

    csv_path = save_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Metrics CSV written: %s (%d rows).", csv_path, len(df))
    return df
