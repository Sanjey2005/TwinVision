"""
Image generation pipeline for TwinVision.

Loads FLUX.1-schnell and Stable Diffusion 3.5 Large models and generates
images from text prompts, saving outputs as PNG files.

Pipelines are cached after first load so models are never reloaded within
the same process. Generation times are tracked and exposed via
get_generation_times() for the orchestrator.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from tqdm import tqdm

from pipeline.config import MODELS, OUTPUT_BASE


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Loaded pipelines keyed by model name; populated lazily on first use.
_pipeline_cache: dict[str, Any] = {}

# Total generation time (seconds) per model, reset at the start of run_all().
_generation_times: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_hf_token() -> str:
    """Return the HuggingFace token from the environment.

    Returns:
        HF_TOKEN string.

    Raises:
        EnvironmentError: If HF_TOKEN is not set.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Get your token at https://huggingface.co/settings/tokens "
            "and add HF_TOKEN=<your_token> to your .env file."
        )
    return token


def _detect_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'.

    Returns:
        Device string suitable for torch.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_inference(
    pipeline: Any,
    model_name: str,
    prompt: str,
    width: int,
    height: int,
) -> Any:
    """Run a single forward pass through a diffusers pipeline.

    Args:
        pipeline: Loaded diffusers pipeline object (FluxPipeline or
            StableDiffusion3Pipeline).
        model_name: Either 'flux' or 'sd35'; used to look up inference
            hyperparameters from config.
        prompt: Text prompt to condition the generation on.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        PIL.Image object produced by the pipeline.
    """
    cfg = MODELS[model_name]
    result = pipeline(
        prompt=prompt,
        num_inference_steps=cfg["inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        width=width,
        height=height,
    )
    return result.images[0]


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------

def load_flux_pipeline(device: str) -> FluxPipeline:
    """Load FLUX.1-schnell in fp16 and place it on device.

    The loaded pipeline is stored in the module-level cache; subsequent
    calls return the cached instance without reloading from disk.

    Args:
        device: Target device string (e.g. 'cuda', 'cpu').

    Returns:
        Ready-to-use FluxPipeline instance.
    """
    if "flux" in _pipeline_cache:
        logger.info("FLUX.1 pipeline already loaded — reusing cached instance.")
        return _pipeline_cache["flux"]

    cfg = MODELS["flux"]
    logger.info("Loading FLUX.1 pipeline from '%s' ...", cfg["model_id"])
    token = _get_hf_token()

    pipeline: FluxPipeline = FluxPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16,
        token=token,
    )
    pipeline = pipeline.to(device)
    _pipeline_cache["flux"] = pipeline
    logger.info("FLUX.1 pipeline ready on '%s'.", device)
    return pipeline


def load_sd35_pipeline(device: str) -> StableDiffusion3Pipeline:
    """Load Stable Diffusion 3.5 Large in fp16 and place it on device.

    SD3.5 Large is a gated HuggingFace model; HF_TOKEN must have access.
    The loaded pipeline is stored in the module-level cache.

    Args:
        device: Target device string (e.g. 'cuda', 'cpu').

    Returns:
        Ready-to-use StableDiffusion3Pipeline instance.
    """
    if "sd35" in _pipeline_cache:
        logger.info("SD3.5 pipeline already loaded — reusing cached instance.")
        return _pipeline_cache["sd35"]

    cfg = MODELS["sd35"]
    logger.info("Loading SD3.5 Large pipeline from '%s' ...", cfg["model_id"])
    token = _get_hf_token()

    pipeline: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16,
        token=token,
    )
    pipeline = pipeline.to(device)
    _pipeline_cache["sd35"] = pipeline
    logger.info("SD3.5 pipeline ready on '%s'.", device)
    return pipeline


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_images(
    prompt: str,
    model_name: str,
    num_images: int = 7,
    prompt_index: int = 0,
    output_dir: Path | None = None,
    device: str | None = None,
) -> list[Path]:
    """Generate images for a single prompt using the specified model.

    Loads the pipeline on first call (cached thereafter). Saves each image
    as a PNG and accumulates per-model generation time into the module-level
    _generation_times dict.

    On torch.cuda.OutOfMemoryError the GPU cache is cleared and generation
    is retried with enable_model_cpu_offload() enabled.

    Args:
        prompt: Text prompt used to condition image generation.
        model_name: Either 'flux' or 'sd35'.
        num_images: Number of images to generate (default 7).
        prompt_index: Zero-based index of this prompt in the batch; used to
            name output files (prompt_{prompt_index}_img_{j}.png).
        output_dir: Directory to save generated PNGs. Defaults to the
            model's output_dir from config (e.g. output/flux/).
        device: Torch device string. Auto-detected if None.

    Returns:
        Ordered list of Paths to saved PNG files.
    """
    if device is None:
        device = _detect_device()

    cfg = MODELS[model_name]
    width, height = cfg["resolution"]

    resolved_output_dir: Path = output_dir if output_dir is not None else OUTPUT_BASE / model_name
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline (uses module-level cache after first call).
    if model_name == "flux":
        pipeline = load_flux_pipeline(device)
    else:
        pipeline = load_sd35_pipeline(device)

    saved_paths: list[Path] = []
    run_total_time: float = 0.0

    for img_idx in tqdm(
        range(num_images),
        desc=f"  {model_name.upper()} prompt_{prompt_index}",
        unit="img",
        leave=False,
    ):
        out_path = resolved_output_dir / f"prompt_{prompt_index}_img_{img_idx}.png"
        t_start = time.perf_counter()

        try:
            image = _run_inference(pipeline, model_name, prompt, width, height)

        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "CUDA OOM on %s image %d — clearing cache and retrying with CPU offload.",
                model_name.upper(),
                img_idx,
            )
            torch.cuda.empty_cache()
            pipeline.enable_model_cpu_offload()
            # After offload is enabled the pipeline manages device placement;
            # do not call .to(device) again.
            image = _run_inference(pipeline, model_name, prompt, width, height)

        elapsed = time.perf_counter() - t_start
        run_total_time += elapsed

        image.save(out_path)
        logger.debug("Saved %s in %.2fs.", out_path, elapsed)
        saved_paths.append(out_path)

    # Accumulate into module-level dict so orchestrator can read total times.
    _generation_times[model_name] = (
        _generation_times.get(model_name, 0.0) + run_total_time
    )

    logger.info(
        "%s: %d images for prompt_%d in %.1fs (avg %.2fs/img).",
        model_name.upper(),
        num_images,
        prompt_index,
        run_total_time,
        run_total_time / max(num_images, 1),
    )
    return saved_paths


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all(
    prompts: list[str],
    test_mode: bool = False,
    output_base: Path | None = None,
    device: str | None = None,
) -> dict[str, list[Path]]:
    """Generate images for all prompts using both FLUX.1-schnell and SD3.5.

    Iterates over both models and every prompt, calling generate_images() for
    each combination. Resets the module-level generation-time accumulator at
    the start so times reflect only this run.

    Args:
        prompts: List of text prompts to process.
        test_mode: If True, generate 2 images per model instead of 7 (fast
            verification run).
        output_base: Root output directory. Each model's images are written to
            output_base/{model_name}/. Defaults to OUTPUT_BASE from config.
        device: Torch device string. Auto-detected if None.

    Returns:
        Dict mapping model name ('flux', 'sd35') to a flat list of all
        generated image Paths (across all prompts, in prompt order).
    """
    global _generation_times
    _generation_times = {}  # reset so get_generation_times() reflects this run only.

    if device is None:
        device = _detect_device()

    num_images: int = 2 if test_mode else 7
    base: Path = output_base if output_base is not None else OUTPUT_BASE

    results: dict[str, list[Path]] = {"flux": [], "sd35": []}

    for model_name in tqdm(["flux", "sd35"], desc="Models", unit="model"):
        logger.info("=== Starting %s generation (%d prompts × %d images) ===",
                    model_name.upper(), len(prompts), num_images)

        model_out_dir: Path = base / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        for prompt_idx, prompt in enumerate(
            tqdm(
                prompts,
                desc=f"{model_name.upper()} prompts",
                unit="prompt",
                leave=False,
            )
        ):
            paths = generate_images(
                prompt=prompt,
                model_name=model_name,
                num_images=num_images,
                prompt_index=prompt_idx,
                output_dir=model_out_dir,
                device=device,
            )
            results[model_name].extend(paths)

        model_time = _generation_times.get(model_name, 0.0)
        logger.info(
            "%s: all prompts done — total %.1fs (%.2fs/img avg).",
            model_name.upper(),
            model_time,
            model_time / max(len(prompts) * num_images, 1),
        )

    return results


# ---------------------------------------------------------------------------
# Timing accessor (for orchestrator / compare modules)
# ---------------------------------------------------------------------------

def get_generation_times() -> dict[str, float]:
    """Return total generation times per model from the last run_all() call.

    Returns:
        Dict mapping model name ('flux', 'sd35') to elapsed seconds.
        Values are 0.0 for any model not yet run.
    """
    return dict(_generation_times)
