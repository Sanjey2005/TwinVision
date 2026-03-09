"""
Central configuration for TwinVision pipeline.

All constants live here. No other module may hardcode values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS: list[str] = [
    "AI robots exploring the surface of Mars, cinematic lighting, 8K ultra-detailed",
    "Futuristic space station orbiting Earth at night, neon lights, photorealistic",
    "Humanoid robots building a city on the Moon, dramatic sunrise, epic scale",
    "Deep space nebula with an AI-piloted spacecraft, volumetric light, cinematic",
    "Alien megastructure discovered by AI probes, mysterious atmosphere, 8K render",
]

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUTPUT_BASE: Path = Path("output")

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "flux": {
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "inference_steps": 4,
        "guidance_scale": 0.0,
        "resolution": (1024, 1024),
    },
    "sd35": {
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "inference_steps": 28,
        "guidance_scale": 7.5,
        "resolution": (1024, 1024),
    },
}

# ---------------------------------------------------------------------------
# Evaluation metric weights (must sum to 1.0)
# ---------------------------------------------------------------------------

METRIC_WEIGHTS: dict[str, float] = {
    "clip_score": 0.30,
    "brisque": 0.20,
    "niqe": 0.20,
    "ssim": 0.15,
    "lpips": 0.15,
}

# ---------------------------------------------------------------------------
# Video encoding (FFmpeg)
# ---------------------------------------------------------------------------

VIDEO_FPS: int = 30
VIDEO_IMG_DURATION: float = 3.0          # seconds each image is displayed
VIDEO_CROSSFADE_DURATION: float = 0.5    # seconds of xfade overlap
VIDEO_RESOLUTION: str = "1024x1024"
VIDEO_CODEC: str = "libx264"
VIDEO_PIX_FMT: str = "yuv420p"

# ---------------------------------------------------------------------------
# Chart colours (dark-mode matching TwinVision frontend)
# ---------------------------------------------------------------------------

CHART_C_FLUX: str = "#ccff00"    # neon lime green
CHART_C_SD35: str = "#00f0ff"    # electric cyan
CHART_C_BG: str = "#0d0d0d"     # near-black background
CHART_C_PANEL: str = "#1a1a1a"   # panel surface
CHART_C_TEXT: str = "#f0f0f0"    # off-white text
CHART_C_GRID: str = "#333333"    # subtle grid lines
CHART_C_MUTED: str = "#888888"   # secondary text / hints

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------

API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
PIPELINE_TIMEOUT_SECONDS: int = 600   # 10 minutes per PRD error-handling spec
