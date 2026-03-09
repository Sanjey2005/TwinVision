# TwinVision — Product Requirements Document (PRD)
**Version:** 1.0  
**Project Name:** TwinVision: Prompt-to-Video Model Comparison  
**Tagline:** Two AI image models, one pipeline from prompt to video, side-by-side evaluation.  
**Target Developer:** Claude Code (Sonnet)  
**Date:** March 2026  

---

## 1. Overview

TwinVision is a full-stack AI college project. It accepts a text prompt from the user, generates images using two distinct AI models (FLUX.1 [schnell] and Stable Diffusion 3.5 Large), assembles those images into videos using FFmpeg, computes automated quality metrics, and presents a comparison report via a React web UI.

Two clear halves:
- **Backend (Python):** AI pipeline running on Google Colab T4 GPU — built by Claude Code
- **Frontend (React/TypeScript):** Visual web interface — built by Gemini via Antigravity

---

## 2. Goals

| Goal | Description |
|------|-------------|
| G1 | Accept a user text prompt via API and generate 7 images per model |
| G2 | Create MP4 videos from each model's image set using FFmpeg with transitions |
| G3 | Compute 5 evaluation metrics: CLIP Score, BRISQUE, NIQE, SSIM, LPIPS |
| G4 | Serve all results (images, videos, metrics) to the frontend via REST API |
| G5 | Provide a colab_notebook.ipynb for GPU-heavy execution |
| G6 | Connect cleanly to the React frontend via a well-defined API contract |

---

## 3. System Architecture

```
User (Browser)
    |-- types prompt, clicks Generate
    v
FastAPI Backend (backend/api.py)
    |
    |-- generate_images.py   (FLUX.1 + SD3.5 on Colab T4)
    |        v
    |-- create_videos.py     (FFmpeg MP4 assembly)
    |        v
    |-- evaluate.py          (CLIP, BRISQUE, NIQE, SSIM, LPIPS)
    |        v
    +-- compare.py           (Winner logic + Charts)
              v
         output/
         |-- flux/         PNG images
         |-- sd35/         PNG images
         |-- videos/       MP4 files
         +-- results/      metrics.csv + chart PNGs
              v
React Frontend (frontend/) reads via REST API
```

---

## 4. Full Project Structure

```
twinvision/
|-- CLAUDE.md
|-- PRD.md                       <- This file
|-- README.md
|-- requirements.txt
|-- .env                         <- HF_TOKEN (never commit)
|
|-- backend/
|   |-- api.py                   <- FastAPI server (main entry)
|   |-- config.py                <- All constants & model configs
|   |-- generate_images.py       <- FLUX.1 + SD3.5 image generation
|   |-- create_videos.py         <- FFmpeg video assembly
|   |-- evaluate.py              <- 5 evaluation metrics
|   |-- compare.py               <- Winner logic + chart generation
|   |-- main.py                  <- CLI orchestrator (for Colab)
|   +-- colab_notebook.ipynb     <- Runnable Colab notebook
|
|-- frontend/                    <- Built by Gemini
|   +-- src/
|       +-- api/
|           +-- twinvision.ts    <- Claude Code writes this file ONLY
|
+-- output/                      <- Auto-created at runtime
    |-- flux/
    |-- sd35/
    |-- videos/
    +-- results/
```

---

## 5. Backend Modules

### 5.1 config.py

Single source of truth for all constants. Must contain:

- PROMPTS: list of 5 text prompts about space/AI themes
  e.g. "AI robots exploring Mars surface, cinematic lighting, 8K"
- MODELS: dict with two entries "flux" and "sd35", each containing:
  - model_id: str (HuggingFace model path)
  - inference_steps: int
  - guidance_scale: float
  - resolution: tuple (1024, 1024)
  - output_dir: Path
- OUTPUT_BASE: Path("output/")
- METRIC_WEIGHTS: dict
  {"clip_score": 0.30, "brisque": 0.20, "niqe": 0.20, "ssim": 0.15, "lpips": 0.15}
- API_HOST: "0.0.0.0"
- API_PORT: 8000

---

### 5.2 generate_images.py

Functions:
```python
def load_flux_pipeline() -> FluxPipeline
def load_sd35_pipeline() -> StableDiffusion3Pipeline
def generate_images(prompt: str, model_name: str, num_images: int = 7) -> list[Path]
def run_all(prompts: list[str], test_mode: bool = False) -> dict[str, list[Path]]
```

Key rules:
- Use torch.float16 for both models
- FLUX.1-schnell: num_inference_steps=4, guidance_scale=0.0
- SD3.5-large: num_inference_steps=28, guidance_scale=7.5
- Save to output/flux/prompt_{i}_img_{j}.png and output/sd35/...
- Track generation_time per image in seconds
- On CUDA OOM: torch.cuda.empty_cache() then retry with enable_model_cpu_offload()
- test_mode=True generates 2 images instead of 7

---

### 5.3 create_videos.py

Functions:
```python
def create_video(image_paths: list[Path], output_path: Path) -> Path
def create_side_by_side_video(flux_video: Path, sd_video: Path, output_path: Path) -> Path
def run_all(flux_images: dict, sd_images: dict) -> dict[str, Path]
```

FFmpeg requirements:
- 3 seconds per image, 30fps output
- Ken Burns zoom effect via zoompan filter
- 0.5s crossfade transition between images
- H.264 codec, yuv420p pixel format
- Output files:
  output/videos/flux_prompt_{i}.mp4
  output/videos/sd35_prompt_{i}.mp4
  output/videos/comparison_prompt_{i}.mp4  (side by side)
- Use subprocess.run() with stderr capture for error reporting

---

### 5.4 evaluate.py

Functions:
```python
def compute_clip_score(images: list[Path], prompt: str) -> dict
def compute_brisque(images: list[Path]) -> dict
def compute_niqe(images: list[Path]) -> dict
def compute_ssim_consistency(images: list[Path]) -> dict
def compute_lpips_consistency(images: list[Path]) -> dict
def run_all_metrics(flux_images: dict, sd_images: dict, prompts: list[str]) -> pd.DataFrame
```

Libraries:
- CLIP Score: torchmetrics.multimodal.CLIPScore with "openai/clip-vit-large-patch14"
- BRISQUE + NIQE: pyiqa.create_metric('brisque') and pyiqa.create_metric('niqe')
- SSIM: torchmetrics.image.StructuralSimilarityIndexMeasure
- LPIPS: torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex')

Output CSV columns:
prompt_index | prompt_text | model | metric | per_image_scores | average_score | generation_time

Saved to: output/results/metrics.csv

---

### 5.5 compare.py

Functions:
```python
def compute_weighted_winner(metrics_df: pd.DataFrame) -> dict
def generate_bar_chart(metrics_df: pd.DataFrame) -> Path
def generate_radar_chart(summary: dict) -> Path
def generate_image_grid(flux_images: dict, sd_images: dict) -> Path
def build_comparison_report(metrics_df: pd.DataFrame) -> dict
```

build_comparison_report() must return:
```json
{
  "overall_winner": "FLUX.1",
  "overall_score": { "flux": 0.72, "sd35": 0.58 },
  "metric_winners": {
    "clip_score": { "winner": "FLUX", "flux": 0.83, "sd35": 0.71, "diff_pct": 16.9 },
    "brisque":    { "winner": "SD3.5", "flux": 34.2, "sd35": 28.1, "diff_pct": 17.8 },
    "niqe":       { "winner": "FLUX", "flux": 3.1, "sd35": 4.2, "diff_pct": 26.2 },
    "ssim":       { "winner": "FLUX", "flux": 0.78, "sd35": 0.61, "diff_pct": 21.8 },
    "lpips":      { "winner": "SD3.5", "flux": 0.42, "sd35": 0.31, "diff_pct": 26.2 }
  },
  "charts": {
    "bar_chart":   "output/results/bar_chart.png",
    "radar_chart": "output/results/radar_chart.png",
    "image_grid":  "output/results/image_grid.png"
  }
}
```

---

### 5.6 api.py — FastAPI Server

Dependencies: fastapi, uvicorn, python-multipart, python-dotenv

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Returns {"status": "ok"} |
| POST | /api/generate | Start a pipeline job |
| GET | /api/status/{job_id} | Poll job progress |
| GET | /api/results/{job_id} | Full results JSON |
| GET | /api/images/{job_id}/{model}/{filename} | Serve image file |
| GET | /api/videos/{job_id}/{filename} | Serve video file |

POST /api/generate
Request:  { "prompt": "AI robots exploring Mars, cinematic 8K" }
Response: { "job_id": "abc12345", "status": "queued" }  HTTP 202

GET /api/status/{job_id}
Response:
{
  "job_id": "abc12345",
  "status": "running",
  "stage": "image_generation",
  "progress": 42,
  "message": "Generating FLUX.1 image 3/7..."
}
status values: "queued" | "running" | "done" | "failed"

GET /api/results/{job_id}
Response:
{
  "job_id": "abc12345",
  "status": "done",
  "prompt": "AI robots exploring Mars...",
  "results": {
    "overall_winner": "FLUX.1",
    "overall_score": { "flux": 0.72, "sd35": 0.58 },
    "metric_winners": { ... },
    "images": {
      "flux": ["/api/images/abc12345/flux/prompt_0_img_0.png", ...],
      "sd35": ["/api/images/abc12345/sd35/prompt_0_img_0.png", ...]
    },
    "videos": {
      "flux":       "/api/videos/abc12345/flux_prompt_0.mp4",
      "sd35":       "/api/videos/abc12345/sd35_prompt_0.mp4",
      "comparison": "/api/videos/abc12345/comparison_prompt_0.mp4"
    },
    "generation_time": { "flux": 18.4, "sd35": 142.7 },
    "charts": {
      "bar_chart":   "/api/images/abc12345/results/bar_chart.png",
      "radar_chart": "/api/images/abc12345/results/radar_chart.png"
    }
  }
}

Implementation rules:
- Use threading.Thread to run pipeline in background per job
- Store job state in jobs: dict[str, dict] in memory
- job_id = uuid.uuid4().hex[:8]
- Enable CORS for http://localhost:5173
- Mount output/ as static files at /static
- Start: uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

---

### 5.7 frontend/src/api/twinvision.ts

Claude Code must write ONLY this one frontend file.
The rest of the frontend is handled by Gemini.

```typescript
const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type JobStatus = {
  job_id: string;
  status: "queued" | "running" | "done" | "failed";
  stage: string;
  progress: number;
  message: string;
};

export type MetricResult = {
  winner: string;
  flux: number;
  sd35: number;
  diff_pct: number;
};

export type TwinVisionResults = {
  job_id: string;
  status: string;
  prompt: string;
  results: {
    overall_winner: string;
    overall_score: Record<string, number>;
    metric_winners: Record<string, MetricResult>;
    images: { flux: string[]; sd35: string[] };
    videos: { flux: string; sd35: string; comparison: string };
    generation_time: Record<string, number>;
    charts: Record<string, string>;
  };
};

export async function startGeneration(prompt: string): Promise<{ job_id: string }>
export async function pollStatus(jobId: string): Promise<JobStatus>
export async function getResults(jobId: string): Promise<TwinVisionResults>
```

---

### 5.8 main.py — CLI Orchestrator

CLI flags:
  python backend/main.py                       Full pipeline, all 5 prompts
  python backend/main.py --test                2 images/model, quick verify
  python backend/main.py --skip-generation     Use existing images
  python backend/main.py --skip-video          Skip FFmpeg step
  python backend/main.py --prompt "text"       Single custom prompt

Prints timestamped stage logs.
Final output: total time taken, overall winner, path to results folder.

---

### 5.9 colab_notebook.ipynb

Generate via nbformat. Cells in order:
1. Markdown: TwinVision title + description
2. Code: !nvidia-smi GPU check
3. Code: !pip install -r requirements.txt
4. Code: upload/clone project files
5. Code: set HF_TOKEN env variable (with instructions)
6. Code: !python backend/main.py --test (verify setup)
7. Code: !python backend/main.py (full run)
8. Code: display flux + sd35 images with IPython.display.Image
9. Code: display videos with IPython.display.Video
10. Code: show metrics table with pd.read_csv + display
11. Code: show charts with IPython.display.Image
12. Code: download results with google.colab.files.download

---

## 6. Environment Variables

.env file (never commit to git):
  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

frontend/.env:
  VITE_API_URL=http://localhost:8000

---

## 7. Error Handling

| Scenario | Required Behavior |
|----------|------------------|
| CUDA OOM | torch.cuda.empty_cache() then retry with enable_model_cpu_offload() |
| FFmpeg missing | Raise clear error: FFmpeg not found. Install: apt-get install ffmpeg |
| HF_TOKEN not set | Raise EnvironmentError with setup instructions |
| API job not found | Return HTTP 404: {"error": "Job not found"} |
| Generation timeout over 10 min | Mark job as "failed" with message |
| Missing images for video | Log warning, skip that image, continue |

---

## 8. Non-Functional Requirements

- Must run on Google Colab T4 GPU (16GB VRAM) without modification
- FLUX.1 target: under 5 sec/image on T4
- SD3.5 target: under 60 sec/image on T4
- --test mode completes full pipeline under 5 minutes on T4
- All functions must have type hints and docstrings
- No hardcoded values — all configs in config.py
- CORS must allow localhost:5173

---

## 9. Implementation Order for Claude Code

| # | File | Note |
|---|------|------|
| 1 | backend/config.py | First — all modules import this |
| 2 | backend/generate_images.py | Core pipeline (use think hard) |
| 3 | backend/create_videos.py | FFmpeg integration |
| 4 | backend/evaluate.py | Metrics (use think hard) |
| 5 | backend/compare.py | Winner logic + charts |
| 6 | backend/api.py | FastAPI server |
| 7 | frontend/src/api/twinvision.ts | TypeScript client |
| 8 | backend/main.py | CLI orchestrator |
| 9 | backend/colab_notebook.ipynb | Colab notebook |
| 10 | README.md | Documentation |

---

## 10. Quick Verify Checklist

- [ ] python backend/main.py --test completes without errors
- [ ] Images exist in output/flux/ and output/sd35/
- [ ] MP4 files exist in output/videos/
- [ ] output/results/metrics.csv has correct columns
- [ ] Chart PNGs exist in output/results/
- [ ] uvicorn backend.api:app --port 8000 starts cleanly
- [ ] GET localhost:8000/api/health returns {"status": "ok"}
- [ ] POST localhost:8000/api/generate with a prompt returns job_id
- [ ] GET localhost:8000/api/results/{job_id} returns full results JSON after pipeline completes
