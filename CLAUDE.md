# CLAUDE.md — TwinVision

## Project Overview
**TwinVision: Prompt-to-Video Model Comparison**
Two AI image models, one pipeline from prompt to video, side-by-side evaluation.

College AI & Cloud Computing project. A user provides a text prompt. Two different AI models generate images from that prompt. Those images are assembled into videos using FFmpeg. Automated evaluation metrics compare both outputs and determine the winner.

## Reference Documents
- `twinvision_prd.md` — Detailed PRD with API contracts, schemas, and function signatures. Always check this before implementing any module.

## Tech Stack
- **Backend Framework:** FastAPI + Uvicorn
- **Image Generation:** diffusers, transformers, accelerate, torch (fp16)
- **Video Creation:** FFmpeg via subprocess (not ffmpeg-python wrapper)
- **Evaluation Metrics:** pyiqa (BRISQUE, NIQE), torchmetrics (CLIP Score, SSIM, LPIPS)
- **Data Models:** Pydantic v2
- **Charts:** matplotlib (dark mode)
- **Runtime:** Google Colab T4 GPU (16GB VRAM)
- **Frontend:** Built separately — Claude Code only builds the backend + API

## Project Structure
```
twinvision/
├── CLAUDE.md
├── twinvision_prd.md
├── requirements.txt
├── .env.example
├── api/
│   ├── server.py                 # FastAPI app entry point
│   ├── routes/
│   │   ├── generate.py           # POST /generate
│   │   ├── status.py             # GET /status/{job_id}
│   │   └── results.py            # GET /results/{job_id}
│   ├── models/
│   │   ├── request_models.py     # Pydantic schemas
│   │   └── job_store.py          # In-memory job state tracker
│   └── middleware/
│       └── cors.py               # CORS for frontend on port 5173
├── pipeline/
│   ├── config.py                 # ALL constants — no hardcoding elsewhere
│   ├── generate_images.py        # FLUX.1 + SD3.5 image generation
│   ├── create_videos.py          # FFmpeg video assembly
│   ├── evaluate.py               # 5 metrics: CLIP, BRISQUE, NIQE, SSIM, LPIPS
│   ├── compare.py                # Scoring, charts, winner determination
│   └── orchestrator.py           # Full pipeline runner, updates job store
├── output/                       # Auto-created per job_id at runtime
│   └── {job_id}/
│       ├── flux/                 # FLUX.1 generated images
│       ├── sd35/                 # SD 3.5 generated images
│       ├── videos/               # flux.mp4, sd35.mp4, comparison.mp4
│       └── results/              # metrics.csv, bar_chart.png, radar_chart.png
├── run_pipeline_cli.py           # CLI runner (no server needed)
├── colab_notebook.ipynb          # Ready-to-run Colab notebook
└── README.md
```

## AI Models Configuration
| Model | ID | Steps | Guidance | Resolution |
|-------|----|-------|----------|------------|
| FLUX.1 [schnell] | black-forest-labs/FLUX.1-schnell | 4 | 0.0 | 1024×1024 |
| Stable Diffusion 3.5 Large | stabilityai/stable-diffusion-3.5-large | 28 | 7.5 | 1024×1024 |

## API Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | /generate | Start pipeline job, returns job_id |
| GET | /status/{job_id} | Poll progress (0-100%) |
| GET | /results/{job_id} | Get full ComparisonPayload JSON |
| GET | /output/{job_id}/{path} | Serve static images/videos |
| GET | /health | Health check with device info |

## Code Style Rules (MANDATORY)
- Type hints on EVERY function parameter and return type
- Docstrings on EVERY function (one-line summary + Args + Returns)
- pathlib.Path for ALL file paths — never os.path.join
- tqdm progress bars in ALL loops that run more than 3 iterations
- logging module for all output — no bare print() in pipeline/ or api/
- try/except with specific exception types — never bare except
- ALL constants come from pipeline/config.py — zero hardcoded values
- torch.float16 for all model loading
- HuggingFace token from environment variable HF_TOKEN — never hardcoded

## Critical Implementation Notes
- FLUX.1 schnell uses guidance_scale=0.0 (classifier-free guidance disabled)
- torch.cuda.OutOfMemoryError → clear cache + retry with enable_model_cpu_offload()
- SSIM/LPIPS need at least 2 images — return [1.0]/[0.0] if only 1 image
- metrics.csv per_image_scores column must be valid JSON strings, not Python repr
- ComparisonPayload image/video URLs must be relative (/output/{job_id}/...) not absolute filesystem paths
- job_store must use threading.Lock() — FastAPI BackgroundTasks are async
- Frontend runs on port 5173, backend on port 8000 — CORS must allow both

## Testing
- Quick test: `python run_pipeline_cli.py --prompt "test prompt" --test` (2 images per model)
- Full run: `python run_pipeline_cli.py --prompt "Futuristic AI robots building a city on Mars"`
- API test: `uvicorn api.server:app --port 8000` then `curl -X POST localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt":"test"}'`
- Check output/{job_id}/ for images, videos, metrics.csv, and chart PNGs

## Build Order
Implement modules in this exact order:
1. pipeline/config.py
2. pipeline/generate_images.py
3. pipeline/create_videos.py
4. pipeline/evaluate.py
5. pipeline/compare.py
6. pipeline/orchestrator.py
7. api/models/ (job_store + request_models)
8. api/routes/ + api/server.py
9. run_pipeline_cli.py + colab_notebook.ipynb
10. Final review
