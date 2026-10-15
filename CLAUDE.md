# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Flask API that serves Project Sidewalk's AI models for tagging and validating accessibility labels (curb ramps, crosswalks, obstacles, surface problems, missing curb ramps) on Google Street View panoramas. Single endpoint `POST /process` in `main.py`. Request/response details for API consumers are in `docs/API_USAGE.md`.

## Common commands

After cloning, fetch the Depth-Anything-V2 submodule and download model checkpoints:

```bash
git submodule init && git submodule update
./download_models.sh   # fetches the depth checkpoint into checkpoints/ (the Docker build also runs this)
```

Build and run via Docker (GPU required — needs CUDA + nvidia-container-toolkit, ~9–10 GB VRAM per instance):

```bash
./docker_build.sh       # builds sidewalk-ai-api, also tagged <API_VERSION>-<date>-<time>-<commit> for rollback
./docker_start.sh       # runs it as container sidewalk-ai-api on 127.0.0.1:60654 → 5000, restart unless-stopped
./docker_build.sh dev && ./docker_start.sh dev   # test instance alongside prod: sidewalk-ai-api-dev on 127.0.0.1:60655
```

Starting either instance gracefully replaces its running container, so `./docker_start.sh` is also how you redeploy. `docker_start.sh` sources `SIDEWALK_AI_API_KEY` from `~/.sidewalk-ai-api.env` (refuses to start without it), and mounts the lab's panorama cache read-only, setting `SIDEWALK_SCRAPES_DIR` to the same path. Ports are bound to loopback only; on the server, an https reverse proxy fronts prod. To reach the dev instance from another machine, use an SSH tunnel (`ssh -L 60655:127.0.0.1:60655 <server>`). The container runs `ignition.sh` (gunicorn, 1 worker, 120 s timeout).

Run locally without Docker (requires CUDA, the submodule, and downloaded checkpoints):

```bash
export SIDEWALK_AI_API_KEY=<key>          # required; without it every request is a 401
export SIDEWALK_ALLOW_CPU=1                # only on a machine without a GPU
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 main:app   # production form (see ignition.sh)
python main.py                            # dev form, Flask's built-in server
```

Smoke test (use port 60654/60655 for the Docker instances):

```bash
curl -X POST -H "Authorization: Bearer $SIDEWALK_AI_API_KEY" -F "label_type=curbramp" -F "panorama_id=rZufe3aXwB2Syor0lKAqqw" \
     -F "x=0.77746582031" -F "y=0.58459472656" http://127.0.0.1:5000/process
```

There is no unit test suite, linter, or formatter configured. For changes that could affect model outputs (dependency bumps, preprocessing, refactors), run the regression check on the server with prod and a dev instance both up: `python3 regression/compare.py` sends the 50 labels in `regression/labels.csv` to both (all served from the panorama cache, so both see identical pixels) and reports changed outcomes (FAIL) or score movement beyond `--tolerance` (DRIFT), then forces one Google download per label type to check the download path still works on dev. It also fails if dev is more than `--max-slowdown` (1.5×) slower than prod on the cached labels, for the whole request or the model stages alone (from `Server-Timing`, API 1.0.1+). It passes (exit 0) only with no FAIL, no SKIP (both errored), no slowdown, and a clean download check. It only uses the standard library.

## Request flow (`/process`)

Form fields: `label_type`, `panorama_id`, `x`, `y` (required), and `city` (optional, used only for the cache lookup). The endpoint stitches together panorama loading, geometry, depth estimation, and two ML inferences:

1. **Panorama load** (`sidewalk_ai_api/panorama.py`) — `Panorama(pano_id, cached_image_path)` first tries the local cache at `$SIDEWALK_SCRAPES_DIR/{city}/{pano_id[:2]}/{pano_id}.jpg` (read with OpenCV). On a miss, it downloads tiles from Google Street View's `streetviewpixels-pa.googleapis.com` endpoint: auto-discovers dimensions by scanning until it hits black tiles, falls back from zoom 4 → zoom 3 if needed, fetches tiles in parallel (100 threads), assembles them, and crops zero-borders. Either way the result is resized to a fixed 8192×4096. The response's `image_source` is `"cache"` or `"download"`; if both fail the endpoint returns 502.
2. **Equirectangular → perspective projection** — the normalized `(x, y)` is converted to yaw/pitch (`Panorama.get_perspective_center_params`), then `to_perspective_image` renders a 90° FOV, 2048×2048 view centered on the label using PyTorch `grid_sample` on the GPU.
3. **Depth-aware crop sizing** (`sidewalk_ai_api/depthanything.py`) — `DepthAnythingPredictor` runs the Depth-Anything-V2 metric model (`vitb` / `vkitti`, max_depth=80) on the perspective image; the crop half-size around the label center is `int((1 / depth_at_center) * 6100)`, so closer features get larger crops. The crop is resized to a 640px longest side and converted to an RGB PIL image.
4. **Tagger inference** (`sidewalk_ai_api/tagger.py`) — only for `crosswalk`, `curbramp`, `obstacle`, `surfaceproblem`. A DINOv2 ViT-base classifier (custom head: 768→256→nc) loads `{model_id}.pth` from `projectsidewalk/{repo}` at the `revision` in `MODEL_MAP`: one repo per label type, `sidewalk-tagger-ai-{label_type}`. The older `sidewalk-tagger-ai-models` repo holds every model from the ASSETS '24 paper and stays unchanged for reproducibility. Tag names (by class index) come from `mappings.json`; `NULL` entries are intentional placeholders that preserve the class indices learned during training — do not remove them. Sigmoid scores are compared against per-tag thresholds in `tag_thresholds.json`: score ≥ `present` → `tags`, score ≤ `absent` → `tags_not_present`, either may be `null` to disable that side. `tag_scores` reports every tag listed in `tag_thresholds.json`, including ones with both thresholds null.
5. **Validator inference** (`sidewalk_ai_api/validator.py`) — for all five label types including `nocurbramp`. Loads `projectsidewalk/sidewalk-validator-ai-{label_type}` via `transformers.AutoModelForImageClassification`, pinned to the repo commit in `MODEL_MAP`'s `revision`. The Docker build bakes taggers and validators into the image (`download_hf_models.py`) and sets `HF_HUB_OFFLINE=1`, so containers never contact HuggingFace; outside Docker they download on first start. The image processor uses `backend="pil"`, which matches transformers 4.x preprocessing exactly; transformers 5's torchvision default resizes slightly differently and shifts scores. Returns `(label_name, confidence)`.
6. **Calibrated accuracy lookup** — `accuracy_mappings/{label_type}.json` maps confidence thresholds to empirical accuracies (model confidence ≠ probability of correctness). The code walks ascending thresholds and keeps the highest one ≤ confidence, returned as `validation_estimated_accuracy`.

Each stage's time is returned in a `Server-Timing` response header (`timed()` and an `after_request` hook in `main.py`).

At startup `main.py` exits if torch can't see a GPU, rather than silently running every model on the CPU; `SIDEWALK_ALLOW_CPU=1` overrides this.

All taggers, validators, and the depth model are instantiated **once at module import** in `main.py` and live on the GPU for the lifetime of the process. Cold start takes a minute or two; gunicorn is intentionally run with a single worker.

## Conventions and gotchas

- **Never run `git add -A`** (or `git add .`): stage files by name. The working tree often has untracked scratch files, like regression results and probe scripts, that must not be committed to this public repo.

- **Auth:** a `before_request` hook in `main.py` requires `Authorization: Bearer <SIDEWALK_AI_API_KEY>` on every request (constant-time compare, 401 otherwise, fails closed when the env var is unset). The Sidewalk website sends the same key from its own `SIDEWALK_AI_API_KEY` env var, so rotate both together. Never commit the key or put it on a `docker run` command line.
- `x` and `y` in API requests are **normalized** equirectangular coordinates in `[0, 1]`, not pixels. They get multiplied by the fixed 8192×4096 panorama size.
- The set of supported label types is hardcoded in two lists in `main.py`: `TAGGER_LABEL_TYPES` (4 types) and `VALIDATOR_LABEL_TYPES` (5 types, includes `nocurbramp`). Adding a new label type requires updating both lists, `mappings.json` and `tag_thresholds.json` (for taggers), `accuracy_mappings/`, and `MODEL_MAP` in `sidewalk_ai_api/model_map.py`.
- `MODEL_MAP` carries `model_id` and `training_date` strings (MM-DD-YYYY) returned in the API response — keep these in sync when updating checkpoints. `MODEL_MAP` is also what the API loads: each entry carries the HuggingFace commit `revision` (taggers also a `repo`), returned as `tagger_revision`/`validator_revision`; a new model version must update it, or the API keeps serving the old weights. `download_models.sh` pins the depth checkpoint URL to a repo commit.
- `API_VERSION` is read from the file of the same name at startup and echoed in every response. Bump it when behavior changes.
- **Dependencies:** `requirements.txt` is a full pinned freeze, installed with `pip install --no-deps --break-system-packages` into the system Python (3.12) of the `pytorch/pytorch:2.14.1-cuda13.0` base image. Because of `--no-deps`, bumping a package means checking its own requirements by hand. torch/torchvision/triton must exactly match the base image's versions; otherwise pip replaces them with builds lacking CUDA. Major bumps of torch or transformers can change model outputs, so compare outputs before and after. All checkpoints load under `torch.load`'s `weights_only=True` default; keep it that way for new checkpoints (save plain state dicts).
- The Dockerfile sets `MALLOC_MMAP_THRESHOLD_`/`MALLOC_TRIM_THRESHOLD_` so glibc reuses freed memory for the ~100 MB arrays each request makes. Without them every such array is fresh memory from the kernel, and when makelab2 is low on free memory each one cost ~25,000 page faults (~300 ms instead of ~15 ms).
- The `Depth-Anything-V2` directory is a git submodule and is added to `sys.path` at runtime in `depthanything.py` (`Depth-Anything-V2/metric_depth`). The `dinov2/` directory is a vendored copy of Meta's DINOv2 (used directly by `tagger.py` via `from dinov2.models.vision_transformer import vit_base`); do not treat it like normal first-party code.
- Tile fetches go to Google's undocumented Street View tile API, which requires browser-like headers (`HEADERS` in `panorama.py`). All fetches share one `requests.Session` (`SESSION`) to reuse connections, which roughly halved download time. Beyond one retry per tile, a timeout, and a zoom-3 fallback, there is no retry or rate-limit handling — be conservative if you change `_fetch_panorama` / `_fetch_tile`.
