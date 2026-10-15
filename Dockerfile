FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install wget gcc ffmpeg libsm6 libxext6 -y && rm -rf /var/lib/apt/lists/*

# Install dependencies and models before copying the code, so code changes don't redo these large steps.
# --break-system-packages: the base image's torch lives in the system Python, so install next to it.
COPY requirements.txt download_models.sh ./
RUN pip install --no-deps --no-cache-dir --break-system-packages -r requirements.txt
RUN mkdir -p checkpoints && bash download_models.sh
# Bake in the tagger and validator weights pinned in MODEL_MAP, so containers start without downloading them.
COPY download_hf_models.py ./
COPY sidewalk_ai_api/model_map.py sidewalk_ai_api/
RUN python3 download_hf_models.py

COPY . /app

# Keep freed memory in the process for reuse. Each request makes several ~100 MB arrays, and by default glibc returns
# each one to the kernel when freed, so the next is fresh memory that has to be faulted in page by page. When the
# server is short of free memory, that took ~300 ms per array instead of ~15 ms.
ENV MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=4294967296

# Load the baked-in models without contacting HuggingFace, so a missing file fails at startup instead of downloading.
ENV HF_HUB_OFFLINE=1

EXPOSE 5000

CMD ["bash", "ignition.sh"]
