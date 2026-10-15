FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install wget gcc ffmpeg libsm6 libxext6 -y && rm -rf /var/lib/apt/lists/*

# Dependencies and model checkpoints come before the code, so a code change doesn't redo these multi-GB layers.
# The base image installs torch into the system Python (Ubuntu marks it externally managed), so install alongside it.
COPY requirements.txt download_models.sh ./
RUN pip install --no-deps --no-cache-dir --break-system-packages -r requirements.txt
RUN mkdir -p checkpoints && bash download_models.sh

COPY . /app

EXPOSE 5000

CMD ["bash", "ignition.sh"]
