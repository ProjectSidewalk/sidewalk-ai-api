FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

WORKDIR /app

COPY . /app

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install wget gcc mono-mcs libgmp3-dev libmpc-dev ffmpeg libsm6 libxext6 -y

# The base image installs torch into the system Python (Ubuntu marks it externally managed), so install alongside it.
RUN pip install --no-deps --no-cache-dir --break-system-packages -r requirements.txt

RUN chmod +x ./download_models.sh && ./download_models.sh
RUN chmod +x ./ignition.sh

EXPOSE 5000

CMD ["bash", "ignition.sh"]
