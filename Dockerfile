FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY . /app

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install wget gcc mono-mcs libgmp3-dev libmpc-dev ffmpeg libsm6 libxext6 -y

RUN pip install --no-deps --no-cache-dir -r requirements.txt

RUN chmod +x ./download_models.sh && ./download_models.sh
RUN chmod +x ./docker_run.sh

EXPOSE 5000

CMD ["bash", "docker_run.sh"]
