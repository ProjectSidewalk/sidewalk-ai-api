FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install wget -y
RUN pip install --no-deps --no-cache-dir -r requirements.txt

RUN chmod +x ./download_models.sh && ./download_models.sh
RUN chmod +x ./docker_run.sh

EXPOSE 5000

CMD ["/bin/bash", "docker_run.sh"]
