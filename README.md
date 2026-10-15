
# sidewalk-ai-api

  

This serves as an API implementation of the Project Sidewalk AI models for [tagging](https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models) and [validating](https://huggingface.co/collections/projectsidewalk/project-sidewalk-validator-models-6797bd158d5e385e464dbe45) labels.

**This README contains the basic steps for setting up the API, but for a detailed usage guide on using the API, please see [API_USAGE.md](https://github.com/ProjectSidewalk/sidewalk-ai-api/blob/main/docs/API_USAGE.md)**  

## Requirements

- Docker

- NVIDIA GPU drivers, version 580 or newer (the image uses CUDA 13.0), on a Turing or newer GPU

	- If you aren't sure whether you have these, check if the `nvidia-smi` command works

- NVIDIA Container Toolkit for Docker

	- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

	- Don't forget [this step!!](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)

- A good amount of disk space - the docker image is quite large
- Lots of VRAM (9-10 GB MINIMUM)

### After cloning, don't forget to download the submodules
```bash
git submodule init
git submodule update
```

## Download checkpoints
```bash
./download_models.sh
```
This fetches the depth model. The tagger and validator models download from HuggingFace on first start, at the revisions pinned in `sidewalk_ai_api/model_map.py` (the Docker build bakes them into the image).

## Build

```bash

docker  build  -t  sidewalk-ai-api  .

```

## Run

Every request needs a password, set as `SIDEWALK_AI_API_KEY` on both this server and the Sidewalk website. Keep it in a file only you can read, which `docker_start.sh` loads on its own:

```bash

echo  'export SIDEWALK_AI_API_KEY=<your-key>'  >  ~/.sidewalk-ai-api.env
chmod  600  ~/.sidewalk-ai-api.env
./docker_start.sh

```

Or, to run the container by hand, export it first (rather than typing it into the `docker run` line, so it stays out of your shell history):

```bash

export  SIDEWALK_AI_API_KEY=<your-key>
docker  run  --gpus  all  --runtime  nvidia  -d  -p  5000:5000  -e  SIDEWALK_AI_API_KEY  sidewalk-ai-api

```

To test changes next to the production instance, build and start with `dev`. This runs a separate `sidewalk-ai-api-dev` container on `127.0.0.1:60655` (it needs its own ~9–10 GB of VRAM). To reach it from your own machine, tunnel in with `ssh -L 60655:127.0.0.1:60655 <server>` and send requests to `http://localhost:60655/process`. Remove it with `docker rm -f sidewalk-ai-api-dev` when you're done. To check that your changes didn't change the models' outputs, run `python3 regression/compare.py` on the server while both are running; it compares prod and dev on a fixed set of labels.

```bash

./docker_build.sh  dev
./docker_start.sh  dev

```

To redeploy prod, pull and run `./docker_build.sh && ./docker_start.sh` (about a minute or two of downtime while the models load). Each build is also tagged with its version, build time, and commit, e.g. `sidewalk-ai-api:1.0.1-2026-10-14-1803-5a9487c`. To roll back to an older build:

```bash

docker images sidewalk-ai-api                              # list builds, newest first
docker tag sidewalk-ai-api:<older-tag> sidewalk-ai-api
./docker_start.sh

```

Each build contains its own copy of every model, so rolling back also rolls back the models. Old builds take several GB each; delete ones you don't need with `docker rmi sidewalk-ai-api:<tag>`.

  

## Try an example

```bash

curl -X POST -H "Authorization: Bearer $SIDEWALK_AI_API_KEY" -F "label_type=curbramp" -F "panorama_id=rZufe3aXwB2Syor0lKAqqw" -F "x=0.77746582031" -F "y=0.58459472656" http://127.0.0.1:5000/process

```
Please note that x and y are normalized coordinates (between 0 and 1) on the equirectangular image.
>  ```{"api_version":"1.0.4","image_source":"cache","label_type":"curbramp","tag_scores":{"debris / pooled water":1.6009700232189061e-07,"missing tactile warning":0.9999617338180542,"narrow":9.827788744587451e-06,"not enough landing space":1.6410160696977982e-06,"not level with street":5.565861016521012e-08,"points into traffic":1.844670691752981e-06,"steep":4.775402430823306e-06,"surface problem":4.025090305503909e-08},"tagger_model_id":"validated-dino-cls-b-curbramp-tags-best","tagger_revision":"cbd3a9632fc594a6f62de850757ae199761d0c4a","tagger_training_date":"12-03-2024","tags":["missing tactile warning"],"tags_not_present":["points into traffic","surface problem"],"validation_estimated_accuracy":0.9598393574297188,"validation_result":"correct","validation_score":0.999808132648468,"validator_model_id":"sidewalk-validator-ai-curbramp","validator_revision":"969ad5a33079db7885c9e9a4eb5cd0f39d8fe57a","validator_training_date":"08-16-2025"} ```

Please note that `tag_scores` and `tags` will not be returned if the `label_type` does not match the following:
`["crosswalk", "curbramp", "obstacle", "surfaceproblem"]`

Likewise, `validation_result`, `validation_score`, and `validation_estimated_accuracy` will not be returned if the `label_type` does not match the following:
`["crosswalk", "curbramp", "obstacle", "surfaceproblem", "nocurbramp"]`

This is because we do not have models for other label types yet.

**For a more detailed guide on using the API, please see [API_USAGE.md](https://github.com/ProjectSidewalk/sidewalk-ai-api/blob/main/docs/API_USAGE.md)** 
