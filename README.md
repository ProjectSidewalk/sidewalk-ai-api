
# sidewalk-ai-api

  

This serves as an API implementation of the Project Sidewalk AI models for [tagging](https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models) and [validating](https://huggingface.co/collections/projectsidewalk/project-sidewalk-validator-models-6797bd158d5e385e464dbe45) labels.

**This README contains the basic steps for setting up the API, but for a detailed usage guide on using the API, please see [API_USAGE.md](https://github.com/ProjectSidewalk/sidewalk-ai-api/blob/main/docs/API_USAGE.md)**  

## Requirements

- Docker

- NVIDIA GPU drivers

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

To redeploy prod, pull and run `./docker_build.sh && ./docker_start.sh`; the start script replaces the running container (expect a minute or two of downtime while the models load). Each build is also tagged with the API version, date, and commit it came from, e.g. `sidewalk-ai-api:1.0.1-2026-10-14-fbb1947`, so older images stay identifiable. To roll back, point the plain name at an older build and restart:

```bash

docker images sidewalk-ai-api                              # list builds, newest first
docker tag sidewalk-ai-api:<older-tag> sidewalk-ai-api
./docker_start.sh

```

Old builds aren't deleted automatically, and each one takes several GB. Remove ones you no longer need with `docker rmi sidewalk-ai-api:<tag>`.

  

## Try an example

```bash

curl -X POST -H "Authorization: Bearer $SIDEWALK_AI_API_KEY" -F "label_type=curbramp" -F "panorama_id=rZufe3aXwB2Syor0lKAqqw" -F "x=0.77746582031" -F "y=0.58459472656" http://127.0.0.1:5000/process

```
Please note that x and y are normalized coordinates (between 0 and 1) on the equirectangular image.
>  ```{"api_version":"1.0.0","image_source":"cache","label_type":"curbramp","tag_scores":{"debris / pooled water":1.6009700232189061e-07,"missing tactile warning":0.9999617338180542,"narrow":9.827788744587451e-06,"not enough landing space":1.6410160696977982e-06,"not level with street":5.565861016521012e-08,"points into traffic":1.844670691752981e-06,"steep":4.775402430823306e-06,"surface problem":4.025090305503909e-08},"tagger_model_id":"validated-dino-cls-b-curbramp-tags-best","tagger_training_date":"12-03-2024","tags":["missing tactile warning"],"tags_not_present":["points into traffic","surface problem"],"validation_estimated_accuracy":0.9598393574297188,"validation_result":"correct","validation_score":0.999808132648468,"validator_model_id":"sidewalk-validator-ai-curbramp","validator_training_date":"01-26-2025"} ```

Please note that `tag_scores` and `tags` will not be returned if the `label_type` does not match the following:
`["crosswalk", "curbramp", "obstacle", "surfaceproblem"]`

Likewise, `validation_result`, `validation_score`, and `validation_estimated_accuracy` will not be returned if the `label_type` does not match the following:
`["crosswalk", "curbramp", "obstacle", "surfaceproblem", "nocurbramp"]`

This is because we do not have models for other label types yet.

**For a more detailed guide on using the API, please see [API_USAGE.md](https://github.com/ProjectSidewalk/sidewalk-ai-api/blob/main/docs/API_USAGE.md)** 
