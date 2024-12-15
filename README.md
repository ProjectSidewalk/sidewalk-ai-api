# sidewalk-tagger-api

This serves as an API reimplementation of the [sidewalk-tagger-ai](https://github.com/ProjectSidewalk/sidewalk-tagger-ai) project, which aims to predict tags for Project Sidewalk label crops.

## Requirements
- Docker
- NVIDIA Container Toolkit for Docker **(only if you are using GPU Version)**
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  - Don't forget [this step!!](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)
- A good amount of disk space - the docker image is quite large
## Build
```bash
docker build -t sidewalk-tagger-api .
```
## Run

CPU Version:
```bash
docker run -p 5000:5000 sidewalk-tagger-api
```
GPU Version:
```bash
docker run --gpus all --runtime nvidia -d -p 5000:5000 sidewalk-tagger-api
```

## Try an example image
```bash
curl -X POST -F "label_type=curbramp" -F "image=@test_image_2.png" http://127.0.0.1:5000/classify
```
Output:
```
{"label_type":"curbramp","probabilities":{"missing-tactile-warning":0.9993209838867188,"narrow":5.09270466864109e-06,"not-enough-landing-space":1.2457426237233449e-06,"not-level-with-street":2.72767948672481e-07,"parallel-lines":3.4737886575527825e-17,"points-into-traffic":0.003298011841252446,"pooled-water":1.511450591351604e-07,"steep":1.2110045588542562e-07,"surface-problem":1.1465004234878506e-07,"tactile-warning":1.4279058291322144e-07},"result":["missing-tactile-warning"]}
```