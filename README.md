# sidewalk-tagger-api

This serves as an API reimplementation of the [sidewalk-tagger-ai](https://github.com/ProjectSidewalk/sidewalk-tagger-ai) project, which aims to predict tags for Project Sidewalk label crops.

## Requirements
- Docker
- NVIDIA GPU drivers **(only if you are using GPU Version)**
  - If you aren't sure whether you have these, check if the `nvidia-smi` command works
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
docker run -d -p 5000:5000 sidewalk-tagger-api
```
GPU Version:
```bash
docker run --gpus all --runtime nvidia -d -p 5000:5000 sidewalk-tagger-api
```

## Try an example
```bash
curl -X POST -F "label_type=curbramp" -F "panorama_id=3-WpZU8MDYwe_9edeLw30w" -F "x=0.18981933593" -F "y=0.63134765625" http://127.0.0.1:5000/process
```
> ```{"label_type":"curbramp","tag_probabilities":{"missing-tactile-warning":0.9999897480010986,"narrow":1.4236595688998932e-06,"not-enough-landing-space":1.0658299061105936e-06,"not-level-with-street":2.5866484065772966e-08,"parallel-lines":1.123407307516129e-17,"points-into-traffic":0.011831503361463547,"pooled-water":3.2974125474538596e-07,"steep":1.8578341496322537e-06,"surface-problem":4.7534200575682917e-08,"tactile-warning":1.781248215593223e-07},"tags":["missing-tactile-warning"]} ```