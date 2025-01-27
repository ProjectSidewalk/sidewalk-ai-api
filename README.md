
# sidewalk-ai-api

  

This serves as an API implementation of the Project Sidewalk AI models for [tagging](https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models) and [validating](https://huggingface.co/collections/projectsidewalk/project-sidewalk-validator-models-6797bd158d5e385e464dbe45) labels.

  

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

## Build

```bash

docker  build  -t  sidewalk-ai-api  .

```

## Run
```bash

docker  run  --gpus  all  --runtime  nvidia  -d  -p  5000:5000  sidewalk-ai-api

```

  

## Try an example

```bash

curl -X POST -F "label_type=curbramp" -F "panorama_id=3-WpZU8MDYwe_9edeLw30w" -F "x=0.18981933593" -F "y=0.63134765625" http://127.0.0.1:5000/process

```
Please note that x and y are normalized coordinates (between 0 and 1) on the equirectangular image.
>  ```{"label_type":"curbramp","tag_scores":{"missing-tactile-warning":0.9999904632568359,"narrow":2.0490285805863095e-06,"not-enough-landing-space":9.689169928606134e-07,"not-level-with-street":2.722915048991581e-08,"parallel-lines":8.671155690381078e-18,"points-into-traffic":0.0051573594100773335,"pooled-water":2.6774074513014057e-07,"steep":1.7215750176546862e-06,"surface-problem":4.324794744547944e-08,"tactile-warning":1.704332674989928e-07},"tags":["missing-tactile-warning"],"validation_estimated_accuracy":0.941747572815534,"validation_result":"correct","validation_score":0.9998575448989868} ```

Please note that `tag_scores` and `tags` will not be returned if the `label_type` does not match the following:
`["crosswalk", "curbramp", "obstacle", "surfaceproblem"]`

Likewise, `validation_result`, `validation_score`, and `validation_estimated_accuracy` will not be returned if the `label_type` does not match the following:
`["crosswalk", "curbramp", "obstacle", "surfaceproblem", "nocurbramp"]`

This is because we do not have models for other label types yet.