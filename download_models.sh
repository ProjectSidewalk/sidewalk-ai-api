#!/bin/bash

pushd ./checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-crosswalk-tags-best.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-curbramp-tags-best.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-obstacle-tags-best.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-surfaceproblem-tags-best.pth
popd