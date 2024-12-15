#!/bin/bash

wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth

pushd ./models
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-crosswalk-tags-best.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-curbramp-tags-best.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-obstacle-tags-best.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-surfaceproblem-tags-best.pth
popd