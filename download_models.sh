#!/bin/bash

# URLs are pinned to repo commits so a new upload to HuggingFace never changes a build by surprise.
pushd ./checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/e96fd0c8cedf1825a55b9c61309f754982948df0/depth_anything_v2_metric_vkitti_vitb.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/65959dbc80b87e4f39385204c4b639cbcf58e1a8/validated-dino-cls-b-crosswalk-tags-best.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/65959dbc80b87e4f39385204c4b639cbcf58e1a8/validated-dino-cls-b-curbramp-tags-best.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/65959dbc80b87e4f39385204c4b639cbcf58e1a8/validated-dino-cls-b-obstacle-tags-best.pth
wget https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/65959dbc80b87e4f39385204c4b639cbcf58e1a8/validated-dino-cls-b-surfaceproblem-tags-best.pth
popd