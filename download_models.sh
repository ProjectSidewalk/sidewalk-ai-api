#!/bin/bash
set -euo pipefail

# Taggers and validators load from HuggingFace at the revisions in MODEL_MAP (see download_hf_models.py).
# The URL is pinned to a repo commit so a new upload to HuggingFace never changes a build by surprise.
pushd ./checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/e96fd0c8cedf1825a55b9c61309f754982948df0/depth_anything_v2_metric_vkitti_vitb.pth
popd