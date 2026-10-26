# Downloads the tagger and validator weights pinned in MODEL_MAP into the HuggingFace cache. The Docker build runs this
# so that containers start without downloading them; elsewhere they download on first start instead. It also fails the
# build unless each validator's upload holds the accuracy map MODEL_MAP binds to it (sidewalk-ai-api#24).
import os
from huggingface_hub import hf_hub_download, snapshot_download
from sidewalk_ai_api.model_map import HUB_ACCURACY_MAP, MODEL_MAP, check_hub_accuracy_map, pinned_revision

# Every revision is checked first, so an unpinned entry fails the build before any download starts.
for kind in ("tagger_models", "validator_models"):
    for label_type, entry in MODEL_MAP[kind].items():
        pinned_revision(entry, f"{kind} {label_type}")

for entry in MODEL_MAP["tagger_models"].values():
    hf_hub_download(f"projectsidewalk/{entry['repo']}", f"{entry['model_id']}.pth", revision=entry["revision"])

for label_type, entry in MODEL_MAP["validator_models"].items():
    # Only the files transformers loads (older revisions of these repos also hold ~2.4 GB of optimizer state), plus the
    # accuracy map the weights were published with, which startup checks again from the cache (main.check_accuracy_maps).
    folder = snapshot_download(f"projectsidewalk/{entry['model_id']}", revision=entry["revision"],
                               allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors",
                                               HUB_ACCURACY_MAP])
    check_hub_accuracy_map(os.path.join(folder, HUB_ACCURACY_MAP), entry, f"validator_models {label_type}")
