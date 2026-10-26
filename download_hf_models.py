# Downloads the tagger and validator weights pinned in MODEL_MAP into the HuggingFace cache. The Docker build runs this
# so that containers start without downloading them; elsewhere they download on first start instead.
from huggingface_hub import hf_hub_download, snapshot_download
from sidewalk_ai_api.model_map import MODEL_MAP, pinned_revision

# Every revision is checked first, so an unpinned entry fails the build before any download starts.
for kind in ("tagger_models", "validator_models"):
    for label_type, entry in MODEL_MAP[kind].items():
        pinned_revision(entry, f"{kind} {label_type}")

for entry in MODEL_MAP["tagger_models"].values():
    hf_hub_download(f"projectsidewalk/{entry['repo']}", f"{entry['model_id']}.pth", revision=entry["revision"])

for entry in MODEL_MAP["validator_models"].values():
    # Only the files transformers loads. The repos also hold ~2.4 GB of optimizer state from training.
    snapshot_download(f"projectsidewalk/{entry['model_id']}", revision=entry["revision"],
                      allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
