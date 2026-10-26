"""Which model weights the API serves: one HuggingFace repo and commit per model.

The validator entries are for the October 2026 retrain (sidewalk-validator-ai round 3: panorama-grouped split,
tilt-corrected label-centred crops, ViT-B depth sizing, q95 WebP crops), each pinned to its 2026-10-26 HuggingFace
upload (publish_to_hf.py in sidewalk-validator-ai). ``pinned_revision()`` refuses an entry whose ``revision`` is
PLACEHOLDER_REVISION or not a 40-hex commit: a build with an unpinned model fails at image build
(download_hf_models.py) or at startup instead of silently serving a repo's HEAD.

Their accuracy maps ship with the ``incorrect`` verdict held back (capped at HOLD_BACK_CAP, so a Disagree never
reaches the website's 0.92 gate): the deployment prior the maps were fitted under is under review
(sidewalk-validator-ai#56) and the revision has no production outcomes yet.
"""
import hashlib
import re

# Marker for a model whose HuggingFace upload does not exist yet; pinned_revision() refuses it. A new validator gets
# this until it is published, then the upload's commit (and the same commit in accuracy_mappings/provenance.json).
PLACEHOLDER_REVISION = "TODO(jon): pin revision after publish"

# Every validator entry also carries ``accuracy_map_sha256``: the SHA-256 of ``accuracy_mappings/<label_type>.json``
# it is calibrated for. A map is only meaningful for the weights it was fitted on, so the two are bound here and
# checked at startup (main.check_accuracy_maps) and by the tests. A new model version or a refitted map changes both.


# The value a held-back verdict's map is capped at (sidewalk-validator-ai calibration/_isotonic_map.HOLD_BACK_CAP):
# just under the website's 0.92 gate, so the verdict is served with its accuracy but never becomes a vote.
HOLD_BACK_CAP = 0.9199


def accuracy_map_sha256(path):
    """SHA-256 hex digest of a map file, as recorded in MODEL_MAP's ``accuracy_map_sha256``."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pinned_revision(entry, what="model"):
    """The 40-hex HuggingFace commit of a MODEL_MAP entry; RuntimeError naming the entry if it is a placeholder.

    Example: pinned_revision(MODEL_MAP["validator_models"]["curbramp"], "validator curbramp") -> "969ad5a3..."
    """
    revision = str(entry.get("revision", ""))
    if revision == PLACEHOLDER_REVISION or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"{what} ({entry.get('model_id')}): revision is not pinned ({revision!r}); "
                           "set the HuggingFace commit in sidewalk_ai_api/model_map.py")
    return revision


MODEL_MAP = {
    "tagger_models": {
        "crosswalk": {
            "model_id": "validated-dino-cls-b-crosswalk-tags-best",
            "training_date": "12-03-2024",
            "repo": "sidewalk-tagger-ai-crosswalk",
            "revision": "7788eb8d836b28df5417fc77970bc7382a9105cf"
        },
        "curbramp": {
            "model_id": "validated-dino-cls-b-curbramp-tags-best",
            "training_date": "12-03-2024",
            "repo": "sidewalk-tagger-ai-curbramp",
            "revision": "cbd3a9632fc594a6f62de850757ae199761d0c4a"
        },
        "obstacle": {
            "model_id": "validated-dino-cls-b-obstacle-tags-best",
            "training_date": "12-03-2024",
            "repo": "sidewalk-tagger-ai-obstacle",
            "revision": "27fb9470678725e976fadbdd30ce41ddbbcddf10"
        },
        "surfaceproblem": {
            "model_id": "validated-dino-cls-b-surfaceproblem-tags-best",
            "training_date": "12-03-2024",
            "repo": "sidewalk-tagger-ai-surfaceproblem",
            "revision": "9b8f1053a64577c9041a68b0fe3f58fccb930543"
        }
    },
    "validator_models": {
        "crosswalk": {
            "model_id": "sidewalk-validator-ai-crosswalk",
            "training_date": "10-21-2026",
            "revision": "b8498b82ebeb6a4b69bfa1c2a1193a868fc97810",
            "accuracy_map_sha256": "8375e72eaffca87a2b7479c42b29e08df45fec020fa6688e1424447197e5febb"
        },
        "curbramp": {
            "model_id": "sidewalk-validator-ai-curbramp",
            "training_date": "10-21-2026",
            "revision": "6791a64185caa5e99c8b65a9bd5f05a73961d919",
            "accuracy_map_sha256": "e097f568a71aae6be073629843031028c6d39b9342c396b4ab81827af0ca2ece"
        },
        "obstacle": {
            "model_id": "sidewalk-validator-ai-obstacle",
            "training_date": "10-21-2026",
            "revision": "64b082c06cb7bef1943e35d707381e777793ad98",
            "accuracy_map_sha256": "b76b567a9d83b8ba03478403e53c9f64556fa86f455d703288756a2fa9e1644f"
        },
        "surfaceproblem": {
            "model_id": "sidewalk-validator-ai-surfaceproblem",
            "training_date": "10-21-2026",
            "revision": "b5497b7111e5216b97697fda410d3f62d933a748",
            "accuracy_map_sha256": "fa4eeec29dbcd164b1d08b43dcb877ef9b86ae19822f9b7bcf8971412742c49e"
        },
        "nocurbramp": {
            "model_id": "sidewalk-validator-ai-nocurbramp",
            "training_date": "10-21-2026",
            "revision": "ca7167d9ae145e10db9eaef64b425f2abad40d8a",
            "accuracy_map_sha256": "8dc1f8309df13041bb6e027b720d41a2e0baa9491fb4e123c5e45c584dafbf26"
        }
    }
}
