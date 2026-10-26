"""
Live-hub check: each pinned validator's confidence_precision_map.json, downloaded fresh from HuggingFace, must have the
sha256 that MODEL_MAP, accuracy_mappings/<type>.json and tests/hub_accuracy_maps.json all record for it.

The unit tests compare those three against each other only (their stand-in hub serves maps by the hashes in
tests/hub_accuracy_maps.json), so an edit that changes all three consistently passes them; the image build and
startup would refuse it, and this catches it before either. Run it from the repo root after any change to MODEL_MAP,
the maps or the record, and before a deploy (five ~11 KB downloads into a temporary cache, no weights, no token):

    python3 regression/check_hub_maps.py

Needs huggingface_hub (requirements.txt). Exits 0 if every validator agrees, 1 naming each one that does not.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import hf_hub_download  # noqa: E402
from sidewalk_ai_api.model_map import HUB_ACCURACY_MAP, MODEL_MAP, accuracy_map_sha256, pinned_revision  # noqa: E402

HUB_RECORD = "tests/hub_accuracy_maps.json"


def main():
    with open(HUB_RECORD) as f:
        record = json.load(f)["maps"]
    failures = []
    # A fresh cache, so the hashes are what the hub serves now rather than an earlier download.
    with tempfile.TemporaryDirectory() as cache:
        for label_type, entry in MODEL_MAP["validator_models"].items():
            repo, revision = f"projectsidewalk/{entry['model_id']}", pinned_revision(entry, f"validator {label_type}")
            try:
                hub = accuracy_map_sha256(hf_hub_download(repo, HUB_ACCURACY_MAP, revision=revision, cache_dir=cache))
            except Exception as e:
                failures.append(f"{label_type}: cannot get {HUB_ACCURACY_MAP} at {revision} ({type(e).__name__}: {e})")
                continue
            local = accuracy_map_sha256(f"accuracy_mappings/{label_type}.json")
            recorded = record.get(repo, {}).get(revision)
            if not hub == entry["accuracy_map_sha256"] == local == recorded:
                failures.append(f"{label_type} @ {revision}: hub {hub}, MODEL_MAP {entry['accuracy_map_sha256']}, "
                                f"accuracy_mappings {local}, {HUB_RECORD} {recorded}")
            else:
                print(f"ok  {label_type:15} {revision[:8]}  {hub[:12]}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
