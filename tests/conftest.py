"""Lets the tests import main.py on a machine with no GPU, no checkpoints, no transformers and no Depth-Anything-V2
submodule: the models are only loaded by main.create_app() with no argument, and the tests pass stand-ins instead.
huggingface_hub is a stand-in too (FakeHub), serving the hub's accuracy maps as recorded in tests/hub_accuracy_maps.json.

Run from the repo root, in a venv with only the CPU packages (no model downloads, no network):

    python3 -m venv .venv
    .venv/bin/pip install numpy torch opencv-python-headless pillow flask requests pytest \\
        "sidewalk-panorama-tools @ git+https://github.com/ProjectSidewalk/sidewalk-panorama-tools@<pin in requirements.txt>"
    .venv/bin/pytest -q
"""

import hashlib
import json
import os
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["SIDEWALK_AI_API_KEY"] = "test-key"
os.environ.pop("SIDEWALK_SCRAPES_DIR", None)

# main.py reads API_VERSION and accuracy_mappings/ by relative path, as it does in the container (WORKDIR /app).
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Nothing in the tests may reach HuggingFace: huggingface_hub is replaced by the stand-in below, and offline mode is set
# in case anything imports the real one.
os.environ["HF_HUB_OFFLINE"] = "1"

HUB_RECORD = os.path.join(REPO_ROOT, "tests", "hub_accuracy_maps.json")


def _hub_maps():
    """{(repo, revision): bytes}: for each revision in tests/hub_accuracy_maps.json, the committed map whose sha256 is
    the one the hub holds there. A recorded hash that no committed map has is left out, so that revision has no map."""
    pool = {}
    for name in os.listdir(os.path.join(REPO_ROOT, "accuracy_mappings")):
        with open(os.path.join(REPO_ROOT, "accuracy_mappings", name), "rb") as f:
            data = f.read()
        pool[hashlib.sha256(data).hexdigest()] = data
    with open(HUB_RECORD) as f:
        record = json.load(f)["maps"]
    return {(repo, rev): pool[sha] for repo, revs in record.items() for rev, sha in revs.items() if sha in pool}


class FakeHub:
    """Stands in for huggingface_hub: hf_hub_download serves ``maps[(repo, revision)]`` (the hub as recorded, which a
    test may edit) from a cache directory under tmp_path, and raises, as offline mode does, for anything else."""
    def __init__(self, root):
        self.root = root
        self.maps = _hub_maps()
        self.calls = []

    def hf_hub_download(self, repo_id, filename, revision=None, **kwargs):
        self.calls.append((repo_id, filename, revision))
        if filename != "confidence_precision_map.json" or (repo_id, revision) not in self.maps:
            raise LookupError(f"{repo_id}/{filename}@{revision} is not in the (stand-in) HuggingFace cache")
        path = self.root / repo_id / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.maps[(repo_id, revision)])
        return str(path)


@pytest.fixture(autouse=True)
def fake_hub(tmp_path_factory, monkeypatch):
    hub = FakeHub(tmp_path_factory.mktemp("hf_cache"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hub.hf_hub_download))
    return hub
