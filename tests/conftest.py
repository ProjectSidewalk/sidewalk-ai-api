"""Lets the tests import main.py on a machine with no GPU, no checkpoints, no transformers and no Depth-Anything-V2
submodule: the models are only loaded by main.create_app() with no argument, and the tests pass stand-ins instead.

Run from the repo root, in a venv with only the CPU packages (no model downloads, no network):

    python3 -m venv .venv
    .venv/bin/pip install numpy torch opencv-python-headless pillow flask requests pytest \\
        "sidewalk-panorama-tools @ git+https://github.com/ProjectSidewalk/sidewalk-panorama-tools@<pin in requirements.txt>"
    .venv/bin/pytest -q
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["SIDEWALK_AI_API_KEY"] = "test-key"
os.environ.pop("SIDEWALK_SCRAPES_DIR", None)

# main.py reads API_VERSION and accuracy_mappings/ by relative path, as it does in the container (WORKDIR /app).
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
