from contextlib import contextmanager
from flask import Flask, g, request, jsonify
from PIL import Image
from sidewalk_ai_api.panorama import Panorama
from sidewalk_ai_api.model_map import MODEL_MAP
from sidewalk_panorama_tools.geometry import label_view
import cv2
import hmac
import json
import logging
import math
import os
import sys
import time
import torch

SCRAPES_DIR = os.environ.get("SIDEWALK_SCRAPES_DIR")

# The password every request must include. If it's not set, every request is rejected rather than letting anyone in.
API_KEY = os.environ.get("SIDEWALK_AI_API_KEY", "").strip()
EXPECTED_AUTH_HEADER = f"Bearer {API_KEY}"
logger = logging.getLogger(__name__)
if not API_KEY:
    logger.warning("SIDEWALK_AI_API_KEY is not set; every request will be rejected with 401.")

# Without a GPU every model silently runs on the CPU, many times slower, so refuse to start.
# Exit code 3 makes gunicorn stop instead of restarting the worker forever.
if not torch.cuda.is_available() and os.environ.get("SIDEWALK_ALLOW_CPU") != "1":
    print("No GPU found. Check the GPU driver and that the container was started with --gpus all, "
          "or set SIDEWALK_ALLOW_CPU=1 to run on the CPU.", file=sys.stderr)
    sys.exit(3)

app = Flask(__name__)


@app.before_request
def require_api_key():
    """Rejects any request without the right password, before doing any real work."""
    provided = request.headers.get("Authorization", "")
    if not API_KEY or not hmac.compare_digest(provided.encode(), EXPECTED_AUTH_HEADER.encode()):
        logger.warning(f"Rejected request from {request.remote_addr}: missing or invalid API key")
        return jsonify({"error": "Missing or invalid API key"}), 401, {"WWW-Authenticate": "Bearer"}
    g.start = time.perf_counter()
    g.timings = {}


@contextmanager
def timed(stage):
    """Records how long a stage of the request took, for the Server-Timing header."""
    start = time.perf_counter()
    try:
        yield
    finally:
        g.timings[stage] = time.perf_counter() - start


@app.after_request
def add_server_timing(response):
    """Adds each stage's time to a Server-Timing header, e.g. `pano;desc=cache;dur=310, depth;dur=95, ..., total;dur=520`."""
    if "start" in g:
        stages = []
        for name, secs in g.timings.items():
            desc = f";desc={g.pano_source}" if name == "pano" and g.get("pano_source") else ""
            stages.append(f"{name}{desc};dur={secs * 1000:.0f}")
        stages.append(f"total;dur={(time.perf_counter() - g.start) * 1000:.0f}")
        response.headers["Server-Timing"] = ", ".join(stages)
    return response


TAGGER_LABEL_TYPES = ["crosswalk", "curbramp", "obstacle", "surfaceproblem"]
VALIDATOR_LABEL_TYPES = ["crosswalk", "curbramp", "obstacle", "surfaceproblem", "nocurbramp"]

# The models load once, here at import, and stay on the GPU. SIDEWALK_TEST_STUBS=1 skips loading them (and their
# imports, which need the Depth-Anything-V2 submodule, transformers and the checkpoints) so the tests in tests/ can
# import the app on a machine without any of those and put stand-ins in these three names. Never set it in production.
if os.environ.get("SIDEWALK_TEST_STUBS") == "1":
    depthanything, taggers, validators = None, {}, {}
else:
    from sidewalk_ai_api.tagger import ImageTagger
    from sidewalk_ai_api.validator import ImageValidator
    from sidewalk_ai_api.depthanything import DepthAnythingPredictor
    depthanything = DepthAnythingPredictor()
    taggers = {label_type: ImageTagger(label_type=label_type) for label_type in TAGGER_LABEL_TYPES}
    validators = {label_type: ImageValidator(label_type=label_type) for label_type in VALIDATOR_LABEL_TYPES}
accuracy_mappings = {label_type: json.load(open(f"accuracy_mappings/{label_type}.json"))
                 for label_type in VALIDATOR_LABEL_TYPES}

with open('API_VERSION', 'r') as file:
    API_VERSION = file.read().strip()

# Rig-tilt correction (docs/API_USAGE.md, "Camera pose and tilt correction"). The default beta is the fraction of the
# rig tilt measured to leak into a stored label point for the pose record the website sends (sidewalk-panorama-tools
# #191); it differs from the package's own default (0.9, for the database's camera_pitch column), so it is set here.
DEFAULT_TILT_BETA = 0.95
MAX_POSE_DEG = 90.0
MAX_TILT_BETA = 1.5
COORD_FRAMES = ("level", "rig")


class BadRequest(ValueError):
    """A request field that fails validation; its message is returned with a 400."""


def optional_float(name):
    """The form field `name` as a float, or None when it is absent, empty or NaN. Raises BadRequest for any other
    value that isn't a finite number."""
    raw = request.form.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise BadRequest(f"{name} must be a number")
    if math.isnan(value):
        return None
    if not math.isfinite(value):
        raise BadRequest(f"{name} must be finite")
    return value


def parse_tilt_fields():
    """Reads the optional pose fields. -> (camera_pitch, camera_roll, tilt_beta, coord_frame, tilt_applied).

    The tilt is applied iff coord_frame is "level" and camera_pitch is known (a missing roll counts as 0). "rig" means
    x and y already point at image pixels (e.g. the AI labeler's own labels), so they are never corrected.

    Example: camera_pitch=3, coord_frame absent -> (3.0, None, 0.95, "level", True).
    """
    camera_pitch = optional_float("camera_pitch")
    camera_roll = optional_float("camera_roll")
    for name, value in (("camera_pitch", camera_pitch), ("camera_roll", camera_roll)):
        if value is not None and abs(value) > MAX_POSE_DEG:
            raise BadRequest(f"{name} must be between -{MAX_POSE_DEG:g} and {MAX_POSE_DEG:g} degrees")

    raw_beta = request.form.get("tilt_beta", "").strip()
    try:
        tilt_beta = float(raw_beta) if raw_beta else DEFAULT_TILT_BETA
    except ValueError:
        raise BadRequest("tilt_beta must be a number")
    if not (math.isfinite(tilt_beta) and 0 <= tilt_beta <= MAX_TILT_BETA):
        raise BadRequest(f"tilt_beta must be between 0 and {MAX_TILT_BETA:g}")

    coord_frame = request.form.get("coord_frame", "").strip() or "level"
    if coord_frame not in COORD_FRAMES:
        raise BadRequest(f"coord_frame must be one of {list(COORD_FRAMES)}")

    tilt_applied = coord_frame == "level" and camera_pitch is not None
    return camera_pitch, camera_roll, tilt_beta, coord_frame, tilt_applied


@app.route("/process", methods=["POST"])
def process():
    if "label_type" not in request.form:
        return jsonify({"error": "Missing label_type parameter"}), 400
    if "panorama_id" not in request.form:
        return jsonify({"error": "Missing panorama_id parameter"}), 400
    if "x" not in request.form:
        return jsonify({"error": "Missing x parameter"}), 400
    if "y" not in request.form:
        return jsonify({"error": "Missing y parameter"}), 400

    try:
        label_x = float(request.form["x"])
        label_y = float(request.form["y"])
    except ValueError:
        return jsonify({"error": "x and y must be numbers"}), 400
    if not (math.isfinite(label_x) and math.isfinite(label_y)) or label_x < 0 or label_x > 1 or label_y < 0 or label_y > 1:
        return jsonify({"error": "x and y must be between 0 and 1. They should be normalized values."}), 400

    label_type = request.form["label_type"]
    if label_type not in VALIDATOR_LABEL_TYPES:
        return jsonify({"error": f"Invalid label_type. Choose from {VALIDATOR_LABEL_TYPES}"}), 400

    try:
        camera_pitch, camera_roll, tilt_beta, coord_frame, tilt_applied = parse_tilt_fields()
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400

    panorama_id = request.form["panorama_id"]
    city = request.form.get("city")

    cached_image_path = None
    if SCRAPES_DIR and city and len(panorama_id) >= 2:
        cached_image_path = os.path.join(SCRAPES_DIR, city, panorama_id[:2], f"{panorama_id}.jpg")

    with timed("pano"):
        panorama = Panorama(panorama_id, cached_image_path=cached_image_path)
    g.pano_source = panorama.image_source
    if panorama.panorama_image is None:
        return jsonify({"error": f"Failed to fetch panorama"}), 502
    with timed("project"):
        height, width = panorama.panorama_image.shape[:2]
        # The shared geometry (sidewalk-panorama-tools): a 90 degree view aimed at the label, 2048 px for the 8192 px
        # pano, with the rig-tilt rotation when tilt_applied (otherwise no rotation at all).
        if tilt_applied:
            view, tilt = label_view(label_x, label_y, 90, width // 4, camera_pitch, camera_roll, tilt_beta)
        else:
            view, tilt = label_view(label_x, label_y, 90, width // 4)
        perspective_image = panorama.to_perspective_image(view, tilt)

    with timed("depth"):
        depth = depthanything.predict_depth(perspective_image)

    center_x, center_y = perspective_image.shape[1] // 2, perspective_image.shape[0] // 2
    img_h, img_w = perspective_image.shape[:2]

    inv_depth = 1 / depth[center_y, center_x]
    crop_size_half = int(inv_depth * 6100)

    start_x = max(0, center_x - crop_size_half)
    start_y = max(0, center_y - crop_size_half)
    end_x = min(img_w, center_x + crop_size_half)
    end_y = min(img_h, center_y + crop_size_half)

    crop_height, crop_width = perspective_image[start_y:end_y, start_x:end_x].shape[:2]
    if crop_width > crop_height:
        resize_width, resize_height = 640, int(640 * crop_height / crop_width)
    else:
        resize_width, resize_height = int(640 * crop_width / crop_height), 640
    perspective_image = cv2.resize(perspective_image[start_y:end_y, start_x:end_x], (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)

    perspective_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB)
    perspective_image = Image.fromarray(perspective_image)

    response = {
        "label_type": label_type,
        "image_source": panorama.image_source,
        "tilt_applied": tilt_applied,
        "tilt_beta": tilt_beta if tilt_applied else None,
        "coord_frame": coord_frame,
    }

    # Perform tagging inference only if label_type is in TAGGER_LABEL_TYPES
    if label_type in TAGGER_LABEL_TYPES:
        classifier = taggers[label_type]
        try:
            with timed("tagger"):
                tags_present, tags_not_present, confidence_scores = classifier.inference(perspective_image)
            response.update({
                "tags": tags_present,
                "tags_not_present": tags_not_present,
                "tag_scores": confidence_scores,
                "tagger_model_id": MODEL_MAP["tagger_models"][label_type]["model_id"],
                "tagger_training_date": MODEL_MAP["tagger_models"][label_type]["training_date"],
                "tagger_revision": MODEL_MAP["tagger_models"][label_type]["revision"],
            })
        except Exception as e:
            return jsonify({"error": f"Inference error: {str(e)}"}), 500

    # Perform validation if label_type is in VALIDATOR_LABEL_TYPES
    validator = validators[label_type]
    try:
        with timed("validator"):
            validation_result, validation_confidence = validator.validate(perspective_image)

        # Find the highest mapping that's less than or equal to the confidence
        accuracy = 0
        for threshold, acc in accuracy_mappings[label_type][validation_result].items():
            if float(threshold) <= validation_confidence:
                accuracy = acc

        response.update({
            "validation_result": validation_result,
            "validation_score": validation_confidence,
            "validation_estimated_accuracy": accuracy,
            "validator_model_id": MODEL_MAP["validator_models"][label_type]["model_id"],
            "validator_training_date": MODEL_MAP["validator_models"][label_type]["training_date"],
            "validator_revision": MODEL_MAP["validator_models"][label_type]["revision"],
        })
    except Exception as e:
        return jsonify({"error": f"validation error: {str(e)}"}), 500

    response.update({
        "api_version": API_VERSION
    })
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
