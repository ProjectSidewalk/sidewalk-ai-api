from collections import namedtuple
from contextlib import contextmanager
from flask import Blueprint, Flask, current_app, g, request, jsonify
from werkzeug.exceptions import BadRequest
from PIL import Image
from sidewalk_ai_api.panorama import Panorama
from sidewalk_ai_api.model_map import MODEL_MAP, accuracy_map_sha256
from sidewalk_panorama_tools.geometry import label_view
import cv2
import hmac
import io
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

# The routes and hooks live on a blueprint; create_app() builds the Flask app around it with the models.
api = Blueprint("api", __name__)


@api.before_app_request
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


@api.after_app_request
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

accuracy_mappings = {label_type: json.load(open(f"accuracy_mappings/{label_type}.json"))
                 for label_type in VALIDATOR_LABEL_TYPES}


ACCURACY_MAPS_PROVENANCE = "accuracy_mappings/provenance.json"


def check_accuracy_maps(label_types=VALIDATOR_LABEL_TYPES, provenance_path=ACCURACY_MAPS_PROVENANCE):
    """Raises RuntimeError unless every accuracy map is the one fitted for the validator revision MODEL_MAP pins, and
    is well formed: both verdicts, ascending confidence keys in [0.5, 1], non-decreasing accuracies in [0, 1].

    Three things must agree per label type: the file's SHA-256, MODEL_MAP's ``accuracy_map_sha256``, and the
    ``validator_revision`` + ``sha256`` recorded for that map in ``accuracy_mappings/provenance.json`` against the
    pinned ``revision``. Called by load_models(), so re-pinning other weights without refitting the map (or
    swapping a map) stops the process at startup instead of serving an accuracy that belongs to another model.
    """
    with open(provenance_path) as f:
        provenance = json.load(f).get("maps", {})
    for label_type in label_types:
        path = f"accuracy_mappings/{label_type}.json"
        entry = MODEL_MAP["validator_models"][label_type]
        want = entry.get("accuracy_map_sha256")
        got = accuracy_map_sha256(path)
        if got != want:
            raise RuntimeError(f"{path} (sha256 {got}) is not the map MODEL_MAP binds to validator {label_type} "
                               f"({want}); refit the map for these weights or fix accuracy_map_sha256")
        prov = provenance.get(label_type)
        if not prov:
            raise RuntimeError(f"{provenance_path} has no entry for {label_type}")
        if prov.get("sha256") != got:
            raise RuntimeError(f"{provenance_path}: {label_type} sha256 {prov.get('sha256')} is not {path}'s {got}")
        if prov.get("validator_revision") != entry.get("revision"):
            raise RuntimeError(f"{path} was fitted for validator revision {prov.get('validator_revision')!r} but "
                               f"MODEL_MAP pins {entry.get('revision')!r} for {label_type}; refit the map for the "
                               f"pinned weights (and update {provenance_path}) before serving them")
        m = accuracy_mappings[label_type]
        if set(m) != {"correct", "incorrect"}:
            raise RuntimeError(f"{path}: expected verdicts correct and incorrect, got {sorted(m)}")
        for verdict, table in m.items():
            keys = [float(k) for k in table]
            values = list(table.values())
            if keys != sorted(keys) or keys[0] != 0.5 or keys[-1] != 1.0:
                raise RuntimeError(f"{path} {verdict}: thresholds must ascend from 0.5 to 1.0")
            if any(not 0.0 <= v <= 1.0 for v in values) or any(b < a for a, b in zip(values, values[1:])):
                raise RuntimeError(f"{path} {verdict}: accuracies must be non-decreasing in [0, 1]")

with open('API_VERSION', 'r') as file:
    API_VERSION = file.read().strip()

# The depth model, the taggers and the validators (by label type): what /process runs.
Models = namedtuple("Models", "depth taggers validators")


def load_models():
    """Loads every model onto the GPU, once per process (a minute or two). Exits with code 3 if there is no GPU,
    unless SIDEWALK_ALLOW_CPU=1, rather than silently running every model on the CPU; code 3 makes gunicorn stop
    instead of restarting the worker forever."""
    if not torch.cuda.is_available() and os.environ.get("SIDEWALK_ALLOW_CPU") != "1":
        print("No GPU found. Check the GPU driver and that the container was started with --gpus all, "
              "or set SIDEWALK_ALLOW_CPU=1 to run on the CPU.", file=sys.stderr)
        sys.exit(3)
    check_accuracy_maps()
    # Imported here, not at the top, so tests can import this module without transformers or the submodule.
    from sidewalk_ai_api.tagger import ImageTagger
    from sidewalk_ai_api.validator import ImageValidator
    from sidewalk_ai_api.depthanything import DepthAnythingPredictor
    return Models(
        depth=DepthAnythingPredictor(),
        taggers={label_type: ImageTagger(label_type=label_type) for label_type in TAGGER_LABEL_TYPES},
        validators={label_type: ImageValidator(label_type=label_type) for label_type in VALIDATOR_LABEL_TYPES},
    )


def create_app(models=None):
    """Builds the app. Production (ignition.sh: gunicorn "main:create_app()") passes nothing, so the real models
    load here, at startup; tests pass a Models of stand-ins.

    Example::

        app = create_app()                                    # loads the models
        app = create_app(Models(stub_depth, {}, validators))  # tests
    """
    app = Flask(__name__)
    app.config["MODELS"] = load_models() if models is None else models
    app.register_blueprint(api)
    return app


# Rig-tilt correction (docs/API_USAGE.md, "Camera pose and tilt correction"). The default beta is the fraction of the
# rig tilt measured to leak into a stored label point for the pose record the website sends (sidewalk-panorama-tools
# #191). The package's DEFAULT_BETA is the same value; it is spelled out here because it is part of this API's contract.
DEFAULT_TILT_BETA = 0.95
# Real rig tilt is a few degrees; a value near 90 is a bad record, and would silently apply a huge rotation.
MAX_POSE_DEG = 30.0
MAX_TILT_BETA = 1.5
COORD_FRAMES = ("level", "rig")


class InvalidField(BadRequest):
    """A form field that fails validation; the error handler returns its description as a JSON 400."""


@api.app_errorhandler(BadRequest)
def bad_request(e):
    return jsonify({"error": e.description}), 400


def parse_float(name, default=None, nan_is_none=False, lo=None, hi=None, required=False):
    """The form field `name` as a float, checked against [lo, hi] (inclusive; None for no bound).

    Absent or empty gives `default` (or InvalidField if `required`); NaN gives None when `nan_is_none` (a NULL
    database column, "unknown"); anything else that isn't a finite number in range raises InvalidField.

    Example: parse_float("camera_pitch", nan_is_none=True, lo=-30, hi=30) -> 3.0, or None when absent.
    """
    raw = request.form.get(name, "").strip()
    if not raw:
        if required:
            raise InvalidField(f"Missing {name} parameter")
        return default
    try:
        value = float(raw)
    except ValueError:
        raise InvalidField(f"{name} must be a number")
    if math.isnan(value) and nan_is_none:
        return None
    if not math.isfinite(value):
        raise InvalidField(f"{name} must be finite")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        if lo is None:
            raise InvalidField(f"{name} must be at most {hi:g}")
        if hi is None:
            raise InvalidField(f"{name} must be at least {lo:g}")
        raise InvalidField(f"{name} must be between {lo:g} and {hi:g}")
    return value


def parse_tilt_fields():
    """Reads the optional pose fields. -> (camera_pitch, camera_roll, tilt_beta, coord_frame).

    camera_pitch/camera_roll are None when unknown (absent, empty or NaN); tilt_beta defaults to 0.95; coord_frame to
    "level". "rig" means x and y already point at image pixels (e.g. the AI labeler's own labels), so the caller
    never corrects them.

    Example: camera_pitch=3, nothing else -> (3.0, None, 0.95, "level").
    """
    camera_pitch = parse_float("camera_pitch", nan_is_none=True, lo=-MAX_POSE_DEG, hi=MAX_POSE_DEG)
    camera_roll = parse_float("camera_roll", nan_is_none=True, lo=-MAX_POSE_DEG, hi=MAX_POSE_DEG)
    tilt_beta = parse_float("tilt_beta", default=DEFAULT_TILT_BETA, lo=0.0, hi=MAX_TILT_BETA)
    coord_frame = request.form.get("coord_frame", "").strip() or "level"
    if coord_frame not in COORD_FRAMES:
        raise InvalidField(f"coord_frame must be one of {list(COORD_FRAMES)}")
    return camera_pitch, camera_roll, tilt_beta, coord_frame


# The WebP quality the training crops were written with (sidewalk-validator-ai data_generation.save_webp).
WEBP_QUALITY = 95


def webp_round_trip(image, quality=WEBP_QUALITY):
    """`image` (PIL, RGB) encoded as WebP at `quality` and decoded again, as every training crop was.

    Example: webp_round_trip(Image.fromarray(crop)) -> a new RGB PIL image with WebP's lossy changes applied.
    """
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


@api.route("/process", methods=["POST"])
def process():
    models = current_app.config["MODELS"]
    if "label_type" not in request.form:
        return jsonify({"error": "Missing label_type parameter"}), 400
    if "panorama_id" not in request.form:
        return jsonify({"error": "Missing panorama_id parameter"}), 400
    # x and y are normalized pano coordinates, not pixels.
    label_x = parse_float("x", lo=0.0, hi=1.0, required=True)
    label_y = parse_float("y", lo=0.0, hi=1.0, required=True)

    label_type = request.form["label_type"]
    if label_type not in VALIDATOR_LABEL_TYPES:
        return jsonify({"error": f"Invalid label_type. Choose from {VALIDATOR_LABEL_TYPES}"}), 400

    camera_pitch, camera_roll, tilt_beta, coord_frame = parse_tilt_fields()

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
        width = panorama.panorama_image.shape[1]
        # The shared geometry (sidewalk-panorama-tools): a 90 degree view aimed at the label, 2048 px for the 8192 px
        # pano. In the level frame, a known camera_pitch gives the rig-tilt rotation (None when it is zero: pitch 0
        # and no roll, or beta 0); the rig frame never rotates.
        try:
            if coord_frame == "level" and camera_pitch is not None:
                view, tilt = label_view(label_x, label_y, 90, width // 4, camera_pitch, camera_roll, tilt_beta)
            else:
                view, tilt = label_view(label_x, label_y, 90, width // 4)
        except ValueError as e:  # the package's own input checks (e.g. y_norm outside [0, 1]): the request's fault
            raise InvalidField(str(e))
        tilt_applied = tilt is not None
        perspective_image = panorama.to_perspective_image(view, tilt)

    with timed("depth"):
        depth = models.depth.predict_depth(perspective_image)

    center_x, center_y = perspective_image.shape[1] // 2, perspective_image.shape[0] // 2
    img_h, img_w = perspective_image.shape[:2]

    # Half-size of the crop around the label, in pixels: 6100 / depth in metres, so closer features get larger
    # crops. Computed in float64 as training's data_generation.py does; int(float32(1 / d) * 6100) differs for about
    # one label in 90,000 (sidewalk-validator-ai docs/serving-parity-2026-10.md).
    depth_at_label = float(depth[center_y, center_x])
    if not math.isfinite(depth_at_label) or depth_at_label <= 0:   # training's depth_sized_crop raises the same way
        raise InvalidField(f"bad depth {depth_at_label} at the label: cannot size the crop")
    crop_size_half = int(6100 / depth_at_label)

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
        classifier = models.taggers[label_type]
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
    validator = models.validators[label_type]
    try:
        with timed("validator"):
            # The validators were trained on crops saved as quality-95 WebP (sidewalk-validator-ai
            # data_generation.save_webp), so they get the same lossy round trip here, in memory (~70 ms, counted in
            # this stage); on raw crops p(incorrect) moved by up to 0.11. The taggers above keep the raw crop: their
            # training data did not go through this pipeline.
            validator_image = webp_round_trip(perspective_image)
            validation_result, validation_confidence = validator.validate(validator_image)

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
    create_app().run(host="0.0.0.0", port=5000)
