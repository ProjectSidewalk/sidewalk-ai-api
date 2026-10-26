"""/process through Flask's test client, with the models and the panorama replaced by stand-ins.

Covers the pose fields (camera_pitch, camera_roll, tilt_beta, coord_frame): their validation, the response fields,
and that the tilt moves the view only for coord_frame=level. The rendered view is the image /process hands to the
depth model, which the stub records.
"""

import importlib
import inspect
import io
import json
import re
import runpy
import sys
import time
import types
from datetime import datetime

import cv2
import numpy as np
import pano_pose
import pytest
from PIL import Image
from sidewalk_panorama_tools.geometry import label_pixel, label_view, tilt_matrix

import main
from sidewalk_ai_api import model_map
from sidewalk_ai_api.panorama import Panorama

AUTH = {"Authorization": "Bearer test-key"}
PANO_W, PANO_H = 2048, 1024  # /process renders width // 4 = a 512 px view
VIEW = PANO_W // 4


def stub_depth_map(shape):
    """A depth gradient, different along each axis, so a wrong index into it changes the crop."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return 20.0 + 0.05 * cols + 0.1 * rows


class StubDepth:
    """Records the perspective image and reports stub_depth_map (58.4 m at the centre of a 512 px view: a crop
    half-size of int(6100 / 58.4) = 104 px)."""
    def __init__(self):
        self.images = []

    def predict_depth(self, image):
        self.images.append(image)
        return stub_depth_map(image.shape[:2])


class StubValidator:
    """Records the crop it is given."""
    def __init__(self):
        self.images = []

    def validate(self, image):
        self.images.append(image)
        return "correct", 0.97


class StubTagger:
    """Records the crop it is given."""
    def __init__(self):
        self.images = []

    def inference(self, image):
        self.images.append(image)
        return [], [], {}


@pytest.fixture
def app_with(monkeypatch):
    """Returns (client, depth stub, set_pano) with the models stubbed and a synthetic panorama served from 'cache'."""
    depth = StubDepth()
    validator = StubValidator()
    tagger = StubTagger()
    models = main.Models(depth=depth,
                         taggers={t: tagger for t in main.TAGGER_LABEL_TYPES},
                         validators={t: validator for t in main.VALIDATOR_LABEL_TYPES})
    depth.validator, depth.tagger = validator, tagger  # so tests can reach the crops through the depth stub
    state = {"image": np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8)}

    def fake_panorama(pano_id, cached_image_path=None):
        pano = Panorama.__new__(Panorama)
        pano.pano_id, pano.panorama_image, pano.image_source, pano.zoom = pano_id, state["image"], "cache", None
        return pano

    monkeypatch.setattr(main, "Panorama", fake_panorama)

    def set_pano(image):
        state["image"] = image

    return main.create_app(models).test_client(), depth, set_pano


def post(client, **fields):
    data = {"label_type": "curbramp", "panorama_id": "testpano", "x": "0.3", "y": "0.6"}
    data.update({k: str(v) for k, v in fields.items()})
    return client.post("/process", data=data, headers=AUTH)


def marker_pano(px, py, radius=1.5):
    """A black pano with a small white disc centred on continuous pixel (px, py) (centre = i + 0.5)."""
    yy, xx = np.mgrid[0:PANO_H, 0:PANO_W] + 0.5
    dx = (xx - px + PANO_W / 2) % PANO_W - PANO_W / 2
    disc = (dx ** 2 + (yy - py) ** 2) <= radius ** 2
    img = np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8)
    img[disc] = 255
    return img


def centroid(image):
    """Intensity-weighted centroid of a rendered view, in continuous output pixels (centre = i + 0.5)."""
    w = image[..., 0].astype(np.float64)
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]] + 0.5
    return (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()


# --- request validation ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fields, message", [
    ({"x": ""}, "Missing x parameter"),
    ({"x": "abc"}, "x must be a number"),
    ({"y": "nan"}, "y must be finite"),
    ({"x": "1.01"}, "x must be between 0 and 1"),
    ({"camera_pitch": "abc"}, "camera_pitch must be a number"),
    ({"camera_pitch": "inf"}, "camera_pitch must be finite"),
    ({"camera_pitch": "30.5"}, "camera_pitch must be between -30 and 30"),
    ({"camera_pitch": "85"}, "camera_pitch must be between -30 and 30"),
    ({"camera_roll": "-31"}, "camera_roll must be between -30 and 30"),
    ({"camera_roll": "-inf"}, "camera_roll must be finite"),
    ({"tilt_beta": "-0.1"}, "tilt_beta must be between"),
    ({"tilt_beta": "1.6"}, "tilt_beta must be between"),
    ({"tilt_beta": "nan"}, "tilt_beta must be finite"),
    ({"tilt_beta": "x"}, "tilt_beta must be a number"),
    ({"coord_frame": "gravity"}, "coord_frame must be one of"),
])
def test_bad_pose_fields_are_400(app_with, fields, message):
    client, depth, _ = app_with
    r = post(client, **fields)
    assert r.status_code == 400
    assert message in r.get_json()["error"]
    assert depth.images == []  # rejected before any work


def test_malformed_body_is_a_json_400(app_with):
    """A missing required field is a JSON 400, like every other validation error."""
    client, _, _ = app_with
    r = client.post("/process", data={"label_type": "curbramp", "x": "0.5", "y": "0.5"}, headers=AUTH)
    assert r.status_code == 400 and "error" in r.get_json()


def test_models_load_only_without_stand_ins(monkeypatch):
    """create_app() with no argument is the only path that loads models, and it refuses to start without a GPU."""
    monkeypatch.delenv("SIDEWALK_ALLOW_CPU", raising=False)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exit_info:
        main.create_app()
    assert exit_info.value.code == 3
    assert not hasattr(main, "app")


def test_missing_key_is_still_401(app_with):
    client, _, _ = app_with
    r = client.post("/process", data={"label_type": "curbramp", "panorama_id": "p", "x": "0.5", "y": "0.5"})
    assert r.status_code == 401


# --- response fields ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fields, applied, beta, frame", [
    ({}, False, None, "level"),
    ({"camera_pitch": "", "camera_roll": ""}, False, None, "level"),
    ({"camera_pitch": "nan"}, False, None, "level"),
    ({"camera_roll": "2"}, False, None, "level"),          # roll without pitch: no correction
    ({"camera_pitch": "3"}, True, 0.95, "level"),           # default beta, roll absent -> 0
    ({"camera_pitch": "-2.5", "camera_roll": "1", "tilt_beta": "0.88"}, True, 0.88, "level"),
    ({"camera_pitch": "3", "coord_frame": "rig"}, False, None, "rig"),
    ({"camera_pitch": "3", "camera_roll": "1", "coord_frame": "rig", "tilt_beta": "1"}, False, None, "rig"),
    ({"camera_pitch": "0"}, False, None, "level"),                         # zero rotation: not applied
    ({"camera_pitch": "0", "camera_roll": ""}, False, None, "level"),
    ({"camera_pitch": "3", "tilt_beta": "0"}, False, None, "level"),
    ({"camera_pitch": "0", "camera_roll": "2"}, True, 0.95, "level"),      # roll alone rotates once pitch is known
    ({"x": "1.0", "camera_pitch": "3"}, True, 0.95, "level"),  # the seam: label_view wraps x
])
def test_response_fields(app_with, fields, applied, beta, frame):
    client, _, _ = app_with
    r = post(client, **fields)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["tilt_applied"] is applied
    assert body["tilt_beta"] == beta
    assert body["coord_frame"] == frame
    assert body["image_source"] == "cache"
    assert body["api_version"] == main.API_VERSION
    assert body["validation_result"] == "correct"


def test_no_pitch_is_the_untilted_view(app_with):
    """Without camera_pitch the view is label_view's with no rotation, whatever beta and coord_frame say."""
    client, depth, set_pano = app_with
    rng = np.random.default_rng(0)
    set_pano(rng.integers(0, 256, (PANO_H, PANO_W, 3), dtype=np.uint8))
    assert post(client).status_code == 200
    assert post(client, tilt_beta="1.2", camera_roll="4").status_code == 200
    assert post(client, coord_frame="rig").status_code == 200
    assert post(client, camera_pitch="0").status_code == 200   # a known zero pitch is no rotation either
    assert post(client, camera_pitch="3", tilt_beta="0").status_code == 200
    first = depth.images[0]
    for image in depth.images[1:]:
        assert np.array_equal(image, first)
    assert first.shape == (VIEW, VIEW, 3)


# --- the tilt moves the view only in the level frame ----------------------------------------------------------------

@pytest.mark.parametrize("x, y", [(0.3, 0.6), (0.5, 0.62), (0.85, 0.55)])
def test_marker_at_corrected_pixel_is_centred_only_for_level(app_with, x, y):
    """A label stored at (x, y) in the level frame; its content sits at pano_pose.corrected_pixel in the rig-frame
    pano. With camera_pitch=3 and coord_frame=level the view must put that content at its centre; coord_frame=rig
    must not correct, leaving it off-centre by what label_pixel predicts."""
    client, depth, set_pano = app_with
    pitch, beta = 3.0, 0.95
    rx, ry = pano_pose.corrected_pixel(x * PANO_W, y * PANO_H, PANO_W, PANO_H, pitch, 0.0, beta)
    set_pano(marker_pano(rx, ry))

    r = post(client, x=x, y=y, camera_pitch=pitch)
    assert r.get_json()["tilt_applied"] is True
    cx, cy = centroid(depth.images[-1])
    assert abs(cx - VIEW / 2) <= 1 and abs(cy - VIEW / 2) <= 1

    r = post(client, x=x, y=y, camera_pitch=pitch, coord_frame="rig")
    assert r.get_json()["tilt_applied"] is False
    cx, cy = centroid(depth.images[-1])
    view, _ = label_view(x, y, 90, VIEW)
    ex, ey = label_pixel(x * PANO_W, y * PANO_H, PANO_W, PANO_H, view, None, tilt_matrix(pitch, 0.0, beta))
    assert abs(cx - ex) <= 1 and abs(cy - ey) <= 1
    # And the offset is real: several pixels, so the level-frame pass above isn't trivially true.
    assert np.hypot(ex - VIEW / 2, ey - VIEW / 2) > 4


def test_rig_frame_marker_at_stored_pixel_is_centred(app_with):
    """coord_frame=rig: x/y already name the image pixel, so a marker there is centred even with a pitch sent."""
    client, depth, set_pano = app_with
    x, y = 0.3, 0.6
    set_pano(marker_pano(x * PANO_W, y * PANO_H))
    r = post(client, x=x, y=y, camera_pitch=3, coord_frame="rig")
    assert r.get_json()["tilt_applied"] is False
    cx, cy = centroid(depth.images[-1])
    assert abs(cx - VIEW / 2) <= 1 and abs(cy - VIEW / 2) <= 1


# --- review additions -----------------------------------------------------------------------------------------------

def test_nan_roll_with_pitch_is_unknown_roll(app_with):
    """The website sends NULL roll often: NaN roll with a pitch is roll 0, and the tilt is still applied."""
    client, depth, set_pano = app_with
    set_pano(np.random.default_rng(1).integers(0, 256, (PANO_H, PANO_W, 3), dtype=np.uint8))
    body = post(client, camera_pitch="3", camera_roll="nan").get_json()
    assert body["tilt_applied"] is True and body["tilt_beta"] == 0.95
    post(client, camera_pitch="3")
    assert np.array_equal(depth.images[0], depth.images[1])


def test_label_view_value_error_is_a_400(app_with, monkeypatch):
    """A ValueError from the package's own input checks (e.g. y_norm outside [0, 1]) is the request's fault."""
    client, _, _ = app_with

    def bad_view(*args, **kwargs):
        raise ValueError("y_norm must be in [0, 1], got 1.5")

    monkeypatch.setattr(main, "label_view", bad_view)
    r = post(client)
    assert r.status_code == 400
    assert r.get_json()["error"] == "y_norm must be in [0, 1], got 1.5"


@pytest.mark.parametrize("kwargs, raw, message", [
    ({"lo": 0.0}, "-1", "v must be at least 0"),
    ({"hi": 2.0}, "3", "v must be at most 2"),
    ({"lo": 0.0, "hi": 2.0}, "3", "v must be between 0 and 2"),
])
def test_parse_float_bounds_messages(kwargs, raw, message):
    app = main.create_app(main.Models(None, {}, {}))
    with app.test_request_context("/process", method="POST", data={"v": raw}):
        with pytest.raises(main.InvalidField) as e:
            main.parse_float("v", **kwargs)
    assert e.value.description == message


def test_parse_float_one_sided_ok():
    app = main.create_app(main.Models(None, {}, {}))
    with app.test_request_context("/process", method="POST", data={"v": "5"}):
        assert main.parse_float("v", lo=0.0) == 5.0
        assert main.parse_float("w", default=1.5, hi=2.0) == 1.5


def test_server_timing_header(app_with):
    client, _, _ = app_with
    timing = post(client).headers["Server-Timing"]
    for stage in ("pano;desc=cache;dur=", "project;dur=", "depth;dur=", "tagger;dur=", "validator;dur=", "total;dur="):
        assert stage in timing


def test_auth_applies_to_unknown_paths(app_with):
    client, _, _ = app_with
    assert client.get("/no-such-path").status_code == 401
    assert client.get("/no-such-path", headers=AUTH).status_code == 404


def test_crop_uses_depth_at_view_centre(app_with):
    """The crop is the view around its centre, half-size int(6100 / depth[cy, cx]), resized to 640 px, as RGB: the
    tagger gets it raw, the validator after the q95 WebP round trip the training crops went through."""
    client, depth, set_pano = app_with
    set_pano(np.random.default_rng(2).integers(0, 256, (PANO_H, PANO_W, 3), dtype=np.uint8))
    assert post(client).status_code == 200
    view = depth.images[-1]
    c = VIEW // 2
    half = int(6100 / stub_depth_map(view.shape[:2])[c, c])
    assert half == 104
    expected = cv2.cvtColor(cv2.resize(view[c - half:c + half, c - half:c + half], (640, 640),
                                       interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    assert np.array_equal(np.asarray(depth.tagger.images[-1]), expected)
    served = depth.validator.images[-1]
    assert served.mode == "RGB" and served.size == (640, 640)
    assert np.array_equal(np.asarray(served), np.asarray(main.webp_round_trip(Image.fromarray(expected))))
    # q95 WebP is lossy, so on noise the validator's copy is not the raw crop (and the tagger's still is).
    assert not np.array_equal(np.asarray(served), expected)


def test_crop_half_is_float64_division(app_with, monkeypatch):
    """int(6100 / depth) in float64, as training: a depth where float32(1 / d) * 6100 rounds the other way."""
    client, depth, set_pano = app_with
    d = np.float32(30.5)                                 # 6100 / 30.5 = 200.0 exactly in float64 ...
    assert int(np.float32(1) / d * 6100) == 199          # ... but 199 through the float32 reciprocal
    def flat_depth(image):
        depth.images.append(image)
        return np.full(image.shape[:2], d, dtype=np.float32)

    monkeypatch.setattr(depth, "predict_depth", flat_depth)
    set_pano(np.random.default_rng(3).integers(0, 256, (PANO_H, PANO_W, 3), dtype=np.uint8))
    assert post(client).status_code == 200
    assert np.asarray(depth.tagger.images[-1]).shape == (640, 640, 3)
    # a 200 px half-size cuts a 400 px square; 199 would too, so check the pixels: the raw crop must be the 400 px one
    view = depth.images[-1]
    c = VIEW // 2
    expected = cv2.cvtColor(cv2.resize(view[c - 200:c + 200, c - 200:c + 200], (640, 640),
                                       interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    assert np.array_equal(np.asarray(depth.tagger.images[-1]), expected)


def test_crop_half_truncates_a_fractional_part_above_half(app_with, monkeypatch):
    """int(), not round(): 6100 / depth = 200.7 must give a 200 px half-size (round would give 201)."""
    client, depth, set_pano = app_with
    d = np.float32(6100 / 200.7)
    assert 200.5 < 6100 / float(d) < 201 and int(6100 / float(d)) == 200 and round(6100 / float(d)) == 201

    def flat_depth(image):
        depth.images.append(image)
        return np.full(image.shape[:2], d, dtype=np.float32)

    monkeypatch.setattr(depth, "predict_depth", flat_depth)
    set_pano(np.random.default_rng(5).integers(0, 256, (PANO_H, PANO_W, 3), dtype=np.uint8))
    assert post(client).status_code == 200
    view = depth.images[-1]
    c = VIEW // 2
    expected = cv2.cvtColor(cv2.resize(view[c - 200:c + 200, c - 200:c + 200], (640, 640),
                                       interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    assert np.array_equal(np.asarray(depth.tagger.images[-1]), expected)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_depth_at_the_label_is_a_json_400(app_with, monkeypatch, bad):
    """Depth 0, negative, NaN or inf cannot size a crop: a JSON 400 naming it, as training's ValueError, not an
    HTML 500 from ZeroDivisionError or int(nan)."""
    client, depth, set_pano = app_with
    monkeypatch.setattr(depth, "predict_depth", lambda image: np.full(image.shape[:2], bad, dtype=np.float32))
    r = post(client)
    assert r.status_code == 400 and r.is_json and "bad depth" in r.get_json()["error"]


def test_webp_round_trip_runs_inside_the_validator_stage_and_its_error_handling(app_with, monkeypatch):
    """The round trip is timed as part of the validator stage and an encoder failure is a JSON 500, not HTML."""
    client, depth, set_pano = app_with
    r = post(client)
    assert r.status_code == 200 and "validator;dur=" in r.headers["Server-Timing"]

    def boom(image, quality=95):
        raise RuntimeError("no WebP encoder")

    monkeypatch.setattr(main, "webp_round_trip", boom)
    r = post(client)
    assert r.status_code == 500 and r.is_json and "no WebP encoder" in r.get_json()["error"]


def test_webp_round_trip_is_q95_pillow():
    """The validator's copy is what Pillow's WebP encoder at quality 95 decodes to: lossy, RGB, same size."""
    crop = Image.fromarray(np.random.default_rng(4).integers(0, 256, (480, 640, 3), dtype=np.uint8))
    out = main.webp_round_trip(crop)
    buf = io.BytesIO()
    crop.save(buf, format="WEBP", quality=95)
    assert np.array_equal(np.asarray(out), np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    assert out.mode == "RGB" and out.size == crop.size
    assert not np.array_equal(np.asarray(out), np.asarray(crop))
    assert main.WEBP_QUALITY == 95


# --- model map and accuracy maps -------------------------------------------------------------------------------------

def test_pinned_revision_rejects_placeholders():
    """A placeholder or malformed revision is a RuntimeError naming the model, so an unpublished model cannot load."""
    good = {"model_id": "m", "revision": "0" * 40}
    assert model_map.pinned_revision(good) == "0" * 40
    for bad in (model_map.PLACEHOLDER_REVISION, "", "main", "abc123", "0" * 39, "G" * 40):
        with pytest.raises(RuntimeError, match="revision is not pinned"):
            model_map.pinned_revision({"model_id": "m", "revision": bad}, "validator curbramp")


def test_image_validator_refuses_a_placeholder_before_loading(monkeypatch):
    """ImageValidator goes through pinned_revision(): with the placeholder it raises before transformers is asked
    for anything (a dropped call would download the repo's HEAD)."""
    calls = []

    class NeverLoaded:
        @staticmethod
        def from_pretrained(*a, **k):
            calls.append((a, k))
            raise AssertionError("from_pretrained must not be reached with a placeholder revision")

    fake_transformers = types.SimpleNamespace(AutoImageProcessor=NeverLoaded, AutoModelForImageClassification=NeverLoaded)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("sidewalk_ai_api.validator", None)
    validator = importlib.import_module("sidewalk_ai_api.validator")
    monkeypatch.setitem(model_map.MODEL_MAP["validator_models"]["curbramp"], "revision", model_map.PLACEHOLDER_REVISION)
    with pytest.raises(RuntimeError, match="validator curbramp .* revision is not pinned"):
        validator.ImageValidator("curbramp")
    assert calls == []
    sys.modules.pop("sidewalk_ai_api.validator", None)


def test_download_hf_models_refuses_a_placeholder_before_downloading(monkeypatch):
    """download_hf_models.py checks every revision first: with one placeholder it raises before any download."""
    calls = []

    def never(*a, **k):
        calls.append((a, k))
        raise AssertionError("no download may start with an unpinned entry")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=never, snapshot_download=never))
    for t in main.VALIDATOR_LABEL_TYPES:          # pin every other entry, leave exactly one placeholder
        monkeypatch.setitem(model_map.MODEL_MAP["validator_models"][t], "revision",
                            model_map.PLACEHOLDER_REVISION if t == "obstacle" else "b" * 40)
    with pytest.raises(RuntimeError, match="validator_models obstacle .* revision is not pinned"):
        runpy.run_path("download_hf_models.py", run_name="__main__")
    assert calls == []


def test_model_map_entries_are_pinned():
    """Every entry is a 40-hex commit (never a branch name or the placeholder). A new validator may carry
    PLACEHOLDER_REVISION on a branch until it is published, but main must never be green with one."""
    for label_type, entry in model_map.MODEL_MAP["tagger_models"].items():
        model_map.pinned_revision(entry, f"tagger {label_type}")
    for label_type, entry in model_map.MODEL_MAP["validator_models"].items():
        model_map.pinned_revision(entry, f"validator {label_type}")
        assert label_type in main.VALIDATOR_LABEL_TYPES


def test_model_map_dates_parse_as_the_website_parses_them():
    """training_date must be MM-dd-yyyy: the website parses validator_training_date and tagger_training_date with
    DateTimeFormatter.ofPattern("MM-dd-yyyy") inside a try that, on failure, logs and casts no vote."""
    for kind in ("tagger_models", "validator_models"):
        for label_type, entry in model_map.MODEL_MAP[kind].items():
            raw = entry["training_date"]
            assert re.fullmatch(r"\d{2}-\d{2}-\d{4}", raw), (kind, label_type, raw)
            parsed = datetime.strptime(raw, "%m-%d-%Y")
            assert parsed.strftime("%m-%d-%Y") == raw and 2024 <= parsed.year <= 2100, (kind, label_type, raw)


def test_image_source_values_are_the_websites_enum(app_with):
    """image_source must be "cache" or "download": the website maps it onto its ai_image_source enum
    (AiImageSource in LabelAiAssessmentTable.scala) inside the same try."""
    website_enum = {"cache", "download"}
    literals = set(re.findall(r'self\.image_source = "([^"]*)"', inspect.getsource(Panorama)))
    assert literals and literals <= website_enum, literals
    client, depth, set_pano = app_with
    assert post(client).get_json()["image_source"] in website_enum


def test_incorrect_maps_are_held_back():
    """Every incorrect map ships capped at HOLD_BACK_CAP until the deployment prior is settled and the revision has
    production outcomes (sidewalk-validator-ai#56): a map that could cast a Disagree must not land here by accident."""
    provenance = json.load(open(main.ACCURACY_MAPS_PROVENANCE))["maps"]
    for label_type in main.VALIDATOR_LABEL_TYPES:
        table = main.accuracy_mappings[label_type]["incorrect"]
        assert max(table.values()) <= model_map.HOLD_BACK_CAP, label_type
        # and the provenance says so, which is what makes the server enforce the cap at startup
        assert "incorrect" in provenance[label_type]["hold_back"], label_type
    assert model_map.HOLD_BACK_CAP < 0.92


def test_accuracy_maps_are_bound_to_the_model_map():
    """Each validator entry records the SHA-256 of the map fitted for it; check_accuracy_maps() (run at startup)
    refuses any other file, and a malformed map."""
    for label_type in main.VALIDATOR_LABEL_TYPES:
        entry = model_map.MODEL_MAP["validator_models"][label_type]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["accuracy_map_sha256"]), label_type
        assert model_map.accuracy_map_sha256(f"accuracy_mappings/{label_type}.json") == entry["accuracy_map_sha256"]
    main.check_accuracy_maps()


def test_accuracy_maps_share_one_grid():
    """All five maps use the same 231 confidence thresholds (the grid calibrate.py writes), per verdict."""
    grids = {(t, v): tuple(main.accuracy_mappings[t][v]) for t in main.VALIDATOR_LABEL_TYPES for v in ("correct", "incorrect")}
    assert len(set(grids.values())) == 1 and len(next(iter(grids.values()))) == 231


def test_check_accuracy_maps_rejects_a_map_fitted_for_another_revision(monkeypatch, tmp_path):
    """provenance.json binds each map to a validator revision: re-pinning other weights without refitting is refused."""
    prov = json.load(open("accuracy_mappings/provenance.json"))
    assert set(prov["maps"]) == set(main.VALIDATOR_LABEL_TYPES)
    for t, m in prov["maps"].items():
        assert m["validator_revision"] == model_map.MODEL_MAP["validator_models"][t]["revision"]
        assert m["sha256"] == model_map.MODEL_MAP["validator_models"][t]["accuracy_map_sha256"]
    monkeypatch.setitem(model_map.MODEL_MAP["validator_models"]["crosswalk"], "revision", "a" * 40)
    with pytest.raises(RuntimeError, match="fitted for validator revision .* but MODEL_MAP pins"):
        main.check_accuracy_maps(["crosswalk"])
    other = tmp_path / "provenance.json"
    other.write_text(json.dumps({"maps": {}}))
    with pytest.raises(RuntimeError, match="no entry for curbramp"):
        main.check_accuracy_maps(["curbramp"], provenance_path=str(other))


def _scratch_maps(tmp_path, monkeypatch):
    """A copy of accuracy_mappings/ (maps and provenance) under tmp_path, with cwd moved there, for tampering."""
    scratch = tmp_path / "accuracy_mappings"
    scratch.mkdir()
    for t in main.VALIDATOR_LABEL_TYPES:
        (scratch / f"{t}.json").write_bytes(open(f"accuracy_mappings/{t}.json", "rb").read())
    (scratch / "provenance.json").write_bytes(open("accuracy_mappings/provenance.json", "rb").read())
    monkeypatch.chdir(tmp_path)
    return scratch


def _rebind(scratch, label_type, monkeypatch, hub=None):
    """Point MODEL_MAP, provenance.json and main.accuracy_mappings at the (tampered) scratch map, as a committer
    updating both hashes would; with ``hub``, also publish it at the pinned revision of the stand-in hub."""
    sha = model_map.accuracy_map_sha256(scratch / f"{label_type}.json")
    entry = model_map.MODEL_MAP["validator_models"][label_type]
    monkeypatch.setitem(entry, "accuracy_map_sha256", sha)
    prov = json.load(open(scratch / "provenance.json"))
    prov["maps"][label_type]["sha256"] = sha
    (scratch / "provenance.json").write_text(json.dumps(prov))
    monkeypatch.setitem(main.accuracy_mappings, label_type, json.load(open(scratch / f"{label_type}.json")))
    if hub is not None:
        hub.maps[(f"projectsidewalk/{entry['model_id']}", entry["revision"])] = (scratch / f"{label_type}.json").read_bytes()


def test_startup_refuses_an_uncapped_map_for_a_held_back_verdict(monkeypatch, tmp_path, fake_hub):
    """provenance.json says incorrect is held back: a map whose incorrect values rise above the cap is refused at
    startup even when both hashes were updated for it (the tests are not the only guard)."""
    scratch = _scratch_maps(tmp_path, monkeypatch)
    m = json.load(open(scratch / "crosswalk.json"))
    m["incorrect"]["1.0"] = 0.9768                                   # the uncapped Crosswalk maximum
    (scratch / "crosswalk.json").write_text(json.dumps(m, indent=4) + "\n")
    _rebind(scratch, "crosswalk", monkeypatch)
    with pytest.raises(RuntimeError, match="incorrect reaches 0.9768 but .* held back"):
        main.check_accuracy_maps(["crosswalk"])
    # exactly at the cap is fine; a hold-back of a verdict the map does not have is a provenance error
    m["incorrect"]["1.0"] = model_map.HOLD_BACK_CAP
    (scratch / "crosswalk.json").write_text(json.dumps(m, indent=4) + "\n")
    _rebind(scratch, "crosswalk", monkeypatch, hub=fake_hub)
    main.check_accuracy_maps(["crosswalk"])
    prov = json.load(open(scratch / "provenance.json"))
    prov["maps"]["crosswalk"]["hold_back"] = ["unsure"]
    (scratch / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(RuntimeError, match="holds back 'unsure', which is not a verdict"):
        main.check_accuracy_maps(["crosswalk"])


def test_check_accuracy_maps_compares_the_provenance_sha_with_the_file(monkeypatch, tmp_path):
    """Through the runtime branch: a provenance sha that is not the file's is refused even when MODEL_MAP's is."""
    scratch = _scratch_maps(tmp_path, monkeypatch)
    main.check_accuracy_maps()                                       # the committed set passes from the scratch copy
    prov = json.load(open(scratch / "provenance.json"))
    prov["maps"]["obstacle"]["sha256"] = "0" * 64
    (scratch / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(RuntimeError, match="obstacle sha256 0{64} is not accuracy_mappings/obstacle.json's"):
        main.check_accuracy_maps(["obstacle"])


# --- the hub's confidence_precision_map.json is the authority for each map (sidewalk-ai-api#24) ------------------------

OLD_CROSSWALK = {"revision": "b7936e5bd17a4b96db301f954e5d3a5fe9eaf361", "training_date": "08-16-2025"}  # 2025-08 pin


def _hub_key(label_type):
    entry = model_map.MODEL_MAP["validator_models"][label_type]
    return f"projectsidewalk/{entry['model_id']}", entry["revision"]


def test_hub_record_covers_the_model_map(fake_hub):
    """tests/hub_accuracy_maps.json records, for every pinned validator, the hub map's sha256 at that revision, and it
    is MODEL_MAP's; the stand-in hub serves each from the committed maps. A new pin needs its hash read from the hub."""
    record = json.load(open("tests/hub_accuracy_maps.json"))["maps"]
    for label_type, entry in model_map.MODEL_MAP["validator_models"].items():
        repo, revision = _hub_key(label_type)
        assert record.get(repo, {}).get(revision) == entry["accuracy_map_sha256"], label_type
        assert _hub_key(label_type) in fake_hub.maps, label_type
    main.check_accuracy_maps()
    assert sorted(c[2] for c in fake_hub.calls) == sorted(e["revision"] for e in model_map.MODEL_MAP["validator_models"].values())
    assert {c[1] for c in fake_hub.calls} == {model_map.HUB_ACCURACY_MAP}


def test_startup_refuses_two_swapped_maps_even_with_every_hash_updated(monkeypatch, tmp_path):
    """Swapping crosswalk.json and curbramp.json and updating both sha256s in MODEL_MAP and provenance.json passes
    every in-repo check; the hub still holds each revision's own map, so startup refuses it."""
    scratch = _scratch_maps(tmp_path, monkeypatch)
    cw, cr = (scratch / "crosswalk.json").read_bytes(), (scratch / "curbramp.json").read_bytes()
    (scratch / "crosswalk.json").write_bytes(cr)
    (scratch / "curbramp.json").write_bytes(cw)
    _rebind(scratch, "crosswalk", monkeypatch)
    _rebind(scratch, "curbramp", monkeypatch)
    for t in ("crosswalk", "curbramp"):
        with pytest.raises(RuntimeError, match=f"validator {t} .*confidence_precision_map.json at revision "
                                               f"{_hub_key(t)[1]} has sha256 .* not MODEL_MAP's"):
            main.check_accuracy_maps([t])


def test_startup_refuses_old_weights_repinned_with_the_new_map(monkeypatch, tmp_path, fake_hub):
    """Re-pinning the 2025-08 crosswalk weights in both MODEL_MAP and provenance.json while keeping the new map: that
    revision has no hub map (as on the real hub), so startup refuses it; had it one, a different map is refused too."""
    scratch = _scratch_maps(tmp_path, monkeypatch)
    for field, value in OLD_CROSSWALK.items():
        monkeypatch.setitem(model_map.MODEL_MAP["validator_models"]["crosswalk"], field, value)
    prov = json.load(open(scratch / "provenance.json"))
    prov["maps"]["crosswalk"].update(validator_revision=OLD_CROSSWALK["revision"],
                                     training_date=OLD_CROSSWALK["training_date"])
    (scratch / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(RuntimeError, match=f"validator crosswalk: cannot get confidence_precision_map.json at "
                                           f"revision {OLD_CROSSWALK['revision']}"):
        main.check_accuracy_maps(["crosswalk"])
    old_map = json.load(open(scratch / "crosswalk.json"))
    old_map["correct"]["1.0"] = 0.99                                 # any other map, as fitted for the old weights
    fake_hub.maps[_hub_key("crosswalk")] = json.dumps(old_map).encode()
    with pytest.raises(RuntimeError, match="has sha256 .* not MODEL_MAP's accuracy_map_sha256"):
        main.check_accuracy_maps(["crosswalk"])


@pytest.mark.parametrize("field, value", [("training_date", "08-16-2025"), ("model_id", "sidewalk-validator-ai-curbramp")])
def test_startup_refuses_provenance_metadata_drift(monkeypatch, tmp_path, field, value):
    """provenance.json's training_date and model_id must be MODEL_MAP's: either drifting alone is refused."""
    scratch = _scratch_maps(tmp_path, monkeypatch)
    prov = json.load(open(scratch / "provenance.json"))
    prov["maps"]["crosswalk"][field] = value
    (scratch / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(RuntimeError, match=f"crosswalk {field} '{value}' is not MODEL_MAP's"):
        main.check_accuracy_maps(["crosswalk"])


@pytest.mark.parametrize("field", ["training_date", "model_id"])
def test_startup_refuses_a_provenance_entry_missing_metadata(monkeypatch, tmp_path, field):
    """An absent training_date or model_id is refused like a wrong one, not skipped."""
    scratch = _scratch_maps(tmp_path, monkeypatch)
    prov = json.load(open(scratch / "provenance.json"))
    del prov["maps"]["crosswalk"][field]
    (scratch / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(RuntimeError, match=f"crosswalk {field} None is not MODEL_MAP's"):
        main.check_accuracy_maps(["crosswalk"])


def _run_download(monkeypatch, tmp_path, fake_hub, calls=None):
    """Runs download_hf_models.py against stand-in downloads (snapshot_download writes what the stand-in hub holds at
    each revision); appends every call, in order, to ``calls`` as (function, repo, allow_patterns or filename)."""
    calls = [] if calls is None else calls

    def snapshot_download(repo_id, revision=None, allow_patterns=None):
        calls.append(("snapshot_download", repo_id, allow_patterns))
        folder = tmp_path / "snapshots" / repo_id / revision
        folder.mkdir(parents=True, exist_ok=True)
        if model_map.HUB_ACCURACY_MAP in allow_patterns and (repo_id, revision) in fake_hub.maps:
            (folder / model_map.HUB_ACCURACY_MAP).write_bytes(fake_hub.maps[(repo_id, revision)])
        return str(folder)

    def hf_hub_download(repo_id, filename, revision=None):
        calls.append(("hf_hub_download", repo_id, filename))

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=hf_hub_download, snapshot_download=snapshot_download))
    runpy.run_path("download_hf_models.py", run_name="__main__")
    return calls


def test_download_hf_models_checks_the_hub_map(monkeypatch, tmp_path, fake_hub):
    """The image build downloads each validator's hub map and fails unless its sha256 is MODEL_MAP's: a map hash
    swapped between two entries, or a revision with no map, stops the build."""
    calls = _run_download(monkeypatch, tmp_path, fake_hub)
    n = len(main.VALIDATOR_LABEL_TYPES)
    assert [c[2] for c in calls[:n]] == [[model_map.HUB_ACCURACY_MAP]] * n         # every map, before any weights
    assert len(calls) == 2 * n + len(model_map.MODEL_MAP["tagger_models"])
    vm = model_map.MODEL_MAP["validator_models"]
    cw, cr = vm["crosswalk"]["accuracy_map_sha256"], vm["curbramp"]["accuracy_map_sha256"]
    monkeypatch.setitem(vm["crosswalk"], "accuracy_map_sha256", cr)
    monkeypatch.setitem(vm["curbramp"], "accuracy_map_sha256", cw)
    with pytest.raises(RuntimeError, match="validator_models crosswalk .* not MODEL_MAP's accuracy_map_sha256"):
        _run_download(monkeypatch, tmp_path, fake_hub)
    monkeypatch.setitem(vm["crosswalk"], "accuracy_map_sha256", cw)
    monkeypatch.setitem(vm["curbramp"], "accuracy_map_sha256", cr)
    monkeypatch.setitem(vm["crosswalk"], "revision", OLD_CROSSWALK["revision"])
    with pytest.raises(RuntimeError, match=f"revision {OLD_CROSSWALK['revision']} has no confidence_precision_map.json"):
        _run_download(monkeypatch, tmp_path, fake_hub)


def test_download_hf_models_checks_every_map_before_any_weights(monkeypatch, tmp_path, fake_hub):
    """A mismatch on the last validator stops the build before a single tagger or validator weight is requested."""
    last = list(model_map.MODEL_MAP["validator_models"])[-1]
    monkeypatch.setitem(model_map.MODEL_MAP["validator_models"][last], "accuracy_map_sha256", "0" * 64)
    calls = []
    with pytest.raises(RuntimeError, match=f"validator_models {last} .* not MODEL_MAP's"):
        _run_download(monkeypatch, tmp_path, fake_hub, calls)
    assert len(calls) == len(main.VALIDATOR_LABEL_TYPES)
    assert all(c[0] == "snapshot_download" and c[2] == [model_map.HUB_ACCURACY_MAP] for c in calls)


def test_live_hub_check_script(fake_hub, capsys):
    """regression/check_hub_maps.py (run by hand against the real hub) passes on the committed set and names a
    validator whose hub map is not what MODEL_MAP, the map file and the record say; here against the stand-in hub."""
    with pytest.raises(SystemExit) as done:
        runpy.run_path("regression/check_hub_maps.py", run_name="__main__")
    assert done.value.code == 0 and capsys.readouterr().out.count("ok ") == len(main.VALIDATOR_LABEL_TYPES)
    fake_hub.maps[_hub_key("obstacle")] = b"{}"                      # the hub holds another map than all three say
    with pytest.raises(SystemExit) as done:
        runpy.run_path("regression/check_hub_maps.py", run_name="__main__")
    assert done.value.code == 1 and "FAIL obstacle" in capsys.readouterr().err


def test_load_models_runs_the_map_check_first(monkeypatch):
    """load_models() calls check_accuracy_maps() before importing any model code, on the real startup path."""
    calls = []

    class Stop(Exception):
        pass

    def record():
        calls.append("checked")
        raise Stop()                                                 # so no model module is imported afterwards

    monkeypatch.setattr(main, "check_accuracy_maps", record)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)
    with pytest.raises(Stop):
        main.load_models()
    assert calls == ["checked"]
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("SIDEWALK_ALLOW_CPU", raising=False)
    with pytest.raises(SystemExit):                                  # no GPU: exits before the check, as documented
        main.load_models()
    assert calls == ["checked"]


def test_webp_round_trip_time_is_counted_in_the_validator_stage(app_with, monkeypatch):
    client, depth, set_pano = app_with
    real = main.webp_round_trip

    def slow(image, quality=95):
        time.sleep(0.08)
        return real(image, quality)

    monkeypatch.setattr(main, "webp_round_trip", slow)
    r = post(client)
    assert r.status_code == 200
    stage = dict(re.findall(r"(\w+);dur=(\d+)", r.headers["Server-Timing"]))
    assert int(stage["validator"]) >= 80


def test_check_accuracy_maps_rejects_a_swapped_or_broken_map(monkeypatch, tmp_path):
    bad = tmp_path / "accuracy_mappings"
    bad.mkdir()
    (bad / "curbramp.json").write_text(json.dumps(main.accuracy_mappings["curbramp"]) + "\n ")  # same map, other bytes
    (bad / "provenance.json").write_text(json.dumps(json.load(open("accuracy_mappings/provenance.json"))))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="not the map MODEL_MAP binds"):
        main.check_accuracy_maps(["curbramp"])
    # the right bytes but a non-monotone table: still refused
    broken = {"correct": {"0.5": 0.9, "1.0": 0.8}, "incorrect": {"0.5": 0.0, "1.0": 0.5}}
    (bad / "curbramp.json").write_text(json.dumps(broken))
    sha = model_map.accuracy_map_sha256(bad / "curbramp.json")
    monkeypatch.setitem(model_map.MODEL_MAP["validator_models"]["curbramp"], "accuracy_map_sha256", sha)
    prov = json.load(open(bad / "provenance.json"))
    prov["maps"]["curbramp"]["sha256"] = sha
    (bad / "provenance.json").write_text(json.dumps(prov))
    monkeypatch.setitem(main.accuracy_mappings, "curbramp", broken)
    with pytest.raises(RuntimeError, match="non-decreasing"):
        main.check_accuracy_maps(["curbramp"])


@pytest.mark.parametrize("label_type", main.VALIDATOR_LABEL_TYPES)
def test_accuracy_map_format(label_type):
    """accuracy_mappings/<type>.json: both verdicts, ascending confidence keys in [0.5, 1], non-decreasing accuracies
    in [0, 1], so the lookup's 'highest threshold <= confidence' walk is sound."""
    m = main.accuracy_mappings[label_type]
    assert set(m) == {"correct", "incorrect"}
    for verdict, table in m.items():
        keys = [float(k) for k in table]
        assert keys == sorted(keys) and keys[0] == 0.5 and keys[-1] == 1.0, verdict
        values = list(table.values())
        assert all(0.0 <= v <= 1.0 for v in values), verdict
        assert all(b >= a for a, b in zip(values, values[1:])), verdict
