"""/process through Flask's test client, with the models and the panorama replaced by stand-ins.

Covers the pose fields (camera_pitch, camera_roll, tilt_beta, coord_frame): their validation, the response fields,
and that the tilt moves the view only for coord_frame=level. The rendered view is the image /process hands to the
depth model, which the stub records.
"""

import io

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


def test_model_map_entries_are_pinned_or_placeholder():
    """Every entry is a 40-hex commit or the one placeholder (never a branch name); taggers are always pinned."""
    for label_type, entry in model_map.MODEL_MAP["tagger_models"].items():
        model_map.pinned_revision(entry, f"tagger {label_type}")
    for label_type, entry in model_map.MODEL_MAP["validator_models"].items():
        if entry["revision"] != model_map.PLACEHOLDER_REVISION:
            model_map.pinned_revision(entry, f"validator {label_type}")
        assert label_type in main.VALIDATOR_LABEL_TYPES


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
