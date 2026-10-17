"""/process through Flask's test client, with the models and the panorama replaced by stand-ins.

Covers the pose fields (camera_pitch, camera_roll, tilt_beta, coord_frame): their validation, the response fields,
and that the tilt moves the view only for coord_frame=level. The rendered view is the image /process hands to the
depth model, which the stub records.
"""

import numpy as np
import pano_pose
import pytest
from sidewalk_panorama_tools.geometry import label_pixel, label_view, tilt_matrix

import main
from sidewalk_ai_api.panorama import Panorama

AUTH = {"Authorization": "Bearer test-key"}
PANO_W, PANO_H = 2048, 1024  # /process renders width // 4 = a 512 px view
VIEW = PANO_W // 4


class StubDepth:
    """Records the perspective image and reports 10 m everywhere (crop half-size 610 px, i.e. the whole view)."""
    def __init__(self):
        self.images = []

    def predict_depth(self, image):
        self.images.append(image)
        return np.full(image.shape[:2], 10.0, dtype=np.float32)


class StubValidator:
    def validate(self, image):
        return "correct", 0.97


class StubTagger:
    def inference(self, image):
        return [], [], {}


@pytest.fixture
def app_with(monkeypatch):
    """Returns (client, depth stub, set_pano) with the models stubbed and a synthetic panorama served from 'cache'."""
    depth = StubDepth()
    models = main.Models(depth=depth,
                         taggers={t: StubTagger() for t in main.TAGGER_LABEL_TYPES},
                         validators={t: StubValidator() for t in main.VALIDATOR_LABEL_TYPES})
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
    ({"camera_pitch": "90.5"}, "camera_pitch must be between"),
    ({"camera_roll": "-91"}, "camera_roll must be between"),
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
