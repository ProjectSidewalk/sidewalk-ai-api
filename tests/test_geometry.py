"""The GPU renderer (sidewalk_ai_api.panorama.render_view_torch) against the shared package and the old renderer.

1. render_view_torch must equal sidewalk_panorama_tools.geometry.render_view (the training pipeline's renderer) to
   one intensity level on high-frequency content, including across the seam and near the poles. A half-pixel slip
   (align_corners=True, or no wrap column) moves a 7-levels-per-pixel ramp by ~3 levels, so this pins the grid
   convention.
2. The old Rodrigues renderer (tests/_old_renderer.py) aims the same camera: after undoing its own pixel mapping,
   each old output pixel's ray lands exactly where the package's view puts it. Its two pixel-level quirks are what
   separate the old and new images, and both are pinned here so the size of the change is on record:
     * it read the source about one row low (``lat = h - lat`` with align_corners=True: 1 + lat / 180 rows), and
     * its output rays span the FOV endpoint-to-endpoint (``linspace``), a focal length n / (n - 1) too short.
   Together they move an output pixel by at most ~2 px in the central half of the view and ~0.8-1.2 px at the label.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sidewalk_panorama_tools.geometry import (label_view, pano_px_to_lonlat, project_lonlat, render_view,
                                              sampling_grid)

from _old_renderer import old_view
from sidewalk_ai_api.panorama import render_view_torch


def render_numpy_grid(pano, view, tilt):
    """The pre-torch-grid path: the numpy sampling_grid with the same normalization, padding and sampling."""
    h, w = pano.shape[:2]
    sx, sy = sampling_grid(view, tilt, w, h)
    img = torch.from_numpy(pano).permute(2, 0, 1).unsqueeze(0).float()
    img = torch.cat((img[..., -1:], img, img[..., :1]), dim=3)
    grid = torch.stack((torch.from_numpy(2.0 * (sx + 1.0) / (w + 2) - 1.0),
                        torch.from_numpy(2.0 * sy / h - 1.0)), dim=-1).unsqueeze(0).float()
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()

# (x_norm, y_norm): centre, both sides of the seam, high and low latitudes.
POINTS = [(0.5, 0.5), (0.001, 0.6), (0.9995, 0.4), (0.25, 0.1), (0.75, 0.9), (0.3, 0.02), (0.6, 0.98)]


def structured_pano(width):
    """A (width/2, width, 3) uint8 equirect with content that changes every pixel: two ramps and a checkerboard."""
    h = width // 2
    yy, xx = np.mgrid[0:h, 0:width]
    return np.stack([(xx * 7) % 256, (yy * 5) % 256, ((xx // 8 + yy // 8) % 2) * 255], -1).astype(np.uint8)


@pytest.fixture(scope="module")
def pano_1024():
    return structured_pano(1024)


@pytest.mark.parametrize("x, y", POINTS)
@pytest.mark.parametrize("pitch", [None, 3.0])
def test_torch_matches_package(pano_1024, x, y, pitch):
    view, tilt = label_view(x, y, 90, 256, camera_pitch=pitch, camera_roll=1.0 if pitch else None, beta=0.95)
    assert view.reduction_factor(1024) == 1
    ours = render_view_torch(pano_1024, view, tilt).astype(int)
    ref = render_view(pano_1024, view, tilt).astype(int)
    assert ours.shape == ref.shape == (256, 256, 3)
    assert np.abs(ours - ref).max() <= 1


@pytest.mark.parametrize("x, y", POINTS)
def test_torch_grid_matches_numpy_grid(pano_1024, x, y):
    """At k = 1 the torch-built grid renders the same image as the numpy grid, except where the float32 grid tips
    a value across a rounding boundary: one level, on ~0.03% of values here (0.2% at 8192 -> 2048)."""
    view, tilt = label_view(x, y, 90, 256, camera_pitch=3.0, camera_roll=1.0, beta=0.95)
    assert view.reduction_factor(1024) == 1
    a = render_view_torch(pano_1024, view, tilt).astype(int)
    b = render_numpy_grid(pano_1024, view, tilt).astype(int)
    assert np.abs(a - b).max() <= 1
    assert (a != b).mean() < 0.005


def test_torch_matches_package_full_size():
    """The production size: an 8192 x 4096 pano and a 2048 view, at the seam."""
    pano = structured_pano(8192)
    view, tilt = label_view(0.9995, 0.55, 90, 2048, camera_pitch=3.0, beta=0.95)
    # The package anti-aliases by box-reducing k-fold first; render_view_torch doesn't, which is only right at k = 1.
    assert view.reduction_factor(8192) == 1
    assert np.abs(render_view_torch(pano, view, tilt).astype(int) - render_view(pano, view, tilt)).max() <= 1


def old_rays_in_new_view(pano, x, y):
    """Runs the old renderer and returns, for each output pixel, where its ray lands in the new view, plus the
    raw sample grid of both, in source pixels."""
    h, w = pano.shape[:2]
    _, osx, osy = old_view(pano, x, y)
    view, _ = label_view(x, y, 90, w // 4)
    # The old renderer's lon/lat before its pixel mapping (inverting it removes the one-row offset).
    lon = np.degrees((osx - 0.5) / ((w - 1) / 2.0) * np.pi - np.pi)
    lat = np.degrees(((h - (osy - 0.5)) - (h - 1) / 2.0) / ((h - 1) / 2.0) * (np.pi / 2.0))
    px, py = project_lonlat(lon, lat, view, None)
    return view, px, py, osx, osy


# Every point at 1024 wide, and two at the production size (the old renderer takes ~2 s a view there on the CPU).
AIM_CASES = [(1024, x, y) for x, y in POINTS] + [(8192, 0.5, 0.5), (8192, 0.001, 0.6)]


@pytest.mark.parametrize("width, x, y", AIM_CASES)
def test_old_renderer_aims_the_same_camera(width, x, y):
    pano = structured_pano(width)
    view, px, py, _, _ = old_rays_in_new_view(pano, x, y)
    n = view.width
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    # Old pixel j has its ray at tan = -t + 2 t j / (n - 1); the new view puts that ray at x = n j / (n - 1).
    ok = np.isfinite(px)
    assert ok.mean() > 0.999
    assert np.abs(px - jj * n / (n - 1))[ok].max() < 0.05
    assert np.abs(py - ii * n / (n - 1))[ok].max() < 0.05


@pytest.mark.parametrize("x, y", POINTS)
def test_old_vs_new_displacement_is_bounded(pano_1024, x, y):
    """The raw old-vs-new difference, as each old output pixel's sample point seen in the new view."""
    h, w = pano_1024.shape[:2]
    _, osx, osy = old_view(pano_1024, x, y)
    view, _ = label_view(x, y, 90, w // 4)
    lon, lat = pano_px_to_lonlat(osx, osy, w, h)
    px, py = project_lonlat(lon, lat, view, None)
    n = view.width
    jj, ii = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    disp = np.hypot(px - jj, py - ii)
    centre = slice(n // 4, 3 * n // 4)
    assert np.nanmax(disp[centre, centre]) <= 2.0
    # At the label itself (the view centre) the change is the old row offset, 1 + lat / 180 source rows (0.5 at the
    # nadir, 1 at the horizon, 1.5 at the zenith): at most ~1.2 output pixels.
    assert disp[n // 2, n // 2] <= 1.25


def test_sampling_grid_wraps_x():
    view, _ = label_view(0.0, 0.5, 90, 64)
    sx, _ = sampling_grid(view, None, 1024, 512)
    assert sx.min() >= 0 and sx.max() < 1024
