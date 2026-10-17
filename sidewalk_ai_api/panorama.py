import os
import time
import requests
import numpy as np
from http.cookiejar import DefaultCookiePolicy

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/maps",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# Share one session across tile fetches so connections get reused (opening a new one per tile was ~2x slower).
# Cookies are blocked so requests look the same as before. Idle connections can go dead between downloads, so retry
# once and time out instead of hanging until gunicorn kills the worker.
SESSION = requests.Session()
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=100, max_retries=1))
TILE_TIMEOUT = (5, 10)
SESSION.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))

from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from sidewalk_panorama_tools.geometry import sampling_grid

def render_view_torch(equi_img, view, tilt=None):
    """Renders ``view`` of an equirectangular image on the GPU, with the geometry of the shared package.

    The source pixel of every output pixel comes from ``sidewalk_panorama_tools.geometry.sampling_grid``, the
    same function the training pipeline renders its crops with (``render_view``), so the two views are the
    same up to float32 rounding in ``grid_sample``. Only the bilinear sampling runs in torch.

    Conventions (the package's): source pixel *i*'s centre is at ``i + 0.5`` and x wraps at the seam. With
    ``align_corners=False``, grid_sample maps a normalized ``g`` to the continuous pixel ``(g + 1) / 2 * size``
    with the same centre = i + 0.5 convention, so ``g = 2 * s / size - 1`` and no half-pixel shift is needed.
    grid_sample does not wrap, so the image is padded with one wrapped column on each side (pixel 0 of the
    padded image is column W - 1) and x is shifted by 1; ``padding_mode="border"`` then only clamps y at the
    poles, which is what the package's ``sample_bilinear`` does.

    Args:
        equi_img (numpy.ndarray): (H, W, 3) uint8 equirectangular image (rig frame).
        view (sidewalk_panorama_tools.geometry.View): the camera, from ``label_view``.
        tilt (numpy.ndarray or None): the 3x3 rig-tilt matrix from ``label_view``, or None for no correction.

    Returns:
        numpy.ndarray: (view.height, view.width, 3) uint8 perspective image, rounded like ``render_view``.

    Example::

        view, tilt = label_view(0.5, 0.6, 90, 2048, camera_pitch=3, beta=0.95)
        image = render_view_torch(pano_bgr, view, tilt)  # label at (1024, 1024)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = equi_img.shape[:2]
    sx, sy = sampling_grid(view, tilt, w, h)

    img = torch.from_numpy(np.ascontiguousarray(equi_img)).to(device).permute(2, 0, 1).unsqueeze(0).float()
    img = torch.cat((img[..., -1:], img, img[..., :1]), dim=3)  # wrap one column each side for the seam

    gx = torch.from_numpy(2.0 * (sx + 1.0) / (w + 2) - 1.0)
    gy = torch.from_numpy(2.0 * sy / h - 1.0)
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).to(device=device, dtype=torch.float32)

    persp = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return persp[0].permute(1, 2, 0).round().clamp(0, 255).byte().cpu().numpy()


class Panorama:
    def __init__(self, pano_id, cached_image_path=None):
        self.pano_id = pano_id
        self.panorama_image = None
        self.image_source = None
        self.zoom = None

        start = time.perf_counter()
        source = None

        # Attempt to download a cached image. If none exists, download directly.
        if cached_image_path is not None and os.path.exists(cached_image_path):
            cached = cv2.imread(cached_image_path)
            if cached is not None:
                self.panorama_image = cv2.resize(cached, (8192, 4096), interpolation=cv2.INTER_LINEAR)
                self.image_source = "cache"
                source = f"cache ({cached_image_path})"
            else:
                print(f"[Panorama] {pano_id} cache file unreadable at {cached_image_path}; falling back to Google")

        if self.panorama_image is None:
            self._fetch_panorama()
            if self.panorama_image is not None:
                self.image_source = "download"
            if cached_image_path is not None and source is None:
                source = f"Google (cache miss: {cached_image_path})"
            else:
                source = "Google (no cache configured)"

        elapsed = time.perf_counter() - start
        status = "ok" if self.panorama_image is not None else "FAILED"
        print(f"[Panorama] {pano_id} loaded from {source} in {elapsed:.3f}s [{status}]")

    def _is_black_tile(self, tile):
        if tile is None:
            return True
        tile_array = np.array(tile)
        return np.all(tile_array == 0)

    def _fetch_tile(self, x, y, zoom=4):
        if self.zoom != None:
            zoom = self.zoom

        url = (
            f"https://streetviewpixels-pa.googleapis.com/v1/tile"
            f"?cb_client=maps_sv.tactile&panoid={self.pano_id}"
            f"&x={x}&y={y}&zoom={zoom}"
        )
        try:
            response = SESSION.get(url, headers=HEADERS, timeout=TILE_TIMEOUT)
            if response.status_code == 200:
                tile = Image.open(io.BytesIO(response.content))
                if self.zoom != None or not self._is_black_tile(tile):
                    self.zoom = zoom
                    return x, y, tile
        except Exception as e:
            print(e)

        if self.zoom == None:
            # Try fallback with zoom=3
            fallback_url = (
                f"https://streetviewpixels-pa.googleapis.com/v1/tile"
                f"?cb_client=maps_sv.tactile&panoid={self.pano_id}"
                f"&x={x}&y={y}&zoom=3"
            )
            try:
                response = SESSION.get(fallback_url, headers=HEADERS, timeout=TILE_TIMEOUT)
                if response.status_code == 200:
                    tile = Image.open(io.BytesIO(response.content))
                    self.zoom = 3
                    return x, y, tile
            except Exception as e:
                print(e)

        return x, y, None

    def _find_panorama_dimensions(self):
        tiles_cache = {}
        x, y = 5, 2

        is_first = True

        while True:
            tile = self._fetch_tile(x, y)[2]
            if tile is None:
                return None  # Invalid panorama

            if is_first:
                is_first = False
                if self._is_black_tile(tile):
                    return None  # Invalid panorama

            tiles_cache[(x, y)] = tile

            if self._is_black_tile(tile):
                y = y - 1

                while True:
                    tile = self._fetch_tile(x, y)[2]
                    tiles_cache[(x, y)] = tile

                    if self._is_black_tile(tile):
                        return x - 1, y, tiles_cache

                    x += 1

            x += 1
            y += 1

    def _fetch_remaining_tiles(self, max_x, max_y, existing_tiles):
        tiles_cache = existing_tiles.copy()

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for x in range(max_x + 1):
                for y in range(max_y + 1):
                    if (x, y) not in tiles_cache:
                        futures.append(executor.submit(self._fetch_tile, x, y))

            for future in as_completed(futures):
                x, y, tile = future.result()
                if tile is not None:
                    tiles_cache[(x, y)] = tile

        return tiles_cache

    def _assemble_panorama(self, tiles, max_x, max_y):
        tile_size = list(tiles.values())[0].size[0]
        panorama = Image.new('RGB', (tile_size * (max_x + 1), tile_size * (max_y + 1)))

        for (x, y), tile in tiles.items():
            panorama.paste(tile, (x * tile_size, y * tile_size))

        return panorama

    def _crop(self, image):
        y_nonzero, x_nonzero, _ = np.nonzero(image)
        return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    def _fetch_panorama(self):
        dimension_result = self._find_panorama_dimensions()
        if dimension_result is None:
            self.panorama_image = None
            return

        max_x, max_y, initial_tiles = dimension_result
        full_tiles = self._fetch_remaining_tiles(max_x, max_y, initial_tiles)
        result = cv2.cvtColor(np.array(self._assemble_panorama(full_tiles, max_x, max_y)), cv2.COLOR_RGB2BGR)
        self.panorama_image = cv2.resize(self._crop(result), (8192, 4096),
               interpolation = cv2.INTER_LINEAR)

    def get_equi(self):
        return self.panorama_image

    def to_perspective_image(self, view, tilt=None):
        """
        Renders a perspective view of the panorama (see ``render_view_torch``).

        Args:
            view (sidewalk_panorama_tools.geometry.View): the camera, from ``label_view``.
            tilt (numpy.ndarray or None): the rig-tilt matrix from ``label_view``, or None for no correction.

        Returns:
            numpy.ndarray: Perspective image, or None if the panorama failed to load.
        """
        if self.panorama_image is None:
            return None
        return render_view_torch(self.panorama_image, view, tilt)
