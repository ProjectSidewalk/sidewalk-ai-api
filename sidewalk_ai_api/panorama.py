import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

def equirectangular_to_perspective(equi_img, fov, theta, phi, height, width):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the image for PyTorch
    img = torch.tensor(equi_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    h, w = equi_img.shape[:2]

    # Compute the tangent of horizontal and vertical FOV
    hFOV = float(height) / width * fov
    w_len = torch.tan(torch.deg2rad(torch.tensor(fov / 2.0, device=device)))
    h_len = torch.tan(torch.deg2rad(torch.tensor(hFOV / 2.0, device=device)))

    # Generate normalized 3D coordinates in the perspective image space
    x_map = torch.ones((height, width), dtype=torch.float32, device=device)
    y_map = torch.linspace(-w_len, w_len, width, device=device).repeat(height, 1)
    z_map = -torch.linspace(-h_len, h_len, height, device=device).unsqueeze(1).repeat(1, width)

    # Normalize vectors
    D = torch.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = torch.stack((x_map, y_map, z_map), dim=-1) / D.unsqueeze(-1)

    # Compute rotation matrices
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    R1, _ = cv2.Rodrigues((z_axis * torch.deg2rad(torch.tensor(theta))).cpu().numpy())
    R2, _ = cv2.Rodrigues((np.dot(R1, y_axis.cpu().numpy()) * -torch.deg2rad(torch.tensor(phi)).item()))

    R1 = torch.tensor(R1, dtype=torch.float32, device=device)
    R2 = torch.tensor(R2, dtype=torch.float32, device=device)

    # Rotate xyz coordinates
    xyz = xyz.view(-1, 3).T
    xyz = torch.matmul(R1, xyz)
    xyz = torch.matmul(R2, xyz).T
    xyz = xyz.view(height, width, 3)

    # Convert 3D coordinates to spherical coordinates
    lat = torch.asin(xyz[:, :, 2])
    lon = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])

    # Normalize longitude and latitude to pixel coordinates
    lon = lon / np.pi * (w - 1) / 2.0 + (w - 1) / 2.0
    lat = lat / (np.pi / 2.0) * (h - 1) / 2.0 + (h - 1) / 2.0

    # Flip latitude to correct the upside-down issue
    lat = h - lat

    # Normalize to range [-1, 1] for PyTorch grid sampling
    lon = (lon / ((w - 1) / 2.0)) - 1
    lat = (lat / ((h - 1) / 2.0)) - 1

    grid = torch.stack((lon, lat), dim=-1).unsqueeze(0)

    # Sample perspective image from equirectangular image
    persp = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Convert back to numpy image
    return (persp[0].permute(1, 2, 0) * 255).byte().cpu().numpy()

def get_perspective_center_params_from_equi_point(label_x, label_y, equi_width, equi_height):
    # Convert equirectangular coordinates to spherical coordinates
    lon = (label_x / equi_width - 0.5) * 2 * np.pi
    lat = (0.5 - label_y / equi_height) * np.pi

    # Convert spherical angles to degrees for yaw and pitch
    theta = np.degrees(lon) # Yaw (theta) is longitude
    phi = np.degrees(lat)   # Pitch (phi) is latitude

    return theta, phi


class Panorama:
    def __init__(self, pano_id):
        self.pano_id = pano_id
        self.panorama_image = None
        self._fetch_panorama()

    def _fetch_tile(x, y, zoom=4):
        url = (
            f"https://streetviewpixels-pa.googleapis.com/v1/tile"
            f"?cb_client=maps_sv.tactile&panoid={self.pano_id}"
            f"&x={x}&y={y}&zoom={zoom}"
        )
        try:
            response = requests.get(url)
            if response.status_code == 200:
                tile = Image.open(io.BytesIO(response.content))
                if not _is_black_tile(tile):
                    return x, y, tile
        except Exception:
            return x, y, None

        # Try fallback with zoom=3
        fallback_url = (
            f"https://streetviewpixels-pa.googleapis.com/v1/tile"
            f"?cb_client=maps_sv.tactile&panoid={self.pano_id}"
            f"&x={x}&y={y}&zoom=3"
        )
        try:
            response = requests.get(fallback_url)
            if response.status_code == 200:
                tile = Image.open(io.BytesIO(response.content))
                return x, y, tile
        except Exception:
            pass

        return x, y, None

    def _is_black_tile(self, tile):
        if tile is None:
            return True
        tile_array = np.array(tile)
        return np.all(tile_array == 0)

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
    
    def to_perspective_image(self, fov, theta, phi, height, width):
        """
        Converts the equirectangular panorama image to a perspective image.

        Args:
            fov (float): Field of view in degrees.
            theta (float): Yaw angle in degrees.
            phi (float): Pitch angle in degrees.
            height (int): Height of the perspective image.
            width (int): Width of the perspective image.

        Returns:
            numpy.ndarray: Perspective image.
        """
        if self.panorama_image is None:
            return None
        return equirectangular_to_perspective(self.panorama_image, fov, theta, phi, height, width)
    
    def get_perspective_center_params(self, label_x, label_y):
        """
        Calculates the yaw (theta) and pitch (phi) angles to center a specific equirectangular point
        in a perspective image with a fixed FOV of 90 degrees.

        Args:
            label_x (int): x-coordinate of the point in the equirectangular image.
            label_y (int): y-coordinate of the point in the equirectangular image.

        Returns:
            tuple: Yaw (theta) and pitch (phi) angles in degrees.
        """
        if self.panorama_image is None:
            return None
        equi_height, equi_width = self.panorama_image.shape[:2]
        return get_perspective_center_params_from_equi_point(label_x, label_y, equi_width, equi_height)
