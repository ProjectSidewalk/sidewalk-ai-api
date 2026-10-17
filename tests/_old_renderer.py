"""The renderer this API used before 1.1.0 (sidewalk_ai_api/panorama.py at f400df7), kept verbatim apart from
returning the float source-pixel grid as well, so tests can compare it with the shared-package geometry.

Not used by the API. ``old_view`` is the old pair of ``get_perspective_center_params`` + ``to_perspective_image``.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_perspective_center_params_from_equi_point(label_x, label_y, equi_width, equi_height):
    lon = (label_x / equi_width - 0.5) * 2 * np.pi
    lat = (0.5 - label_y / equi_height) * np.pi
    return np.degrees(lon), np.degrees(lat)


def equirectangular_to_perspective(equi_img, fov, theta, phi, height, width):
    """The old function, unchanged except that it also returns (sx, sy): the continuous source pixel it samples,
    in the centre = i + 0.5 convention (its align_corners=True index plus 0.5)."""
    device = torch.device("cpu")

    img = torch.tensor(equi_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    h, w = equi_img.shape[:2]

    hFOV = float(height) / width * fov
    w_len = torch.tan(torch.deg2rad(torch.tensor(fov / 2.0, device=device)))
    h_len = torch.tan(torch.deg2rad(torch.tensor(hFOV / 2.0, device=device)))

    x_map = torch.ones((height, width), dtype=torch.float32, device=device)
    y_map = torch.linspace(-w_len, w_len, width, device=device).repeat(height, 1)
    z_map = -torch.linspace(-h_len, h_len, height, device=device).unsqueeze(1).repeat(1, width)

    D = torch.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = torch.stack((x_map, y_map, z_map), dim=-1) / D.unsqueeze(-1)

    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    R1, _ = cv2.Rodrigues((z_axis * torch.deg2rad(torch.tensor(theta))).cpu().numpy())
    R2, _ = cv2.Rodrigues((np.dot(R1, y_axis.cpu().numpy()) * -torch.deg2rad(torch.tensor(phi)).item()))

    R1 = torch.tensor(R1, dtype=torch.float32, device=device)
    R2 = torch.tensor(R2, dtype=torch.float32, device=device)

    xyz = xyz.view(-1, 3).T
    xyz = torch.matmul(R1, xyz)
    xyz = torch.matmul(R2, xyz).T
    xyz = xyz.view(height, width, 3)

    lat = torch.asin(xyz[:, :, 2])
    lon = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])

    lon = lon / np.pi * (w - 1) / 2.0 + (w - 1) / 2.0
    lat = lat / (np.pi / 2.0) * (h - 1) / 2.0 + (h - 1) / 2.0
    lat = h - lat
    sx, sy = lon.numpy().astype(np.float64) + 0.5, lat.numpy().astype(np.float64) + 0.5

    lon = (lon / ((w - 1) / 2.0)) - 1
    lat = (lat / ((h - 1) / 2.0)) - 1
    grid = torch.stack((lon, lat), dim=-1).unsqueeze(0)
    persp = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return (persp[0].permute(1, 2, 0) * 255).byte().cpu().numpy(), sx, sy


def old_view(equi_img, x_norm, y_norm):
    """The old /process projection: a 90 degree, width/4 square view centred on the normalized point."""
    h, w = equi_img.shape[:2]
    theta, phi = get_perspective_center_params_from_equi_point(x_norm * w, y_norm * h, w, h)
    return equirectangular_to_perspective(equi_img, 90, theta, phi, w // 4, w // 4)
