import torch
import sys
import numpy as np

sys.path.append('Depth-Anything-V2/metric_depth')
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

class DepthAnythingPredictor:
    def __init__(self, encoder: str = 'vitb', dataset: str = 'vkitti', max_depth: int = 80):
        """
        Initializes the DepthAnythingPredictor with a model.

        Args:
            encoder (str): Encoder type ('vits', 'vitb', 'vitl').
            dataset (str): Dataset name.
            max_depth (int): Maximum depth value.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = encoder
        self.dataset = dataset
        self.max_depth = max_depth

        model_config = model_configs[self.encoder]
        self.model = DepthAnythingV2(**{**model_config, 'max_depth': self.max_depth})
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{self.dataset}_{self.encoder}.pth', map_location=device))
        self.model.to(device).eval()

    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Takes an OpenCV-style numpy image and outputs the depth map.

        Args:
            image (np.ndarray): Input image in HWC format with values in [0, 255].

        Returns:
            np.ndarray: Depth map as a 2D numpy array.
        """

        return self.model.infer_image(image)