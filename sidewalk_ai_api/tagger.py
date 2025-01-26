import json
import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import transforms
from copy import deepcopy
from dinov2.models.vision_transformer import vit_base

MINIMUM_DETECTION = 0.3

class ResizeAndPad:
    def __init__(self, target_size, multiple):
        self.target_size = target_size
        self.multiple = multiple

    def __call__(self, img):
        img = transforms.Resize(self.target_size)(img)
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple
        img = transforms.Pad(
            (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))(img)
        return img


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, model_size="base", nc=0):
        super().__init__()
        if nc == 0:
            raise ValueError("Number of classes must be greater than 0")

        n_register_tokens = 4
        if model_size == "base":
            model = vit_base(patch_size=14, img_size=526, init_values=1.0, num_register_tokens=n_register_tokens,
                             block_chunks=0)
            self.embedding_size = 768
        else:
            raise ValueError("Unsupported model size: only 'base' is implemented")

        self.transformer = deepcopy(model)
        self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 256), nn.ReLU(), nn.Linear(256, nc))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x


class ImageTagger:
    def __init__(self, label_type: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_type = label_type
        self.model_prefix = "dino"
        self.image_dimension = 256
        self.img_resize_multiple = 14
        self.target_size = (self.image_dimension, self.image_dimension)

        # Load tag mappings
        with open('mappings.json', 'r') as f:
            mappings_data = json.load(f)
            self.tag_names = mappings_data[label_type]
        nc = len(self.tag_names)

        # Initialize model
        self.classifier = DinoVisionTransformerClassifier("base", nc)
        model_path = f"checkpoints/validated-{self.model_prefix}-cls-b-{label_type}-tags-best.pth"
        model_load = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in model_load:
            self.classifier.load_state_dict(model_load['model_state_dict'])
        else:
            self.classifier.load_state_dict(model_load)

        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        self.data_transforms = transforms.Compose([
            ResizeAndPad(self.target_size, self.img_resize_multiple),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_img(self, img):
        img = img.convert('RGB')
        img = self.data_transforms(img)
        return torch.tensor(np.array([img], dtype=np.float32), requires_grad=True)

    def inference(self, img: Image.Image):
        input_tensor = self._load_img(img).to(self.device)

        with torch.no_grad():
            embeddings = self.classifier.transformer(input_tensor)
            x = self.classifier.transformer.norm(embeddings)
            output_tensor = self.classifier.classifier(x)
            probabilities = torch.sigmoid(output_tensor)
            results = dict(zip(self.tag_names, probabilities.tolist()[0]))

        return [tag for tag, prob in results.items() if prob > MINIMUM_DETECTION], results