import json
import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import transforms
from copy import deepcopy
from dinov2.models.vision_transformer import vit_base


# Resizes to target_size then pads to the nearest multiple so patch_size=14 divides evenly.
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


# DINOv2 ViT-base backbone with a two-layer classification head (768 → 256 → nc).
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

        # Load class-index → tag name list (includes NULL placeholders; length determines nc).
        with open('mappings.json', 'r') as f:
            mappings_data = json.load(f)
            self.tag_names = mappings_data[label_type]
        nc = len(self.tag_names)

        # Load per-tag present/absent thresholds; tags with both null are skipped in inference output.
        with open('tag_thresholds.json', 'r') as f:
            self.tag_thresholds = json.load(f)[label_type]

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

        # Single forward pass produces a sigmoid score for every class index.
        with torch.no_grad():
            embeddings = self.classifier.transformer(input_tensor)
            x = self.classifier.transformer.norm(embeddings)
            output_tensor = self.classifier.classifier(x)
            confidence_scores = torch.sigmoid(output_tensor)
            results = dict(zip(self.tag_names, confidence_scores.tolist()[0]))

        # Apply per-tag thresholds; tags with both null still appear in scores but not in present/absent lists.
        tags_present = []
        tags_not_present = []
        scores = {tag: results[tag] for tag in self.tag_thresholds if tag in results}
        for tag, thresholds in self.tag_thresholds.items():
            present_threshold = thresholds.get("present")
            absent_threshold = thresholds.get("absent")
            if present_threshold is None and absent_threshold is None:
                continue
            if tag not in results:
                continue
            prob = results[tag]
            if present_threshold is not None and prob >= present_threshold:
                tags_present.append(tag)
            if absent_threshold is not None and prob <= absent_threshold:
                tags_not_present.append(tag)

        return tags_present, tags_not_present, scores