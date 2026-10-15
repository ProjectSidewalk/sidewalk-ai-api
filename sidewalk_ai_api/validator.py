from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F
from sidewalk_ai_api.model_map import MODEL_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImageValidator:
    def __init__(self, label_type: str):
        repo = f"projectsidewalk/sidewalk-validator-ai-{label_type}"
        # Pin the repo commit so every instance serves the same weights, even if the repo is updated.
        revision = MODEL_MAP["validator_models"][label_type]["revision"]
        # The PIL backend matches the preprocessing of transformers 4.x; the torchvision default resizes differently.
        self.image_processor = AutoImageProcessor.from_pretrained(repo, revision=revision, backend="pil")
        self.model = AutoModelForImageClassification.from_pretrained(repo, revision=revision).to(DEVICE)

    def validate(self, image):
        inputs = self.image_processor(image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_label = logits.argmax(-1).item()
        predicted_label_name = self.model.config.id2label[predicted_label]
        confidence = probabilities[0, predicted_label].item()
        return predicted_label_name, confidence
