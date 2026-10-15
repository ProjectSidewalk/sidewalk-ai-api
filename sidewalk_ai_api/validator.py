from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F
from sidewalk_ai_api.model_map import MODEL_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImageValidator:
    def __init__(self, label_type: str):
        entry = MODEL_MAP["validator_models"][label_type]
        repo = f"projectsidewalk/{entry['model_id']}"
        # Pin the repo commit so every instance serves the same weights, even if the repo is updated.
        revision = entry["revision"]
        # The PIL backend matches the preprocessing of transformers 4.x; the torchvision default resizes differently.
        self.image_processor = AutoImageProcessor.from_pretrained(repo, revision=revision, backend="pil")
        # transformers falls back to torchvision with only a warning if the PIL class can't load.
        assert type(self.image_processor).__name__.endswith("Pil"), type(self.image_processor).__name__
        # transformers 5 loads weights in the checkpoint's saved precision; keep them float32 to match the inputs.
        self.model = AutoModelForImageClassification.from_pretrained(
            repo, revision=revision, dtype=torch.float32).to(DEVICE)

    def validate(self, image):
        inputs = self.image_processor(image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_label = logits.argmax(-1).item()
        predicted_label_name = self.model.config.id2label[predicted_label]
        confidence = probabilities[0, predicted_label].item()
        return predicted_label_name, confidence
