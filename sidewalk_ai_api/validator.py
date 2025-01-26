from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImageValidator:
    def __init__(self, label_type: str):
        self.image_processor = AutoImageProcessor.from_pretrained(f"johnomeara/sidewalk-validator-ai-{label_type}")
        self.model = AutoModelForImageClassification.from_pretrained(f"johnomeara/sidewalk-validator-ai-{label_type}").to(DEVICE)
    
    def validate(self, image):
        inputs = self.image_processor(image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_label = logits.argmax(-1).item()
        predicted_label_name = self.model.config.id2label[predicted_label]
        confidence = probabilities[0, predicted_label].item()
        return predicted_label_name, confidence