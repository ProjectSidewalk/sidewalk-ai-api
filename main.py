from flask import Flask, request, jsonify
from PIL import Image
from image_classifier import ImageClassifier
import io

app = Flask(__name__)

LABEL_TYPES = ["crosswalk", "curbramp", "obstacle", "surfaceproblem"]
classifiers = {label_type: ImageClassifier(label_type=label_type) for label_type in LABEL_TYPES}

@app.route("/classify", methods=["POST"])
def classify():
    if "label_type" not in request.form:
        return jsonify({"error": "Missing label_type parameter"}), 400
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400
    
    label_type = request.form["label_type"]
    if label_type not in classifiers:
        return jsonify({"error": f"Invalid label_type. Choose from {LABEL_TYPES}"}), 400

    image_file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(image_file.read()))
    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 400

    classifier = classifiers[label_type]
    try:
        result, probabilities = classifier.inference(image)
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

    return jsonify({"label_type": label_type, "result": result, "probabilities": probabilities}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
