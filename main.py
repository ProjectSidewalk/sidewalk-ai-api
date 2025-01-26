from flask import Flask, request, jsonify
from sidewalk_ai_api.tagger import ImageTagger
from sidewalk_ai_api.validator import ImageValidator
from PIL import Image
from sidewalk_ai_api.panorama import Panorama
from sidewalk_ai_api.depthanything import DepthAnythingPredictor
import cv2
import json

app = Flask(__name__)

depthanything = DepthAnythingPredictor()

TAGGER_LABEL_TYPES = ["crosswalk", "curbramp", "obstacle", "surfaceproblem"]
taggers = {label_type: ImageTagger(label_type=label_type) for label_type in TAGGER_LABEL_TYPES}

VALIDATOR_LABEL_TYPES = ["crosswalk", "curbramp", "obstacle", "surfaceproblem", "nocurbramp"]
validators = {label_type: ImageValidator(label_type=label_type) for label_type in VALIDATOR_LABEL_TYPES}
accuracy_mappings = {label_type: json.load(open(f"accuracy_mappings/{label_type}.json")) 
                 for label_type in VALIDATOR_LABEL_TYPES}

@app.route("/process", methods=["POST"])
def process():
    if "label_type" not in request.form:
        return jsonify({"error": "Missing label_type parameter"}), 400
    if "panorama_id" not in request.form:
        return jsonify({"error": "Missing panorama_id parameter"}), 400
    if "x" not in request.form:
        return jsonify({"error": "Missing x parameter"}), 400
    if "y" not in request.form:
        return jsonify({"error": "Missing y parameter"}), 400

    label_x = float(request.form["x"])
    label_y = float(request.form["y"])
    if label_x < 0 or label_x > 1 or label_y < 0 or label_y > 1:
        return jsonify({"error": "x and y must be between 0 and 1. They should be normalized values."}), 400

    label_type = request.form["label_type"]
    if label_type not in VALIDATOR_LABEL_TYPES:
        return jsonify({"error": f"Invalid label_type. Choose from {VALIDATOR_LABEL_TYPES}"}), 400

    panorama = Panorama(request.form["panorama_id"])
    height, width = panorama.panorama_image.shape[:2]
    theta, phi = panorama.get_perspective_center_params(label_x * width, label_y * height)
    perspective_image = panorama.to_perspective_image(90, theta, phi, width // 4, width // 4)

    depth = depthanything.predict_depth(perspective_image)

    center_x, center_y = perspective_image.shape[1] // 2, perspective_image.shape[0] // 2
    img_h, img_w = perspective_image.shape[:2]

    inv_depth = 1 / depth[center_y, center_x]
    crop_size_half = int(inv_depth * 6100)

    start_x = max(0, center_x - crop_size_half)
    start_y = max(0, center_y - crop_size_half)
    end_x = min(img_w, center_x + crop_size_half)
    end_y = min(img_h, center_y + crop_size_half)

    crop_height, crop_width = perspective_image[start_y:end_y, start_x:end_x].shape[:2]
    if crop_width > crop_height:
        resize_width, resize_height = 640, int(640 * crop_height / crop_width)
    else:
        resize_width, resize_height = int(640 * crop_width / crop_height), 640
    perspective_image = cv2.resize(perspective_image[start_y:end_y, start_x:end_x], (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)

    perspective_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB)
    perspective_image = Image.fromarray(perspective_image)

    response = {"label_type": label_type}

    # Perform tagging inference only if label_type is in TAGGER_LABEL_TYPES
    if label_type in TAGGER_LABEL_TYPES:
        classifier = taggers[label_type]
        try:
            result, probabilities = classifier.inference(perspective_image)
            response.update({"tags": result, "tag_scores": probabilities})
        except Exception as e:
            return jsonify({"error": f"Inference error: {str(e)}"}), 500

    # Perform validation if label_type is in VALIDATOR_LABEL_TYPES
    validator = validators[label_type]
    try:
        validation_result, validation_confidence = validator.validate(perspective_image)
        
        # Find the highest mapping that's less than or equal to the confidence
        accuracy = 0
        for threshold, acc in accuracy_mappings[label_type][validation_result].items():
            if float(threshold) <= validation_confidence:
                accuracy = acc
        
        response.update({
            "validation_result": validation_result,
            "validation_score": validation_confidence,
            "validation_estimated_accuracy": accuracy
        })
    except Exception as e:
        return jsonify({"error": f"validation error: {str(e)}"}), 500

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
