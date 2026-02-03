import os
from flask import Flask, request, jsonify, json
from dotenv import load_dotenv
from datetime import datetime, timezone
from inference.image_handler import load_and_preprocess_image
from inference.predictor import predict_image

load_dotenv()

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image not provided"}), 400

    try:
        image_tensor = load_and_preprocess_image(
            request.files["image"]
        )
        result = predict_image(image_tensor)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/model", methods=["GET"])
def model():
    trained_at = os.getenv("TRAINED_AT_MODEL")

    return jsonify({
        "id": os.getenv("NAME_MODEL", "unknown"),
        "framework": os.getenv("FRAMEWORK_MODEL", "unknown"),
        "trained_at": trained_at or datetime.now(timezone.utc).isoformat()
    }), 200

@app.route("/model/evaluation", methods=["GET"])
def evaluation():
    with open("model/metrics.json") as f:
        metrics = json.load(f)
    
    last_modified_ts = os.path.getmtime("model/metrics.json")
    evaluation_date = datetime.fromtimestamp(last_modified_ts, tz=timezone.utc).isoformat()
    metrics["model_id"] = os.getenv("NAME_MODEL", "unknown")
    metrics["dataset_name"] = os.getenv("DATASET_NAME_MODEL", "unknown"),
    metrics["evaluation_date"] = evaluation_date
    return jsonify(metrics), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True)