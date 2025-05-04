import os
import tempfile
import numpy as np
import cv2
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from keras_core.models import load_model
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from layers import CBAMBlock

app = Flask(__name__)
CORS(app)

# Load both models
base_model = load_model("../models/base_mtl_run1.keras")
cbam_model = load_model("../models/cbam_mtl_run1.keras")
# Fruit class labels
classes = [
    "Apple",
    "Banana",
    "Grape",
    "Guava",
    "Jujube",
    "Orange",
    "Pomegranate",
    "Strawberry",
]

fresh = ["Fresh", "Rotten"]


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.asarray(img, dtype=np.float64)
    return np.expand_dims(img, axis=0), img


def predict_with_model(model, image_array):
    pred_freshness, pred_fruit = model.predict(image_array, verbose=0)
    freshness_score = float(pred_freshness[0][0])
    freshness = fresh[1] if freshness_score > 0.5 else fresh[0]
    freshness_confidence = (
        (1 - freshness_score) if freshness == "Fresh" else freshness_score
    )

    fruit_idx = np.argmax(pred_fruit[0])
    fruit_label = classes[fruit_idx]
    fruit_score = float(pred_fruit[0][fruit_idx])

    return {
        "freshness": freshness,
        "freshness_score": freshness_confidence,
        "fruit": fruit_label,
        "fruit_confidence": fruit_score,
    }


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image_path = temp_file.name
        image_file.save(image_path)

    try:
        img_array, original_img = preprocess_image(image_path)
        print(original_img.shape)
        print(f"Original image dtype: {original_img.dtype}")
        print(f"Image min value: {original_img.min()}")
        print(f"Image max value: {original_img.max()}")
        base_predictions = predict_with_model(base_model, img_array)
        cbam_predictions = predict_with_model(cbam_model, img_array)
        results = {
            "base_net_prediction": base_predictions,
            "cbam_net_prediction": cbam_predictions,
        }
    finally:
        os.remove(image_path)

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
