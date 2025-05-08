import os
import tempfile
import numpy as np
import cv2
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from keras_core.models import load_model
import matplotlib
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from layers import CBAMBlock
from keras_core.models import Model

app = Flask(__name__)
CORS(app)

# Load both models
base_model = load_model("../models/base_mtl_run1.keras")
cbam_model = load_model("../models/cbam_mtl_run1.keras")


layer_name1 = "separable_conv2d_1"
layer_name2 = "separable_conv2d_3"


def extract_feature_maps(model, image, layer_name, n_images=1):
    from keras_core import Input

    try:
        backbone_layer = model.get_layer("sequential")
    except ValueError:
        try:
            backbone_layer = model.get_layer("functional_15")
        except ValueError:
            raise ValueError("Backbone layer not found")

    inputs = Input(shape=model.input_shape[1:])
    x = inputs

    def apply_layers(layers, x, target_layer_name):
        for layer in layers:
            try:
                x_out = layer(x, training=False)
            except Exception as e:
                print(f"Skipping layer {layer.name} due to error: {e}")
                x_out = x

            if layer.name == target_layer_name:
                return x_out, True

            if isinstance(layer, Model):
                x_out, found = apply_layers(layer.layers, x_out, target_layer_name)
                if found:
                    return x_out, True

            x = x_out
        return x, False

    x, _ = apply_layers(backbone_layer.layers, x, layer_name)
    feature_model = Model(inputs=inputs, outputs=x)
    feature_model.summary()
    feature_maps = feature_model.predict(image[:n_images])
    return feature_maps


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
        img, original_img = preprocess_image(image_path)
        print(original_img.shape)
        print(f"Original image dtype: {original_img.dtype}")
        print(f"Image min value: {original_img.min()}")
        print(f"Image max value: {original_img.max()}")
        start_cbam = time.time()
        cbam_predictions = predict_with_model(cbam_model, img)
        end_cbam = time.time()
        cbam_time = round(end_cbam - start_cbam, 4)

        start_base = time.time()
        base_predictions = predict_with_model(base_model, img)
        end_base = time.time()
        base_time = round(end_base - start_base, 4)

        # Timing for CBAM model

        results = {
            "base_net_prediction": base_predictions,
            "cbam_net_prediction": cbam_predictions,
            "inference_time_sec": {"base_model": base_time, "cbam_model": cbam_time},
        }
    finally:
        os.remove(image_path)

    return jsonify(results)


@app.route("/feature-maps/base", methods=["POST"])
def feature_maps_base():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image_path = temp_file.name
        image_file.save(image_path)

    try:
        img_array, _ = preprocess_image(image_path)
        base_maps = extract_feature_maps(
            base_model, img_array, layer_name="separable_conv2d_1", n_images=1
        )

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle("Base Model Feature Maps")
        for i in range(16):
            axes[i // 4, i % 4].imshow(base_maps[0, :, :, i], cmap="viridis")
            axes[i // 4, i % 4].axis("off")

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        return send_file(
            buffer, mimetype="image/png", download_name="base_feature_maps.png"
        )

    finally:
        os.remove(image_path)


@app.route("/feature-maps/cbam", methods=["POST"])
def feature_maps_cbam():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image_path = temp_file.name
        image_file.save(image_path)

    try:
        img_array, _ = preprocess_image(image_path)
        cbam_maps = extract_feature_maps(
            cbam_model, img_array, layer_name="separable_conv2d_3", n_images=1
        )

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle("CBAM Model Feature Maps")
        for i in range(16):
            axes[i // 4, i % 4].imshow(cbam_maps[0, :, :, i], cmap="viridis")
            axes[i // 4, i % 4].axis("off")

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        return send_file(
            buffer, mimetype="image/png", download_name="cbam_feature_maps.png"
        )

    finally:
        os.remove(image_path)


if __name__ == "__main__":
    app.run(debug=True)
