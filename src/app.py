from flask import Flask, jsonify, request
import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
import tempfile
from flask_cors import CORS
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)


def predict_image(img_path):
    # Define the model structure
    vgg11 = models.vgg11(pretrained=False)
    vgg11.classifier = nn.Sequential(
        nn.Linear(25088, 4069),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(4069, 4069),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(4069, 1096),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(1096, 296),
        nn.ReLU(),
        nn.Linear(296, 56),
        nn.ReLU(),
        nn.Linear(56, 1),
    )

    # Load the saved model weights
    vgg11.load_state_dict(torch.load("../models/vgg11_freshness_model.pth"))
    vgg11.eval()  # Set the model to evaluation mode

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg11 = vgg11.to(device)

    # Define preprocessing transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        output = vgg11(input_tensor).squeeze().item()

    return output


def get_yolo_preds(img_path):
    """
    Run YOLO predictions, save cropped fruits, classify them, and label the entire image.
    Returns the classification, confidence level, and grade.
    """
    labels_path = "../models/coco.txt"
    yolo_cfg = "../models/yolov3.cfg"
    yolo_weights = "../models/yolov3.weights"
    confidence_threshold = 0.5
    overlapping_threshold = 0.3

    # Load COCO labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = f.read().strip().split("\n")

    # Initialize YOLO model
    net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Read and preprocess input image
    image = cv2.imread(img_path)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(
        [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    )

    boxes, confidences, classIDs = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold and classID in (46, 47, 49):
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, overlapping_threshold
    )

    if len(indices) == 0:
        return jsonify({"results": [], "message": "No objects detected"})

    results = []

    for i in indices.flatten():
        classID = classIDs[i]
        confidence = confidences[i]
        label = labels[classID]

        # Crop fruit and grade
        (x, y, w, h) = boxes[i]
        cropped_fruit = image[y : y + h, x : x + w]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            cropped_fruit_path = temp_img.name
            cv2.imwrite(cropped_fruit_path, cropped_fruit)
        grade = predict_image(cropped_fruit_path)

        # Draw bounding box on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label}: {grade}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        results.append({"label": label, "confidence": confidence, "grade": grade})

    # Encode image to base64
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return {"results": results, "image": image_base64}


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

    results = get_yolo_preds(image_path)
    os.remove(image_path)  # Clean up the temporary file
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
