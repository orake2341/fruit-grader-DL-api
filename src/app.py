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
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)


# VGG11
def predict_image(img_path, label):
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
    if label == "banana":
        vgg11.load_state_dict(torch.load("../models/banana_vgg11_freshness_model.pth"))
    elif label == "apple":
        vgg11.load_state_dict(torch.load("../models/apple_vgg11_freshness_model.pth"))
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
    Run YOLOv8 predictions, save cropped fruits, classify them, and label the entire image.
    Returns the classification, confidence level, and grade.
    """
    model = YOLO("../models/yolov3.pt")  # Load pre-trained YOLOv8 model
    results = model(img_path)[0]  # Run inference on the image

    image = cv2.imread(img_path)
    detected_fruits = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = model.names[class_id]  # Get class label

        # Filter only fruit-related classes (modify as needed)
        if label.lower() in ["apple", "banana", "orange"]:
            cropped_fruit = image[y1:y2, x1:x2]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                cropped_fruit_path = temp_img.name
                cv2.imwrite(cropped_fruit_path, cropped_fruit)
            grade = predict_image(cropped_fruit_path, label.lower())

            # Draw bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{label}: {grade:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            detected_fruits.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "grade": grade,
                    "path": cropped_fruit_path,
                }
            )

    # Encode processed image to base64
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return {"results": detected_fruits, "image": image_base64}


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
