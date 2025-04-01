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
from efficientnet_pytorch import EfficientNet

app = Flask(__name__)
CORS(app)


# def load_vgg11_model(label):
#     vgg11 = models.vgg11(pretrained=False)
#     vgg11.classifier = nn.Sequential(
#         nn.Linear(25088, 4069),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(4069, 4069),
#         nn.ReLU(),
#         nn.Linear(4069, 1096),
#         nn.ReLU(),
#         nn.Dropout(0.3),
#         nn.Linear(1096, 296),
#         nn.ReLU(),
#         nn.Linear(296, 56),
#         nn.ReLU(),
#         nn.Linear(56, 1),  # Single output for regression (freshness grade)
#     )
#     model_path = f"../models/{label}_vgg11_freshness_model.pth"
#     vgg11.load_state_dict(torch.load(model_path))
#     vgg11.eval()
#     return vgg11.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# def predict_vgg11(img_path, label):
#     model = load_vgg11_model(label)
#     transform = transforms.Compose(
#         [
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )
#     image = Image.open(img_path).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
#     with torch.no_grad():
#         return model(input_tensor).squeeze().item()


# class MultiTaskEfficientNet(nn.Module):
#     def __init__(self, num_classes=3):
#         super(MultiTaskEfficientNet, self).__init__()
#         self.efficientnet = EfficientNet.from_pretrained("efficientnet-b3")
#         in_features = self.efficientnet._fc.in_features
#         self.efficientnet._fc = nn.Identity()

#         # Classification head (Fruit Type)
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes),
#         )

#         # Regression head (Freshness Grading)
#         self.regressor = nn.Sequential(
#             nn.Linear(in_features, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1),
#         )

#     def forward(self, x):
#         shared_features = self.efficientnet(x)
#         fruit_output = self.classifier(shared_features)
#         freshness_output = self.regressor(shared_features)
#         return fruit_output, freshness_output


# def load_efficientnet():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MultiTaskEfficientNet()
#     model.load_state_dict(
#         torch.load("../models/EfficientNetB3_Fruit_Grading.pth", map_location=device)
#     )
#     model.eval()
#     return model.to(device)


# def predict_efficientnet(img_path):
#     model = load_efficientnet()
#     transform = transforms.Compose(
#         [
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )
#     image = Image.open(img_path).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
#     with torch.no_grad():
#         fruit_output, freshness_output = model(input_tensor)
#     label_idx = torch.argmax(fruit_output).item()
#     freshness_score = freshness_output.item()
#     labels = ["apple", "banana", "orange"]

#     return labels[label_idx], freshness_score


def get_yolo_preds(img_path):
    model = YOLO("../models/yolov5.pt")
    results = model(img_path)[0]
    image = cv2.imread(img_path)
    detected_fruits = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = model.names[class_id]
        if label.lower() in [
            "freshapples",
            "freshbanana",
            "freshoranges",
            "rottenapples",
            "rottenbanana",
            "rottenoranges",
        ]:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            # cropped_fruit = image[y1:y2, x1:x2]
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            #     cropped_fruit_path = temp_img.name
            #     cv2.imwrite(cropped_fruit_path, cropped_fruit)

            # # vgg_grade = predict_vgg11(cropped_fruit_path, label.lower())
            # # eff_label, eff_grade = predict_efficientnet(cropped_fruit_path)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(
            #     image,
            #     f"{label}: VGG={vgg_grade:.2f}, Eff={eff_grade:.2f}",
            #     (x1, y1 - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     2,
            # )
            # print(f"VGG Grade for {label}: {vgg_grade}")
            # print(f"EfficientNet Label: {eff_label}, Grade: {eff_grade}")

            detected_fruits.append(
                {
                    "yolo_label": label,
                    "confidence": confidence,
                    # "vgg_grade": vgg_grade,
                    # "efficientnet_label": eff_label,
                    # "efficientnet_grade": eff_grade,
                }
            )
    _, buffer = cv2.imencode(".jpg", image)
    return {
        "results": detected_fruits,
        "image": base64.b64encode(buffer).decode("utf-8"),
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
    results = get_yolo_preds(image_path)
    os.remove(image_path)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
