from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
import ipykernel

app = Flask(__name__)


def predict_image(img_path):
    # Define the model structure
    vgg11 = models.vgg11(pretrained=False)  # Don't load pretrained weights again
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

    # Define preprocessing transformations (same as during training)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess the image
    def preprocess_image(image_path):
        try:
            image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
        except FileNotFoundError:
            print(f"Error: Image {image_path} not found.")
            return None
        return transform(image).unsqueeze(0)  # Add batch dimension

    input_tensor = preprocess_image(img_path).to(device)

    # Perform prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        output = vgg11(input_tensor).squeeze().item()  # Get the single output value

    return output


def get_yolo_preds():
    """
    Run YOLO predictions, save cropped fruits, classify them, and label the entire image.
    """
    labels_path = "../models/coco.txt"
    yolo_cfg = "../models/yolov3.cfg"
    yolo_weights = "../models/yolov3.weights"
    input_img_path = "../data/Augmented Dataset/Banana10_aug1.jpg"
    output_img_path = "../results/output.jpg"
    confidence_threshold = 0.5
    overlapping_threshold = 0.3

    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} does not exist.")

    # Load COCO labels
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = f.read().strip().split("\n")
    except FileNotFoundError:
        print(f"Error: {labels_path} not found. Please check the file path.")
        return

    # Initialize YOLO model
    net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Read input image
    image = cv2.imread(input_img_path)
    if image is None:
        print(f"Error: Unable to load image {input_img_path}")
        return

    # Resize the image to 412x412 (stretching)
    image_resized = cv2.resize(image, (416, 416))  # Resize to 412x412

    # Now use the resized image for further processing
    (H, W) = image_resized.shape[:2]

    # Create a blob and perform inference
    blob = cv2.dnn.blobFromImage(
        image_resized, 1 / 255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    layer_outputs = net.forward(
        [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    )

    # Initialize lists for detections
    boxes, confidences, classIDs = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Only keep detections for "banana, apple, orange"
            if confidence > confidence_threshold and classID in (46, 47, 49):
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Non-maxima suppression
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, overlapping_threshold
    )

    # Check if any objects are detected
    if len(indices) == 0:
        print("No objects detected!")
        return  # Exit the function early if no detections
    else:
        print(f"{len(indices)} objects detected.")

    image_with_boxes = image_resized.copy()

    # Collect cropped fruit, classify them, and grade them
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]

            # Ensure the cropped region is within bounds
            y_end = min(y + h, image_resized.shape[0])
            x_end = min(x + w, image_resized.shape[1])

            cropped_fruit = image_resized[y:y_end, x:x_end]

            # Check if the cropped fruit is valid (non-empty)
            if cropped_fruit.size == 0:
                print(f"Skipping empty cropped fruit {i + 1}")
                continue

            cropped_fruit_path = f"../results/cropped/fruit_{i + 1}.jpg"

            # Save the cropped fruit if valid
            if cv2.imwrite(cropped_fruit_path, cropped_fruit):
                print(f"Cropped fruit {i + 1} Saved: {cropped_fruit_path}")
            else:
                print(f"Error saving cropped fruit {i + 1}")

            # Grade the Fruit
            grade = predict_image(cropped_fruit_path)
            print(f"Fruit {i + 1} Grade as: {grade:.2f}")

            # Annotate bounding box and classification in the image
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = (0, 0, 255)
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[classIDs[i]]}: {grade:.2f}"
            cv2.putText(
                image_with_boxes,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Save the output image with bounding boxes
        if cv2.imwrite(output_img_path, image_with_boxes):
            print(f"Image with bounding boxes saved to {output_img_path}")

        # Final classification of the image
        if labels[classIDs[i]] == "apple":
            print("The image is apple")
        elif labels[classIDs[i]] == "banana":
            print("The image is banana")
        else:
            print("The image is orange")


# Run the YOLO predictions
if __name__ == "__main__":
    get_yolo_preds()
