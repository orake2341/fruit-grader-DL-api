import cv2
import torch
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("../../models/yolov5.pt")

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define class names
CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

# Input and output folders
input_folder = "../../data/Dataset/train/images"
output_folder = "../../data/VGG16/train"
os.makedirs(output_folder, exist_ok=True)

# Ensure class-specific directories exist
for class_name in CLASS_NAMES:
    class_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)

        # Perform detection on GPU
        results = model(image_path, device=device)

        # Loop through detections
        for i, r in enumerate(results):
            for j, (box, cls) in enumerate(zip(r.boxes.xyxy, r.boxes.cls)):
                class_id = int(cls)
                if 0 <= class_id < len(CLASS_NAMES):  # Ensure valid class ID
                    x1, y1, x2, y2 = map(int, box)

                    # Crop the detected object
                    cropped = img[y1:y2, x1:x2]

                    # Save cropped image to class-specific folder
                    class_name = CLASS_NAMES[class_id]
                    class_folder = os.path.join(output_folder, class_name)
                    crop_filename = f"{os.path.splitext(filename)[0]}_{j}.png"
                    crop_path = os.path.join(class_folder, crop_filename)
                    cv2.imwrite(crop_path, cropped)
                    print(f"Saved: {crop_path}")

print("Object detection and cropping complete.")
