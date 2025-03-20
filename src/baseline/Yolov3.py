import cv2
import torch
import os
from ultralytics import YOLO

# Load YOLOv8 model (or change to 'yolov3.pt' if using YOLOv3)
model = YOLO("../../models/best.pt")

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# COCO class index for banana
BANANA_CLASS_ID = 2

# Input and output folders
input_folder = "../../data/orangeall"
output_folder = "../../data/orangeallcropped"
os.makedirs(output_folder, exist_ok=True)

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
                if int(cls) == BANANA_CLASS_ID:  # Filter only bananas
                    x1, y1, x2, y2 = map(int, box)

                    # Crop the banana
                    cropped = img[y1:y2, x1:x2]

                    # Save cropped banana image
                    crop_filename = f"{os.path.splitext(filename)[0]}_cropped.png"
                    crop_path = os.path.join(output_folder, crop_filename)
                    cv2.imwrite(crop_path, cropped)
                    print(f"Saved: {crop_path}")

print("Apple detection and cropping complete.")
