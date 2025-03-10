from ultralytics import YOLO
import cv2
import os

# Load trained YOLOv8 model
model = YOLO("../../models/yolo11x.pt")

# Define paths
input_folder = "../../data/newdataset/augbanana"  # Input images folder
output_folder = "../../data/newdataset/New folder"  # Output labels folder
no_banana_folder = (
    "../../data/newdataset/no"  # Folder to save images with no detected bananas
)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(no_banana_folder, exist_ok=True)

# COCO class ID for banana (change if necessary)
BANANA_CLASS_ID = 46

# Process each image
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        file_path = os.path.join(input_folder, filename)

        # Run YOLO detection
        results = model(file_path)

        # Store detected bananas
        detected_bananas = []

        # Loop through detected objects
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # Class ID
                if class_id == BANANA_CLASS_ID:  # Keep only bananas
                    x_center, y_center, width, height = box.xywhn[0]  # Normalized bbox
                    detected_bananas.append(
                        f"0 {x_center} {y_center} {width} {height}\n"
                    )

        # Only create the annotation file if at least one banana is found
        if detected_bananas:
            label_path = os.path.join(output_folder, filename.replace(".png", ".txt"))
            with open(label_path, "w") as f:
                f.writelines(detected_bananas)
            print(f"Saved annotation: {label_path}")
        else:
            # If no bananas are detected, save the image to the no_banana folder
            no_banana_path = os.path.join(no_banana_folder, filename)
            img = cv2.imread(file_path)
            cv2.imwrite(no_banana_path, img)
            print(f"Saved image with no banana: {no_banana_path}")

print(
    "Processing complete! Banana annotations saved and images with no bananas stored."
)
