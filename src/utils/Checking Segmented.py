import os
import shutil

label_dir = "../../data/Dataset/test/labels"
segmented_dir = "../../data/Dataset/test/segmented_labels"

# Ensure the target directory exists
os.makedirs(segmented_dir, exist_ok=True)

for file in os.listdir(label_dir):
    if file.endswith(".txt"):  # Assuming YOLO format
        file_path = os.path.join(label_dir, file)

        with open(file_path, "r") as f:
            lines = f.readlines()

        # Check for segmentation format
        for line in lines:
            values = list(map(float, line.strip().split()))
            if len(values) > 5:  # YOLO bounding box has 5 values, segmentation has more
                print(f"Segmented: {file} -> Moving to {segmented_dir}")
                shutil.move(file_path, os.path.join(segmented_dir, file))
                break
