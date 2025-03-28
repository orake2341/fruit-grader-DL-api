import os
import random
import shutil
from collections import defaultdict

# Define paths to dataset folders
DATASET_PATHS = {
    "Train": "../../data/Dataset/train",
    "Test": "../../data/Dataset/test",
    "Valid": "../../data/Dataset/valid",
}

# Define class IDs to undersample and their target count
UNDERSAMPLE_CLASSES = {
    0: 400,  # fresh apple
    1: 500,  # fresh banana
    2: 500,  # fresh orange
    3: 200,  # fresh carrot
    4: 200,  # fresh bellpepper
    5: 400,  # fresh cucumber
    6: 500,  # fresh mango
    7: 400,  # fresh potato
    10: 500,  # rotten carrot
    15: 300,  # rotten potato
}

# Dictionary to store class-wise filenames
class_files = defaultdict(list)

# Collect all label files and sort them by class
for split, path in DATASET_PATHS.items():
    labels_path = os.path.join(path, "labels")
    images_path = os.path.join(path, "images")

    for file in os.listdir(labels_path):
        if file.endswith(".txt"):
            file_path = os.path.join(labels_path, file)
            with open(file_path, "r") as f:
                for line in f:
                    class_id = int(line.split()[0])  # Get class ID
                    class_files[class_id].append(
                        (file_path, images_path, file)
                    )  # Store label & image info

# Perform selective undersampling
undersampled_files = []
for class_id, files in class_files.items():
    target = UNDERSAMPLE_CLASSES.get(
        class_id
    )  # Check if this class needs undersampling
    if target and len(files) > target:
        undersampled_files.extend(random.sample(files, target))  # Undersample
    else:
        undersampled_files.extend(files)  # Keep all if not undersampling

# Define output paths for undersampled dataset
OUTPUT_PATH = "../../data/undersampled"
output_labels = os.path.join(OUTPUT_PATH, "labels")
output_images = os.path.join(OUTPUT_PATH, "images")

os.makedirs(output_labels, exist_ok=True)
os.makedirs(output_images, exist_ok=True)

# Copy the selected label files and corresponding images
for label_file, image_dir, file_name in undersampled_files:
    shutil.copy(label_file, os.path.join(output_labels, file_name))

    # Find and copy the corresponding image
    image_extensions = [".jpg", ".png", ".jpeg"]
    for ext in image_extensions:
        image_path = os.path.join(image_dir, file_name.replace(".txt", ext))
        if os.path.exists(image_path):
            shutil.copy(
                image_path, os.path.join(output_images, file_name.replace(".txt", ext))
            )
            break  # Copy only the first matching image

print(
    f"✅ Selective undersampling completed! The specified classes are now ≤ 700 instances."
)
