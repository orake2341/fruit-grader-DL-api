import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
image_directory = "../../data/AllImage"  # Images + Labels must be in this folder
output_directory = "../../data/Dataset"

# Define YOLO dataset structure
train_dir = os.path.join(output_directory, "train", "images")
val_dir = os.path.join(output_directory, "val", "images")
test_dir = os.path.join(output_directory, "test", "images")

train_label_dir = os.path.join(output_directory, "train", "labels")
val_label_dir = os.path.join(output_directory, "val", "labels")
test_label_dir = os.path.join(output_directory, "test", "labels")

# Create necessary directories
for folder in [
    train_dir,
    val_dir,
    test_dir,
    train_label_dir,
    val_label_dir,
    test_label_dir,
]:
    os.makedirs(folder, exist_ok=True)

# ✅ Get all image-label pairs with class labels
image_extensions = (".jpg", ".jpeg", ".png")
image_label_pairs = []
labels = []  # Stores class labels

for file in os.listdir(image_directory):
    if file.endswith(image_extensions):
        image_path = os.path.join(image_directory, file)
        label_path = (
            os.path.splitext(image_path)[0] + ".txt"
        )  # Corresponding label file

        if os.path.exists(label_path):  # Only include if the label file exists
            with open(label_path, "r") as f:
                first_line = f.readline().strip()  # Read first line
                if first_line:
                    class_label = int(first_line.split()[0])  # Extract class ID
                    image_label_pairs.append((image_path, label_path))
                    labels.append(class_label)

# ✅ Stratified Split (ensures proportional class distribution)
train_pairs, temp_pairs, train_labels, temp_labels = train_test_split(
    image_label_pairs, labels, test_size=0.3, stratify=labels, random_state=42
)

val_pairs, test_pairs, _, _ = train_test_split(
    temp_pairs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)


# ✅ Function to move images & labels
def move_files(pairs, dest_image_folder, dest_label_folder):
    for image_path, label_path in pairs:
        shutil.copy(
            image_path, os.path.join(dest_image_folder, os.path.basename(image_path))
        )  # Move image
        shutil.copy(
            label_path, os.path.join(dest_label_folder, os.path.basename(label_path))
        )  # Move label


# ✅ Move images & labels into YOLO dataset folders
move_files(train_pairs, train_dir, train_label_dir)
move_files(val_pairs, val_dir, val_label_dir)
move_files(test_pairs, test_dir, test_label_dir)

print(
    "✅ Stratified dataset split completed (fruit types are balanced across train, val, and test)!"
)



