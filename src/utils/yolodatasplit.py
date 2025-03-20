import os
import shutil
import glob
import random
from sklearn.model_selection import train_test_split

# Define paths
image_dir = "../../data/Dataset/images"
label_dir = "../../data/Dataset/labels"

output_dirs = {
    "train": {"images": "dataset/images/train", "labels": "Dataset/labels/train"},
    "val": {"images": "dataset/images/val", "labels": "Dataset/labels/val"},
    "test": {"images": "dataset/images/test", "labels": "Dataset/labels/test"},
}

# Create output directories
for key in output_dirs:
    os.makedirs(output_dirs[key]["images"], exist_ok=True)
    os.makedirs(output_dirs[key]["labels"], exist_ok=True)

# Get all image files
image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(
    os.path.join(image_dir, "*.png")
)

# Shuffle dataset
random.seed(42)
random.shuffle(image_files)

# Extract corresponding label files
label_files = [
    os.path.join(label_dir, os.path.splitext(os.path.basename(img))[0] + ".txt")
    for img in image_files
]

# First split: 70% Train, 30% Temp (Val + Test)
train_images, temp_images, train_labels, temp_labels = train_test_split(
    image_files, label_files, test_size=0.3, random_state=42
)

# Second split: 50% of temp goes to validation, 50% to testing (final 15%-15%)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42
)


# Function to move files
def move_files(file_list, target_dir):
    for file in file_list:
        if os.path.exists(file):
            shutil.move(file, target_dir)


# Move images and labels
move_files(train_images, output_dirs["train"]["images"])
move_files(train_labels, output_dirs["train"]["labels"])
move_files(val_images, output_dirs["val"]["images"])
move_files(val_labels, output_dirs["val"]["labels"])
move_files(test_images, output_dirs["test"]["images"])
move_files(test_labels, output_dirs["test"]["labels"])

print("Dataset successfully split")
