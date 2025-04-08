import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

# Main function to split and organize dataset with class subdirectories


def organize_dataset(image_folder, output_folder):
    image_paths = list(
        Path(image_folder).rglob("*.jpg")
    )  # Recursively search all subfolders
    image_classes = [
        p.parent.name for p in image_paths
    ]  # Class is derived from parent folder name

    if not image_paths:
        raise ValueError(f"No .jpg images found in: {image_folder}")

    # Split the dataset into train (80%), valid (10%), test (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths,
        image_classes,
        test_size=0.3,
        stratify=image_classes,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    splits = {
        "train": (X_train, y_train),
        "valid": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for split_name, (paths, labels) in splits.items():
        for path, label in zip(paths, labels):
            class_folder = os.path.join(output_folder, split_name, label)
            os.makedirs(class_folder, exist_ok=True)
            dst_path = os.path.join(class_folder, path.name)
            shutil.copy(str(path), dst_path)

    print(
        "✅ Dataset organized into train, valid, and test folders with class subdirectories."
    )


organize_dataset(
    image_folder="../../data/Dataset", output_folder="../../data/newDataset"
)
