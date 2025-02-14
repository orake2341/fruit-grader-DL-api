import os
import cv2
import albumentations as A
import numpy as np

# Define individual augmentations
augmentations = {
    "rotate90": A.RandomRotate90(p=1.0),
    "rotate": A.Rotate(limit=180, p=1.0, border_mode=cv2.BORDER_CONSTANT),
    "brightness": A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0, p=1.0
    ),
    "contrast": A.RandomBrightnessContrast(
        brightness_limit=0, contrast_limit=0.2, p=1.0
    ),
    "erosion": A.Emboss(p=1.0),  # Simulates erosion effect
    "remove_regions": A.CoarseDropout(max_holes=3, max_height=50, max_width=50, p=1.0),
}


# Function to process folder
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to load {filename}")
                continue

            base_name = os.path.splitext(filename)[0]

            # Save original image
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_original.jpg"), image)

            # Apply each augmentation separately
            for aug_name, aug in augmentations.items():
                augmented = aug(image=image)["image"]
                output_filename = f"{base_name}_{aug_name}.jpg"
                cv2.imwrite(os.path.join(output_folder, output_filename), augmented)
                print(f"Saved: {output_filename}")


# Input and output folder paths
input_folder = "../../data/Apple"
output_folder = "../../data/AugmentedApple"

process_folder(input_folder, output_folder)
