import os
import cv2
import albumentations as A
import numpy as np

# Define individual augmentations
augmentations = {
    "rotate": A.Rotate(
        limit=30, border_mode=cv2.BORDER_REPLICATE, crop_border=False, p=1.0
    ),
    "dropout": A.CoarseDropout(
        num_holes_range=(1, 10),  # Randomly drop between 1 to 10 regions
        hole_height_range=(20, 50),  # Random height between 20 and 50 pixels
        hole_width_range=(20, 50),  # Random width between 20 and 50 pixels
        p=1.0,
    ),
    "sharpen": A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
    "brightness_contrast": A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=1.0
    ),
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
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_original.png"), image)

            # Apply each augmentation separately to original and mirrored
            for aug_name, aug in augmentations.items():
                augmented = aug(image=image)["image"]
                output_filename = f"{base_name}_{aug_name}.png"
                cv2.imwrite(os.path.join(output_folder, output_filename), augmented)
                print(f"Saved: {output_filename}")


# Input and output folder paths
input_folder = "../../data/Dataset/train"
output_folder = "../../data/Dataset/aug"

process_folder(input_folder, output_folder)
