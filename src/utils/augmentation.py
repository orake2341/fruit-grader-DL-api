import os
import cv2
import albumentations as A
import numpy as np

# Define individual augmentations
augmentations = {
    "rotate90": A.RandomRotate90(p=1.0),
    "rotate": A.Rotate(
        limit=(-60, 60), p=1.0, border_mode=cv2.BORDER_REPLICATE, crop_border=False
    ),
    "brightness": A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0, p=1.0
    ),
    "contrast": A.RandomBrightnessContrast(
        brightness_limit=0, contrast_limit=0.2, p=1.0
    ),
    "erosion": A.Emboss(p=1.0),  # Simulates erosion effect
    "remove_regions": A.CoarseDropout(max_holes=10, max_height=50, max_width=50, p=1.0),
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

            # # Create and save mirrored version
            # mirrored_image = cv2.flip(image, 1)
            # cv2.imwrite(
            #     os.path.join(output_folder, f"{base_name}_mirrored.png"), mirrored_image
            # )

            # Apply each augmentation separately to original and mirrored
            for aug_name, aug in augmentations.items():
                augmented = aug(image=image)["image"]
                output_filename = f"{base_name}_{aug_name}.png"
                cv2.imwrite(os.path.join(output_folder, output_filename), augmented)
                print(f"Saved: {output_filename}")

                # # Apply augmentation to mirrored image
                # mirrored_augmented = aug(image=mirrored_image)["image"]
                # mirrored_output_filename = f"{base_name}_mirrored_{aug_name}.png"
                # cv2.imwrite(
                #     os.path.join(output_folder, mirrored_output_filename),
                #     mirrored_augmented,
                # )
                # print(f"Saved: {mirrored_output_filename}")


# Input and output folder paths
input_folder = "../../data/Orange"
output_folder = "../../data/AugmentedOrange"

process_folder(input_folder, output_folder)
