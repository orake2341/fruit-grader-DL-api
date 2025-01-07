from albumentations import Rotate, RandomScale, RandomBrightnessContrast, Compose
import cv2
import os

# Define augmentation pipeline
augment = Compose(
    [
        RandomScale(scale_limit=(0.8, 1.2), p=1.0),
        Rotate(limit=15, p=1.0),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    ]
)


def augment_images_sequentially(
    input_dir, output_dir, file_prefix, start_idx, end_idx, num_augments=5
):
    """
    Augments images with sequential filenames and saves them.

    Parameters:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save augmented images.
        file_prefix (str): Common prefix for filenames (e.g., 'banana').
        start_idx (int): Starting index for filenames.
        end_idx (int): Ending index for filenames.
        num_augments (int): Number of augmentations per image.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(start_idx, end_idx + 1):
        input_file = os.path.join(input_dir, f"{file_prefix}{i}.png")

        # Check if file exists
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue

        # Read the image
        image = cv2.imread(input_file)

        for j in range(num_augments):
            # Apply augmentation
            augmented = augment(image=image)
            augmented_image = augmented["image"]

            # Save augmented image
            output_file = os.path.join(output_dir, f"{file_prefix}{i}_aug{j + 1}.jpg")
            cv2.imwrite(output_file, augmented_image)


if __name__ == "__main__":
    # Define paths and parameters
    input_directory = "../../data/Banana/"  # Replace with your input images directory
    output_directory = (
        "../../data/Augmented Dataset"  # Replace with your output images directory
    )
    file_prefix = "Banana"  # Common prefix for filenames (e.g., 'banana')
    start_index = 1  # Starting index of image filenames
    end_index = 612  # Ending index of image filenames
    num_augmentations = 5  # Number of augmentations per image

    # Call the function to augment images
    augment_images_sequentially(
        input_dir=input_directory,
        output_dir=output_directory,
        file_prefix=file_prefix,
        start_idx=start_index,
        end_idx=end_index,
        num_augments=num_augmentations,
    )
