import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Load a sample image (replace this with your image)
image_path = (
    "../../data/Dataset/FreshApple/IMG-20240612-WA0025.jpg"  # Use your image path here
)
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
img = cv2.resize(img, (100, 100))  # Resize if necessary for consistency
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Create a folder to save the augmented images
output_folder = "augmented_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Generate and save 20 augmented images
aug_iter = datagen.flow(
    img, batch_size=5, save_to_dir=output_folder, save_prefix="aug", save_format="jpeg"
)
