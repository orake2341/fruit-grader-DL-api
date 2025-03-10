import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# Load dataset CSV
df = pd.read_csv("../../data/appleall/apple_freshness.csv")

# Define source image folder (where original images are stored)
image_folder = "../../data/ap"

# Create dataset folders for train and validation
output_folder = "../../data/Dataset"
train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "val")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# We don't need binning for regression - keep the grading as a continuous float
# The grading values are used directly as the regression target

# Split dataset into 90% train and 10% validation (no stratification needed for regression)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Save train and validation CSV files
train_csv = os.path.join(output_folder, "train_split.csv")
val_csv = os.path.join(output_folder, "val_split.csv")
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)


# Function to move images to respective folders
def move_images(df, destination_folder):
    for _, row in df.iterrows():
        filename = row["image_name"]
        src_path = os.path.join(image_folder, filename)
        dest_path = os.path.join(destination_folder, filename)

        # Ensure the source image exists before moving
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        else:
            print(f"Warning: {filename} not found!")


# Shuffle the data to ensure random distribution
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Move images to train and validation folders
move_images(train_df, train_folder)
move_images(val_df, val_folder)

# Output the split results
print(
    f"Dataset split completed!\nTrain: {len(train_df)} samples\nValidation: {len(val_df)} samples"
)
