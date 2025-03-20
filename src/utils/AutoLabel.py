import csv
import os
import re  # For extracting numbers from filenames

# Define the input folder and output CSV file
input_folder = "../../data/Dataset/Vgg11/orange"  # Replace with your actual folder path
filename = "orange.csv"

# Get all image filenames in the folder
image_files = [
    f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

# Calculate the total number of images dynamically
total_images = len(image_files)


# Function to extract the number following "Banana" and calculate freshness grade
def calculate_freshness(filename, total_images):
    # Extract the number after "Banana" in the filename
    match = re.search(r"orange(\d+)", filename)
    if match:
        image_number = int(match.group(1))  # Get the matched number
        freshness_grade = 10 - (image_number - 1) * (10 / 350)
        return max(0, min(freshness_grade, 10))  # Clamp the value between 0 and 10
    return 0  # Default freshness grade if no number is found


# Check if the CSV file already exists
file_exists = os.path.exists(filename)

# Write to the CSV file
with open(filename, mode="a", newline="") as file:  # "a" mode for append if it exists
    writer = csv.writer(file)

    # Write headers only if the file doesn't exist
    if not file_exists:
        writer.writerow(["image_name", "grading"])

    # Write image names and their corresponding freshness grades
    for image_name in image_files:
        grading = calculate_freshness(image_name, total_images)
        writer.writerow([image_name, grading])

print(
    f"CSV file '{filename}' updated successfully with {len(image_files)} image names and dynamic freshness grades!"
)
