import os
import shutil

# Path to the original dataset and the destination folder
source_folder = "../../data/newdataset/apple/appleframes1280"
destination_folder = "../../data/newdataset/newapple"
os.makedirs(destination_folder, exist_ok=True)

# List all apple images (assuming .jpg format)
apple_images = sorted(
    [
        f
        for f in os.listdir(source_folder)
        if f.startswith("apple") and f.endswith(".png")
    ],
    key=lambda x: int(x.replace("apple", "").replace(".png", "")),
)

# Calculate indices to select 700 evenly spaced images
selected_indices = [round(i * (len(apple_images) / 700)) for i in range(700)]

# Copy and rename selected images while retaining order
for new_index, old_index in enumerate(selected_indices, start=1):
    old_image_name = apple_images[old_index]
    old_image_path = os.path.join(source_folder, old_image_name)
    new_image_name = f"apple{new_index}.png"
    new_image_path = os.path.join(destination_folder, new_image_name)
    shutil.copy(old_image_path, new_image_path)

print(
    f"Reduced dataset created with {len(selected_indices)} images, renamed from apple1 to apple700."
)
