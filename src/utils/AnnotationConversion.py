import os

# Directories
segmented_dir = (
    "../../data/Dataset/test/segmented_labels"  # Folder containing segmentation labels
)
converted_dir = (
    "../../data/Dataset/test/converted_labels"  # Output folder for bounding boxes
)

# Ensure the target directory exists
os.makedirs(converted_dir, exist_ok=True)

for file in os.listdir(segmented_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(segmented_dir, file)
        new_file_path = os.path.join(converted_dir, file)

        with open(file_path, "r") as f:
            lines = f.readlines()

        converted_lines = []

        for line in lines:
            values = list(map(float, line.strip().split()))

            if len(values) > 5:  # Segmented format detected
                class_id = int(values[0])  # First value is the class ID
                points = values[1:]  # The rest are x, y coordinates

                # Extract x and y coordinates separately
                x_coords = points[0::2]
                y_coords = points[1::2]

                # Compute bounding box (xmin, ymin, xmax, ymax)
                xmin, ymin = min(x_coords), min(y_coords)
                xmax, ymax = max(x_coords), max(y_coords)

                # Convert to YOLO format: cx, cy, w, h
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin

                converted_lines.append(f"{class_id} {cx} {cy} {w} {h}\n")
            else:
                converted_lines.append(line)  # Already in bounding box format

        # Save the converted labels
        with open(new_file_path, "w") as f:
            f.writelines(converted_lines)

        print(f"Converted: {file} -> Bounding Box Format")
