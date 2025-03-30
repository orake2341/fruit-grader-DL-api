import cv2
import os

# Paths
image_name = "Screen-Shot-2018-06-12-at-9_06_33-PM_png.rf.d6d22700f78989b36086cd05dc0bfdee.jpg"  # Change this to the image filename you want to check
image_dir = "../../data/Dataset/valid/images"  # Folder containing images
label_dir = "../../data/Dataset/valid/converted_labels"  # Folder containing YOLO labels
output_dir = (
    "../../data/Dataset/valid/outputs"  # Folder to save images with bounding boxes
)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Construct full paths
image_path = os.path.join(image_dir, image_name)
label_file = os.path.splitext(image_name)[0] + ".txt"
label_path = os.path.join(label_dir, label_file)

# Load image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not open image {image_name}")
    exit()

# Get image dimensions
img_h, img_w, _ = image.shape

# Check if label file exists
if not os.path.exists(label_path):
    print(f"No label found for {image_name}")
else:
    # Read bounding box labels
    with open(label_path, "r") as f:
        labels = f.readlines()

    for line in labels:
        values = list(map(float, line.strip().split()))
        class_id = int(values[0])  # First value is class ID
        cx, cy, w, h = values[1:]  # YOLO format: center_x, center_y, width, height

        # Convert YOLO format (normalized) to pixel coordinates
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)

        # Draw bounding box
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Put label text
        label_text = f"Class {class_id}"
        cv2.putText(
            image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # Save and show image
    output_path = os.path.join(output_dir, f"output_{image_name}")
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")

    # Show the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
