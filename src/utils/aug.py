import os
import cv2
import numpy as np
import albumentations as A


# YOLO format: class x_center y_center width height (normalized)
def load_yolo_annotations(txt_path):
    """Load YOLO annotation file."""
    bboxes = []
    labels = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f.readlines():
                values = line.strip().split()
                label = int(values[0])  # Class label
                bbox = list(map(float, values[1:]))  # x_center, y_center, width, height
                labels.append(label)
                bboxes.append(bbox)
    return labels, bboxes


def save_yolo_annotations(txt_path, labels, bboxes):
    """Save updated YOLO annotation file."""
    with open(txt_path, "w") as f:
        for label, bbox in zip(labels, bboxes):
            f.write(f"{int(label)} " + " ".join(map(str, bbox)) + "\n")


# Define basic augmentation sets (excluding shift)
augmentations = {
    "bright": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0),
    "contrast": A.RandomBrightnessContrast(
        brightness_limit=0, contrast_limit=0.2, p=1.0
    ),
    "eroded": A.Emboss(p=1.0),
    "removeregion": A.CoarseDropout(
        num_holes_range=(10, 10),
        hole_height_range=(50, 100),
        hole_width_range=(50, 100),
        fill=0,
        p=1.0,
    ),
    "flip": A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=1.0),
    "rotate": A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_REPLICATE),
}

# Ensure bounding boxes are correctly transformed
bbox_params = A.BboxParams(
    format="yolo",  # YOLO format (x_center, y_center, width, height)
    label_fields=["labels"],
    min_visibility=0.3,  # Discard boxes with less than 30% visibility after transformation
)

# Define the shift transform you want to apply on every augmentation
shift_transform = A.ShiftScaleRotate(
    shift_limit=0.1,
    scale_limit=0,
    rotate_limit=15,
    border_mode=cv2.BORDER_REPLICATE,
    p=1.0,
)


def apply_augmentations(
    image_path, annotation_path, output_images, output_labels, base_filename
):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labels, bboxes = load_yolo_annotations(annotation_path)
    bboxes = (
        np.array(bboxes, dtype=np.float32)
        if bboxes
        else np.empty((0, 4), dtype=np.float32)
    )

    # For each augmentation, apply both the shift_transform and the augmentation transform
    for aug_type, transform in augmentations.items():
        pipeline = A.Compose([shift_transform, transform], bbox_params=bbox_params)
        augmented = pipeline(image=img, bboxes=bboxes.tolist(), labels=labels)
        img_aug = augmented["image"]
        bboxes_aug = augmented["bboxes"]
        labels_aug = augmented["labels"]

        # Save augmented image and labels
        new_filename = f"{base_filename}_{aug_type}.png"
        new_labelname = f"{base_filename}_{aug_type}.txt"

        save_img_path = os.path.join(output_images, new_filename)
        save_label_path = os.path.join(output_labels, new_labelname)

        cv2.imwrite(save_img_path, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))
        save_yolo_annotations(save_label_path, labels_aug, bboxes_aug)

    print(f"Saved augmented images and labels for {base_filename}")


# Directories
input_images = "../../data/newdataset/neworange"
input_labels = "../../data/newdataset/neworangelabel"  # YOLO labels (.txt)
output_images = "../../data/newdataset/augorange"
output_labels = "../../data/newdataset/augorangelabel"
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Process all images
for filename in os.listdir(input_images):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_images, filename)
        label_path = os.path.join(
            input_labels, filename.replace(".jpg", ".txt").replace(".png", ".txt")
        )
        base_filename = os.path.splitext(filename)[0]

        apply_augmentations(
            img_path, label_path, output_images, output_labels, base_filename
        )

print("Augmented dataset with updated annotations saved successfully!")
