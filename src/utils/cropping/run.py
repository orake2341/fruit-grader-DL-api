import cv2
import os
import numpy as np


def get_yolo_preds(
    net,
    input_img_path,
    output_img_path=None,
    confidence_threshold=0.5,
    overlapping_threshold=0.3,
    labels=None,
    show_display=True,
):
    print(f"Processing: {input_img_path}")

    # Ensure output directory exists if the output_img_path is provided
    if output_img_path:
        output_dir = os.path.dirname(output_img_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    cropped_output_dir = "../../../results/cropped"
    if not os.path.exists(cropped_output_dir):
        os.makedirs(cropped_output_dir)

    # Get the output layer names of the YOLO network
    ln = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    ln = [ln[i - 1] for i in unconnected_out_layers.flatten()]  # Fixing the layer index

    # Read the input image
    image = cv2.imread(input_img_path)
    if image is None:
        print(f"Error: Unable to load image {input_img_path}")
        return

    print(f"Image shape: {image.shape}")

    (H, W) = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialize lists for detections
    boxes = []
    confidences = []
    classIDs = []

    # Loop through each output layer
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Only keep detections for "banana, apple, orange"
            if confidence > confidence_threshold and classID in (46, 47, 49):
                # Scale bounding boxes back to image dimensions
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Perform non-maxima suppression to reduce overlapping boxes
    bboxes = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, overlapping_threshold
    )
    image_for_cropping = image.copy()
    if len(bboxes) > 0:
        for i in bboxes.flatten():
            (x, y, w, h) = boxes[i]

            # Ensure the cropped region is within bounds
            y_end = min(y + h, image_for_cropping.shape[0])
            x_end = min(x + w, image_for_cropping.shape[1])

            cropped_fruit = image_for_cropping[y:y_end, x:x_end]

            # Resize cropped fruit to 224x224
            cropped_fruit_resized = cv2.resize(cropped_fruit, (224, 224))
            base_filename = os.path.splitext(os.path.basename(input_img_path))[0]
            cropped_filename = os.path.join(
                cropped_output_dir,
                f"{base_filename}_cropped.jpg",
            )
            cv2.imwrite(cropped_filename, cropped_fruit_resized)


# YOLO model paths
yolo_path = "../../../models/yolov3.cfg"
weights = "../../../models/yolov3.weights"

# Directory containing augmented images
augmented_images_dir = "../../../data/Augmented Dataset"
output_dir = "../../../results"  # Directory to save output images with bounding boxes
cropped_output_dir = "../../../results/cropped"  # Directory to save cropped images

cuda = True

# Parameters
confidence_threshold = 0.5
overlapping_threshold = 0.5

# Load the YOLO model
net = cv2.dnn.readNetFromDarknet(yolo_path, weights)

if cuda:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Run predictions for all images in the augmented directory
if __name__ == "__main__":
    if not os.path.exists(augmented_images_dir):
        print(f"Error: Augmented images directory '{augmented_images_dir}' not found.")
        exit()

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cropped_output_dir, exist_ok=True)

    # Process each image in the augmented directory
    for image_file in sorted(os.listdir(augmented_images_dir)):
        input_img_path = os.path.join(augmented_images_dir, image_file)
        if image_file.startswith("."):
            print(f"Skipping hidden file: {image_file}")
            continue  # Skip hidden files
        if not os.path.isfile(input_img_path):
            print(f"Skipping directory: {image_file}")
            continue  # Skip directories
        if not input_img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Skipping non-image file: {image_file}")
            continue  # Skip non-image files

        output_img_path = os.path.join(output_dir, f"detected_{image_file}")
        get_yolo_preds(
            net=net,
            input_img_path=input_img_path,
            output_img_path=output_img_path,
            confidence_threshold=confidence_threshold,
            overlapping_threshold=overlapping_threshold,
        )


# # YOLO model paths
# yolo_path = "../../../models/yolov3.cfg"
# weights = "../../../models/yolov3.weights"

# # Directories
# output_dir = "../../../results"  # Directory to save output images with bounding boxes
# cropped_output_dir = "../../../results/cropped"  # Directory to save cropped images

# # CUDA settings
# cuda = True

# # Parameters
# confidence_threshold = 0.5
# overlapping_threshold = 0.5

# # Load the YOLO model
# net = cv2.dnn.readNetFromDarknet(yolo_path, weights)

# if cuda:
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# # Run predictions for a specific image
# if __name__ == "__main__":
#     input_img_path = (
#         "../../../data/Augmented Dataset/Banana612_aug5.jpg"  # Specify your image here
#     )

#     if not os.path.isfile(input_img_path):
#         print(f"Error: Image file '{input_img_path}' not found.")
#         exit()

#     # Ensure output directories exist
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(cropped_output_dir, exist_ok=True)

#     # Output path for the result image
#     output_img_path = os.path.join(
#         output_dir, f"detected_{os.path.basename(input_img_path)}"
#     )

#     # Run YOLO prediction
#     get_yolo_preds(
#         net=net,
#         input_img_path=input_img_path,
#         output_img_path=output_img_path,
#         confidence_threshold=confidence_threshold,
#         overlapping_threshold=overlapping_threshold,
#     )
