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
    print(input_img_path)
    print(output_img_path)

    # Ensure output directory exists if the output_img_path is provided
    if output_img_path:
        output_dir = os.path.dirname(output_img_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    cropped_output_dir = "../../../results/cropped"
    if not os.path.exists(cropped_output_dir):
        os.makedirs(cropped_output_dir)

    # Generate random colors for each class
    np.random.seed(123)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # Get the output layer names of the YOLO network
    ln = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    ln = [ln[i - 1] for i in unconnected_out_layers.flatten()]  # Fixing the layer index

    # Read the input image
    image = cv2.imread(input_img_path)
    if image is None:
        print(f"Error: Unable to load image {input_img_path}")
        return

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

    if len(bboxes) > 0:
        image_for_cropping = image.copy()

        for i in bboxes.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Crop the person from the image
            cropped_person = image_for_cropping[y : y + h, x : x + w]
            # Optionally, resize the cropped person to a fixed size (e.g., 224x224)
            cropped_person_resized = cv2.resize(cropped_person, (224, 224))
            base_filename = os.path.splitext(os.path.basename(input_img_path))[0]
            # Save each cropped person image without any bounding box
            cropped_filename = os.path.join(
                cropped_output_dir,
                f"{base_filename}_cropped.jpg",
            )
            cv2.imwrite(cropped_filename, cropped_person_resized)

            # Print details about each detection
            print(
                f"Label: {labels[classIDs[i]]}, Confidence: {confidences[i]:.2f}, BBox: {boxes[i]}"
            )

        for i in bboxes.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[classIDs[i]]}: {confidences[i]:.2f}"
            cv2.putText(
                image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    # Save the full image with person detections (bounding boxes) if a path is provided
    if output_img_path:
        # Check if the output path has a valid extension (e.g., .jpg, .png)
        if not output_img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            output_img_path += ".jpg"  # Add .jpg extension if none is provided

        # Save the image with detections (bounding boxes visible)
        if cv2.imwrite(output_img_path, image):
            print(f"Image saved successfully to {output_img_path}")
        else:
            print("Failed to save the image.")
