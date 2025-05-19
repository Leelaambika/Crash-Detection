import cv2
import numpy as np
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def detect_objects(image, model, conf_thres=0.4):
    # Perform inference
    results = model(image)

    # Process results
    class_ids, scores, boxes = [], [], []
    for result in results:
        for det in result.boxes.data:  # Access detected boxes
            if det[4] > conf_thres:  # Confidence threshold
                x1, y1, x2, y2 = map(int, det[:4])
                boxes.append([x1, y1, x2, y2])
                scores.append(float(det[4]))
                class_ids.append(int(det[5]))

    return class_ids, scores, boxes

# Load the output image
output_image = cv2.imread("testing1.jpg")

# Load the YOLO model
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Get the number of objects detected
class_ids, scores, boxes = detect_objects(output_image, model)

# Draw bounding boxes on the image
for box, class_id, score in zip(boxes, class_ids, scores):
    x1, y1, x2, y2 = box
    color = (0, 255, 0)  # Green color for bounding boxes
    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
    label = f'Class {class_id}: {score:.2f}'
    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save or display the image with bounding boxes
cv2.imwrite("testing1_with_boxes.jpg", output_image)

# Optionally display the image (if you're using a local environment)
cv2.imshow("Detected Objects", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Set a threshold for the number of objects
num_objects = len(class_ids)

if num_objects > 0:
    message = "Accident detected!"
else:
    message = "No accidents detected."

print(message)
