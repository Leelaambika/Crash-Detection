# Crash-Detection

Crash Detection using YOLOv8:
This project uses the YOLOv8 object detection model to detect crash-like scenarios from images.A crash detection code can be added into RPI and the datasets are from "best" or find in roboflow
 
-Project Structure-

 
 *detect_crash.py # Main crash detection script
 *requirements.txt # Python dependencies
 *data.yaml # Dataset configuration
 *runs/detect/train/weights/best.pt # Trained YOLOv8 model
 *testing1.jpg # Sample test image
 *README.md # Project documentation

-Requirements-

Install required packages with:
pip install -r requirements.txt

-Running the Code-

Make sure your best.pt YOLO model is in:
runs/detect/train/weights/best.pt

Place your test image in the same folder or update the filename in the code.

_Run detection_

python detect_crash.py
The result will be saved as: "testing1_with_boxes.jpg"

-Output-

If crash objects are detected:  “Accident detected!”
If nothing is detected: Continues 

Notes

This is based on a custom-trained YOLOv8 model for crash detection.
Model was trained for 25 epochs using data.yaml and default augmentation parameters.
