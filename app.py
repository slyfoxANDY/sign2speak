import argparse
import cv2
from ultralytics import YOLO
import torch

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--yolo', default='yolov8n-pose.pt', help='Path to YOLOv8 pose model')
args = parser.parse_args()

# Load YOLOv8 pose model
model = YOLO(args.yolo)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO pose detection
    results = model(frame)

    # Draw the results on frame
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Pose App", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
