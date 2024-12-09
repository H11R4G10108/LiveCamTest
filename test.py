from ultralytics import YOLO
from super_gradients.training import models
from super_gradients.common.object_names import Models
# Load the YOLOv8 model (change the path to your custom weights)
model = YOLO('best.pt')  # Replace 'yolov8s.pt' with your model weights path
output = model.predict_webcam()