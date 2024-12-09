from ultralytics import YOLO
import cv2

# Load the YOLOv8 model using your custom weights
model = YOLO('best.pt')  # Replace with your weights file

# Open the webcam (use 0 for the default webcam, or replace with the correct device ID)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process the webcam feed
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the frame
    results = model.predict(frame, conf=0.5)  # Adjust confidence threshold as needed

    # Draw the results on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with detection results

    # Display the annotated frame
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
