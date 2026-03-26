import cv2
from ultralytics import YOLO

# Load mode, model is for detection
model = YOLO("defect_model.tflite", task="detect")

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while True:
    # Read a single frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Failed to grab camera frame.")
        break

    # Run the AI on the current frame
    # verbose=False stops it from spamming your terminal with text every millisecond
    results = model(frame, verbose=False)

    # Ultralytics bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Hardware Trigger
    # Check if the AI found any bounding boxes in this frame
    if len(results[0].boxes) > 0:
        # confidence score of the first thing it detected
        confidence = results[0].boxes.conf[0].item()
        
        # more than 60% sure it sees a defect, trigger the "alarm"
        if confidence > 0.60:
            print(f"DEFECT DETECTED! Confidence: {confidence:.2f} -> Triggering Fake LED")
            
            # Draw a big red warning text on the video feed
            cv2.putText(annotated_frame, "DEFECT TRIGGERED!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    # live, annotated video stream
    cv2.imshow("Defect Detector", annotated_frame)

    # 'q' key to stop the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and turn off the webcam
cap.release()
cv2.destroyAllWindows()