import cv2
import numpy as np

# Define paths to the model files
prototxt_path = r'C:\Users\user\Downloads\new\deploy.prototxt'
model_path = r'C:\Users\user\Downloads\new\mobilenet_iter_73000.caffemodel'

try:
    # Load the MobileNet-SSD model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
except cv2.error as e:
    print(f"Error loading the model: {e}")
    exit()

# Define the list of class labels MobileNet-SSD was trained on
CLASSES = ["mobile", "mouse", "pen", "marker", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "keyboard", "sheep",
           "sofa", "train", "tvmonitor"]

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Prepare the frame for MobileNet-SSD
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input to the model
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('MobileNet-SSD Real-time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
