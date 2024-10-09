import cv2
from deepface import DeepFace

# Load Haar cascade classifier for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Attempt to open webcam
cap = cv2.VideoCapture(0)  # Start with the default camera

if not cap.isOpened():
    cap = cv2.VideoCapture(1)  # Try the next camera index
if not cap.isOpened():
    cap = cv2.VideoCapture(2)  # Try the next camera index

if not cap.isOpened():
    raise IOError("Cannot open webcam. Please check if the camera is connected or already in use.")

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Analyze the frame using DeepFace
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
    except Exception as e:
        print(f"DeepFace analysis error: {e}")
        continue

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display dominant emotion on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, dominant_emotion, (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)

    # Display the resulting frame
    cv2.imshow('Demo video', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
