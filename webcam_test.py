
import cv2

# Open the laptop webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    print("Check whether another app is using the camera.")
    raise SystemExit

# Load OpenCV's face detector
cascade_path = "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("ERROR: Could not load the face detector.")
    camera.release()
    raise SystemExit

print("Webcam started!")
print("Look at the camera. Press Q to close.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read a webcam frame.")
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )
        cv2.putText(
            frame, "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

    cv2.putText(
        frame, f"Faces: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 255), 2
    )

    cv2.imshow("AI Attendance - Face Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print("Webcam closed successfully.")
