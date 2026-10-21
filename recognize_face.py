
import cv2
import numpy as np

# Load the trained model and student labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

label_names = np.load(
    "label_names.npy", allow_pickle=True
).item()

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Face detector could not load.")
    raise SystemExit

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not open.")
    raise SystemExit

print("Face recognition started. Press Q to quit.")

# Initial testing threshold; confidence must be calibrated
# using known and unknown faces before relying on recognition.
threshold = 70

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < threshold and label in label_names:
            student_id = label_names[label]
            message = f"Student: {student_id}"
            color = (0, 255, 0)
        else:
            message = "Unknown Face"
            color = (0, 0, 255)

        cv2.rectangle(
            frame, (x, y), (x + w, y + h), color, 2
        )
        cv2.putText(
            frame, message, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    cv2.imshow("AI Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Face recognition closed.")
