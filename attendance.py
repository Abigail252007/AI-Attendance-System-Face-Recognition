
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Student details
STUDENTS = {
    "STU001": "Abigail"
}

# Load trained face-recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

label_names = np.load(
    "label_names.npy", allow_pickle=True
).item()

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise SystemExit("Face detector XML could not be loaded.")

# Create attendance file if it doesn't exist
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    pd.DataFrame(
        columns=["Student ID", "Name", "Date", "Time"]
    ).to_csv(attendance_file, index=False)

# Load attendance already recorded today
today = datetime.now().strftime("%Y-%m-%d")
df = pd.read_csv(attendance_file)

recorded_today = set(
    df.loc[df["Date"].astype(str) == today, "Student ID"].astype(str)
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Could not open webcam.")

print("Attendance system started. Press Q to quit.")

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
        face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        label, confidence = recognizer.predict(face)

        student_id = label_names.get(label)
        name = STUDENTS.get(str(student_id), "Unknown")

        if confidence < threshold and name != "Unknown":
            color = (0, 255, 0)

            if student_id not in recorded_today:
                now = datetime.now()
                row = pd.DataFrame([{
                    "Student ID": str(student_id),
                    "Name": name,
                    "Date": now.strftime("%Y-%m-%d"),
                    "Time": now.strftime("%H:%M:%S")
                }])

                row.to_csv(
                    attendance_file,
                    mode="a",
                    header=False,
                    index=False
                )

                recorded_today.add(str(student_id))
                print(f"Attendance marked: {name} ({student_id})")

            message = f"{name} - Present"
        else:
            color = (0, 0, 255)
            message = "Unknown Face"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, message, (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Attendance saved in {attendance_file}")
