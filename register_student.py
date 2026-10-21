
import cv2
import os

student_id = input("Enter student ID: ").strip()
student_name = input("Enter student name: ").strip()

if not student_id or not student_name:
    print("Student ID and name cannot be empty.")
    raise SystemExit

save_dir = os.path.join("dataset", student_id)
os.makedirs(save_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Face detector XML not found.")
    raise SystemExit

cap = cv2.VideoCapture(0)
count = 0
target = 30

print("Look at the camera. Press Q to stop.")

while count < target:
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        count += 1
        filename = os.path.join(
            save_dir, f"{student_id}_{count}.jpg"
        )
        cv2.imwrite(filename, face)

        cv2.rectangle(
            frame, (x, y), (x + w, y + h), (0, 255, 0), 2
        )

    cv2.putText(
        frame, f"Images: {count}/{target}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 255), 2
    )

    cv2.imshow("Student Registration", frame)

    if cv2.waitKey(150) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Captured {count} face images for {student_name}.")
print(f"Saved in: {save_dir}")
