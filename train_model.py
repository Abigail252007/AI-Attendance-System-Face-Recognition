
import cv2
import os
import numpy as np

dataset_dir = "dataset"
faces = []
labels = []
label_names = {}

label_number = 0

for student_id in os.listdir(dataset_dir):
    student_path = os.path.join(dataset_dir, student_id)

    if not os.path.isdir(student_path):
        continue

    label_names[label_number] = student_id

    for filename in os.listdir(student_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(student_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                image = cv2.resize(image, (200, 200))
                faces.append(image)
                labels.append(label_number)

    label_number += 1

if len(faces) == 0:
    print("No face images found. Register a student first.")
    raise SystemExit

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels, dtype=np.int32))
recognizer.save("face_model.yml")

np.save("label_names.npy", label_names)

print("Model trained successfully!")
print("Total training images:", len(faces))
print("Registered students:", len(label_names))
print("Saved model as face_model.yml")
