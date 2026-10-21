# AI-Attendance-System-Face-Recognition
AI-based student attendance system using face recognition and OpenCV.
# AI Attendance System Using Face Recognition

## Project Overview
The **AI Attendance System Using Face Recognition** is a Python-based project that uses computer vision to recognize registered students and record their attendance automatically. It uses OpenCV for face detection and the LBPH face recognition algorithm to identify registered faces.

## Objectives
- Automate the student attendance process.
- Detect and recognize registered faces using a webcam.
- Reduce manual attendance-taking effort.
- Store attendance records with student details and dates.

## Technologies Used
- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Face Recognition:** LBPH (Local Binary Patterns Histograms)
- **Libraries:** NumPy, Pandas
- **Storage:** CSV files
- **Hardware:** Webcam

## Features
- Webcam testing
- Student face registration
- Face model training
- Real-time face recognition
- Automatic attendance recording
- Prevention of duplicate attendance entries for the same student on the same day

## Project Files
- `webcam_test.py` – Tests webcam functionality.
- `register_student.py` – Registers student faces.
- `train_model.py` – Trains the face recognition model.
- `recognize_face.py` – Recognizes registered faces.
- `attendance.py` – Records attendance.
- `haarcascade_frontalface_default.xml` – Face detection classifier.
- `face_model.yml` – Trained face recognition model.
- `label_names.npy` – Stores the student label information.

## Installation

### 1. Install Python
Install Python 3.10 or a compatible version on your computer.

### 2. Clone the Repository
```bash
git clone
