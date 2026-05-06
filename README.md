# 🛡️ Face Guard AI — Real-Time Face Attendance System

Face Guard AI is an advanced, production-ready facial recognition system designed to automate classroom or office attendance via CCTV camera feeds. It features a premium web dashboard for real-time monitoring, an automated enrollment system, and a human-in-the-loop review queue for low-confidence detections.

![Demo](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![InsightFace](https://img.shields.io/badge/Model-ResNet--50-orange) ![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey)

---

## ✨ Key Features

- **State-of-the-Art Face Recognition**: Powered by InsightFace (`buffalo_l` ResNet-50 backbone) and RetinaFace for highly accurate face detection and 512-dimensional embedding extraction.
- **Lightning-Fast Matching**: Utilizes FAISS (Facebook AI Similarity Search) to match faces against large databases in milliseconds.
- **Robust Image Preprocessing**: Built-in CLAHE (Contrast Limited Adaptive Histogram Equalization), Bilateral Denoising, and Unsharp Masking to improve recognition in poor CCTV lighting conditions.
- **Three-Tier Confidence System**:
  - **High Confidence (Match)**: Automatically logs attendance.
  - **Soft Match (Review)**: Routes the cropped face to an admin review queue for manual verification.
  - **Unknown**: Ignores unregistered faces.
- **Premium Web Dashboard**: A sleek, dark-themed, glassmorphism UI to monitor attendance, review flagged images, and manage enrolled students.
- **Interactive Live Demo**: An integrated playground where you can drag-and-drop group photos to instantly see bounding boxes, confidence scores, and recognition results.

---

## 🛠️ System Architecture

1. **Input**: Real-time CCTV stream (RTSP) or static image uploads.
2. **Detection**: RetinaFace detects bounding boxes and aligns faces to 112x112 crops.
3. **Enhancement**: Image preprocessing rectifies lighting and removes sensor noise.
4. **Embedding**: ResNet-50 extracts a 512-d L2-normalized identity vector.
5. **Matching**: FAISS computes Cosine Similarity against the enrolled student gallery.
6. **Logging**: SQLite records the timestamp, confidence score, and cropped evidence image.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/SaiSriKumar6686/face_recognition_based_attendance_system.git
cd face_recognition_based_attendance_system
```

### 3. Install Dependencies
Create a virtual environment and install the required Python packages:
```bash
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### 4. Run the Standalone Demo
The system comes pre-configured with a gallery of students and a built-in Flask web server.

```bash
# Set necessary environment variables
# Windows (PowerShell):
$env:PYTHONPATH="." ; $env:PYTHONIOENCODING="utf-8"

# Linux / macOS:
export PYTHONPATH="."
export PYTHONIOENCODING="utf-8"

# Launch the system
python scripts/demo.py --port 5000
```

### 5. Access the Web Interface
- **Admin Dashboard**: `http://localhost:5000/`
- **Interactive Recognition Demo**: `http://localhost:5000/demo`
- **Review Queue**: `http://localhost:5000/review`

---

## 📸 Enrolling New Students

To add a new student to the system:
1. Place 1-5 clear headshots of the student in `data/seed_images/<student_id>/`.
2. Use the Web UI (`http://localhost:5000/students` -> "Enroll Student") or run the CLI script:
```bash
python src/enrollment/enroll_student.py --student_id 23C11A0565 --name "John Doe" --roll_no "12345"
```
The system will automatically detect the face, extract the embedding, and update the FAISS index.

---

## ☁️ Cloud Deployment

This project includes a `Dockerfile` and is fully optimized for remote deployment on container-based hosting platforms like **Hugging Face Spaces**, **AWS ECS**, or **Render**.

**To deploy to Hugging Face Spaces (Free & Recommended for AI Demos):**
1. Create a new "Docker" Space on Hugging Face.
2. Link this GitHub repository or push the code directly to the Space.
3. Hugging Face will automatically build the Dockerfile and launch the web app.

---

## 📝 License & Acknowledgements

- Built utilizing the exceptional [InsightFace](https://github.com/deepinsight/insightface) library.
- Similarity search powered by [FAISS](https://github.com/facebookresearch/faiss).
