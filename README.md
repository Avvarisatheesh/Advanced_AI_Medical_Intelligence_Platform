# 🩺 Advanced AI Medical Intelligence Platform

An end-to-end AI-powered healthcare application that analyzes Chest X-ray images using Deep Learning, explains predictions with Explainable AI (Grad-CAM), generates AI-assisted medical reports using Google Gemini, stores prediction history in SQLite, and provides a modern Flask web interface.

---

## Features

- Chest X-ray Disease Detection
- Deep Learning (CNN)
- Explainable AI using Grad-CAM
- AI Medical Report Generation using Google Gemini
- Medicine Recommendation System
- Prediction History Management
- REST API
- Interactive Dashboard
- User-Friendly Flask Web Application

---

## Technologies Used

### Backend
- Python
- Flask
- TensorFlow
- Keras
- SQLite
- Google Gemini API

### Frontend
- HTML
- CSS
- JavaScript

### Machine Learning
- Convolutional Neural Network (CNN)
- Grad-CAM
- Cosine Similarity

---

## Project Structure


Advanced_AI_Medical_Intelligence_Platform/
│
├── app.py
├── predict.py
├── gradcam.py
├── gemini_helper.py
├── database.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── xray_model.keras
│   ├── vectorizer.pkl
│   ├── vectors.pkl
│   └── medicine_data.pkl
│
├── templates/
├── static/
├── uploads/
└── medical_history.db


---

## Dataset

- Chest X-ray Images
- Classes:
  - NORMAL
  - PNEUMONIA

Image Size:

224 × 224 pixels

---

## Model Performance

Accuracy: *89.74%*

---

## Explainable AI

Grad-CAM highlights the important regions of the X-ray image that influenced the model's prediction.

---

## AI Medical Report

Google Gemini generates:

- Prediction Summary
- Clinical Interpretation
- Recommended Next Steps
- Lifestyle Advice
- Disclaimer

---

## REST API

### Get Prediction History


GET /api/history


Returns prediction history in JSON format.

---

## Database

SQLite stores:

- Filename
- Disease
- Confidence
- Prediction Time

---

## Installation

bash
git clone <repository-url>
cd Advanced_AI_Medical_Intelligence_Platform

pip install -r requirements.txt

python app.py


---

## Future Improvements

- Multi-disease Detection
- CT Scan Support
- MRI Analysis
- Doctor Dashboard
- Cloud Deployment
- Mobile Application
- Authentication
- Electronic Health Record Integration

---

## Author

Avvari Satheesh
B.Tech CSE - Artificial Intelligence
