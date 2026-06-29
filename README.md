# 🦴 Osteoporosis Detection Using Deep Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-ff4b4b?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Medical AI](https://img.shields.io/badge/Medical-AI-purple?style=for-the-badge)

</p>

---

# 📖 Project Overview

Osteoporosis is a silent bone disease that gradually weakens bones by reducing Bone Mineral Density (BMD), increasing the risk of fractures. Traditional diagnosis relies on expensive DEXA (DXA) scanning systems, which are often unavailable in rural or small healthcare centers.

This project presents an **AI-powered Osteoporosis Detection System** that uses **Deep Learning (CNN)** to analyze Knee X-ray images and classify them into:

- ✅ Normal
- ✅ Osteoporosis

The system also generates:

- 🔥 Grad-CAM Heatmap
- 📄 Professional PDF Report
- 📊 Confidence Score
- 🩺 Clinical Impression

through an easy-to-use **Streamlit Web Application**.

---

# 🎯 Objectives

- Develop an AI-based osteoporosis detection system.
- Detect osteoporosis using standard Knee X-ray images.
- Reduce dependence on expensive DEXA scans.
- Provide quick and affordable screening.
- Generate Grad-CAM visualization for explainability.
- Automatically generate professional medical reports.

---

# ✨ Features

✅ Deep Learning Based Detection

✅ Binary Classification

✅ Custom CNN Architecture

✅ Streamlit Web Application

✅ Patient Information Module

✅ Image Upload Interface

✅ Automatic Prediction

✅ Confidence Score

✅ AI Heatmap (Grad-CAM)

✅ Professional PDF Report Generation

✅ User Friendly Interface

---

# 🏗 System Architecture

The complete workflow of the project follows the pipeline below:

```
Patient
    │
    ▼
Upload Knee X-ray
    │
    ▼
Image Preprocessing
    │
    ▼
Custom CNN Model
    │
    ▼
Prediction
    │
    ├────────► Confidence Score
    │
    ├────────► Grad-CAM Heatmap
    │
    └────────► PDF Report Generation
```

---

# ⚙ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | CNN Model |
| Streamlit | Web Application |
| OpenCV | Image Processing |
| NumPy | Numerical Operations |
| PIL | Image Handling |
| Matplotlib | Visualization |
| ReportLab | PDF Report Generation |
| Google Colab | Model Training |
| VS Code | Development |
| GitHub | Version Control |

---

# 🧠 Deep Learning Model

The project uses a **Custom Convolutional Neural Network (CNN)** developed using TensorFlow and Keras.

## Architecture

```
Input Image (256×256×3)

↓

Image Rescaling

↓

Conv2D (32)

↓

MaxPooling

↓

Conv2D (64)

↓

MaxPooling

↓

Conv2D (64)

↓

MaxPooling

↓

Conv2D (64)

↓

MaxPooling

↓

Conv2D (64)

↓

MaxPooling

↓

Conv2D (64)

↓

MaxPooling

↓

Flatten

↓

Dense (64)

↓

Softmax Output
```

### Activation Function

- ReLU

### Output Layer

- Softmax

### Optimizer

- Adam

### Loss Function

- Sparse Categorical Crossentropy

---

# 📂 Dataset

## Dataset Sources

- Osteoporosis Knee X-ray Dataset
- Osteoporosis Database (Kaggle)

## Classes

- Normal
- Osteoporosis

## Image Size

```
256 × 256 × 3
```

---

# 🔄 Image Preprocessing

The uploaded X-ray image undergoes:

- Image Resizing
- Pixel Normalization
- RGB Conversion
- Tensor Conversion
- CNN Input Formatting

---

# 📊 Performance

| Metric | Value |
|---------|--------|
| Training Accuracy | ~94% |
| Validation Accuracy | ~91% |
| Test Accuracy | ~92% |
| Precision | 93% |
| Recall | 92% |
| F1 Score | 92% |

---

# 🔥 Explainable AI

The project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Benefits:

- Highlights important regions of the X-ray.
- Improves model interpretability.
- Helps doctors understand AI decisions.
- Increases trust in predictions.

---

# 📄 PDF Report

The generated PDF includes:

- Patient Details
- Report ID
- Age
- Gender
- Date
- AI Prediction
- Confidence Score
- Clinical Significance
- Reliability
- AI Heatmap
- Clinical Impression
- Medical Disclaimer

---

# 🖥 Web Application

The Streamlit application allows users to:

- Enter patient details
- Upload Knee X-ray
- Predict Bone Condition
- View Confidence Score
- Display Heatmap
- Download Professional PDF Report

---

# 📁 Project Structure

```
OSTEOPOROSIS-DETECTION-USING-DEEP-LEARNING
│
├── FINAL DATASETS
│   ├── normal
│   └── osteoporosis
│
├── IMP FILES
│
├── workflow
│
├── app.py
│
├── osteoporosis_model.h5
│
├── osteoporosis_2_0.ipynb
│
├── Report_Guest.pdf
│
├── How to Run.txt
│
├── PROJ INFO & VIVA.pdf
│
└── README.md
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Osteoporosis-Detection-Using-Deep-Learning.git
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 📷 Working Procedure

```
Upload X-ray Image

↓

Image Preprocessing

↓

CNN Prediction

↓

Confidence Score

↓

Grad-CAM Heatmap

↓

Generate PDF Report

↓

Download Report
```

---

# 🎯 Advantages

- Low Cost
- Fast Prediction
- User Friendly
- Explainable AI
- Automatic Report Generation
- Suitable for Rural Clinics
- Reduces Diagnosis Time
- Supports Early Detection

---

# 🔮 Future Scope

- Multi-Class Classification
- DICOM Image Support
- Cloud Deployment
- Mobile Application
- Hospital Integration
- Doctor Dashboard
- Electronic Health Record Integration
- Multi-Disease Detection

---

# 👨‍💻 Team Members

**Mohammad Hassan**

**Ansari Aaftab Hussain**

**Ansari Maaz Anzar**

**Rehan Ali**

Department of Computer Engineering

MMANTC, Malegaon

Academic Year **2025–2026**

---

# 🙏 Acknowledgement

We sincerely thank:

- MMANTC, Malegaon
- Department of Computer Engineering
- Our Project Guide
- TensorFlow Team
- Streamlit
- Kaggle
- OpenCV Community
- Google Colab
- GitHub

for their valuable support and open-source contributions.

---

# 📚 References

1. TensorFlow Documentation
2. Keras Documentation
3. Streamlit Documentation
4. OpenCV Documentation
5. Kaggle Osteoporosis Datasets
6. WHO Osteoporosis Guidelines

---

# 📜 License

This project has been developed for academic purposes as a Final Year Engineering Project.

---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.

---

<p align="center">

## 🦴 Osteoporosis Detection Using Deep Learning

**Artificial Intelligence for Early Bone Disease Screening**

Developed with ❤️ by Team MMANTC

</p>
