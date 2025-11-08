<div align="center">

# 🏥 Enhancing Neonatal Jaundice Detection 🔬
## A Color Space-Aware PCA-KNN Approach

### *AI-Powered Neonatal Jaundice Detection System*

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=F7B93E&center=true&vCenter=true&width=600&lines=95.36%25+Accuracy+Achieved!;LAB+Color+Space+%2B+KNN;Multiple+ML+Models+Tested;Real-time+Image+Processing" alt="Typing SVG" />

</div>

---

## 👥 Authors

**Aryan Singh¹**, **Manish Pratap Singh²** (Corresponding Author), **Dev Ayush³**, **Rohit Kumar Tiwari⁴** (Corresponding Author), **Sushil Kumar Saroj⁴**

¹ Department of Computer Science Engineering, Indian Institute of Technology Indian School of Mines, Dhanbad, Jharkhand 826004 India  
² Department of Physics, Faculty of Engineering and Technology, V. B. S. Purvanchal University Jaunpur, Uttar Pradesh 222003 India  
³ SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, 603203 India  
⁴ Department of Computer Science Engineering, Madan Mohan Malaviya University of Technology, Gorakhpur Uttar Pradesh 273010 India

📧 **Corresponding Authors**: rohitkushinagar@gmail.com

---

## 📄 Abstract

Early detection of neonatal jaundice caused by elevated bilirubin levels is crucial to prevent neurological damage. Traditional non-invasive methods often rely on resource-intensive deep learning models, limiting their deployment in low-resource settings. This study introduces a lightweight, interpretable, and scalable machine learning pipeline for jaundice detection using infant skin region images. 

Our approach integrates **color space-aware preprocessing**, including CLAHE enhancement in the LAB space and HSV-based thresholding for yellow-tinted skin region extraction. The extracted images undergo dimensionality reduction via **incremental PCA**, followed by classification using a **distance-weighted KNN model**. 

**Achieving an accuracy of 95.36%** and an F1-score of 0.95 across bilirubin classes, our model outperforms previous studies with smaller datasets or complex deep networks. The proposed pipeline, tested across multiple color spaces and classifiers, demonstrates optimal performance with LAB and KNN, offering a **low-cost solution for mobile screening and telehealth applications**.

### 🔑 Keywords
Neonatal Jaundice, K-Nearest Neighbors, Principal Component Analysis, Non-Invasive Screening, Mobile Health Applications

---

## 📊 Project Overview

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
    <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
    <img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
    <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>

A comprehensive machine learning system for detecting neonatal jaundice through image analysis. This project implements a **color space-aware PCA-KNN pipeline** with multiple ML algorithms and color space transformations to achieve **95.36% accuracy** in classifying jaundice severity levels.

### ✨ Key Highlights

- 🎯 **95.36% Accuracy** - LAB color space with K-Nearest Neighbors (k=3)
- 🌈 **4 Color Spaces Tested** - RGB, HSV, YCbCr, and LAB
- 🤖 **5 ML Models Compared** - KNN, Random Forest, XGBoost, SVM, ResNet50
- 📸 **Advanced Image Processing** - CLAHE, Skin ROI extraction, Data augmentation
- 📊 **7000 Balanced Samples** - Binary classification (≤10 mg/dL vs >10 mg/dL)
- 🔬 **Lightweight & Interpretable** - No black-box CNNs, deployable on mobile devices
- ⚡ **Fast Inference** - Suitable for real-time screening

---

## 🎨 Architecture & Workflow

```
┌─────────────────┐
│  Input Images   │
│   (Neonates)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing  │
│   • CLAHE      │
│   • Skin ROI   │
│   • Resize 224×│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Color Space   │
│ Transformation │
│   • RGB/HSV/   │
│    YCbCr/LAB   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Data Augmentation│
│  • Rotation    │
│  • Crop        │
│  • Gentle Noise│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature     │
│   Extraction   │
│  • PCA (100)   │
│  • Flatten     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Models    │
│    • KNN       │
│ • Random Forest│
│   • XGBoost    │
│ • SVM/ResNet50│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prediction   │
│   ≤10 or >10  │
│     mg/dL      │
└─────────────────┘
```

---

## 📋 Dataset Description

This study employs the publicly available **Neo Natal Jaundice dataset** curated by **Xuzhou Central Hospital**, comprising:

- 📷 **2,235 clinical images** from **745 neonates**
- 🏯 Each image taken in controlled clinical environment
- 📍 Visible skin regions from head, face, and chest
- 📏 Resolution: typically **567×567 pixels**
- 🎯 Ground truth: Total Serum Bilirubin (TSB) levels validated by pediatric experts

### 🛡️ Data Preprocessing & Augmentation

**Binary Classification Threshold**: 10.0 mg/dL
- ≤ 10.0 mg/dL: Normal or mild jaundice
- > 10.0 mg/dL: Elevated or severe jaundice

**Augmentation Strategy** (applied to balance dataset):
1. Minor rotation (±2°)
2. Random cropping (224×224)
3. Subtle Gaussian noise (σ=0.005)
4. Resizing to 232×232 before augmentation

**Final Balanced Dataset**: **7,000 samples** (3,500 per class)

**Train/Test Split**: 80/20 stratified split
- Training: 5,600 samples
- Testing: 1,400 samples

---
