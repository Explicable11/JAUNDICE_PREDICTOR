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


## 🔬 Proposed Methodology

Our methodology follows a systematic pipeline integrating perceptual color enhancement, automated ROI extraction, dimensionality reduction, and lightweight classification.

### 1️⃣ Preprocessing and ROI Extraction

#### 🎨 Color Enhancement (CLAHE in LAB Space)
- Convert RGB images to **LAB color space** for perceptual uniformity
- Apply **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to L-channel
- Corrects illumination inconsistencies and reveals subtle yellowing
- Mitigates device-specific lighting artifacts while preserving physiological color cues

#### 🟡 Yellow Region Segmentation (HSV-based)
- Transform CLAHE-enhanced image to **HSV color space**
- Apply calibrated HSV thresholds to isolate yellow-tinted skin regions  
- Use morphological operations (erosion and dilation) to enhance segmented regions
- Extract **150×150-pixel ROI** centered on largest contiguous yellow patch
- Focus on medically relevant zones indicative of bilirubin-induced discoloration

### 2️⃣ Color Space Transformation

After preprocessing, ROI images are converted into multiple color spaces:

- **RGB**: Standard color space baseline
- **HSV**: Separates hue and saturation from brightness
- **YCbCr**: Isolates luminance from chrominance  
- **LAB**: Perceptually uniform, captures blue-yellow axis changes (🌟 BEST for jaundice)

### 3️⃣ Feature Extraction and Dimensionality Reduction

#### Raw Pixel Utilization
- Use raw pixel intensities from 224×224 skin ROI as input
- Preserves original spatial and color information
- Avoids bias from handcrafted statistical descriptors
- Results in **150,528-dimensional vectors**

#### Standardization
- Apply **Zero-mean, unit-variance scaling** via StandardScaler
- Ensures all features contribute equally during PCA and classification

#### Incremental PCA (IPCA)  
- Scalable alternative to traditional PCA that processes dataset in mini-batches
- Avoids RAM overloading with large datasets
- Reduce dimensionality from **150,528 → 100 principal components**
- Captures most discriminative variance while filtering noise

### 4️⃣ Classification (Distance-Weighted KNN)

**K-Nearest Neighbors (KNN)** configuration:
- **k = 5** nearest neighbors  
- **Distance-weighted voting**: Closer neighbors have greater influence
- **Euclidean distance** metric

**Prediction Formula**:
```
ŷ(x) = arg max Σ wi × 𝟙(yi = c)
         c∈C  i∈Nk(x)

where: wi = 1 / (d(x, xi)^2 + ε)
```

### 5️⃣ Performance Evaluation Metrics

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

Stratified 80/20 train-test split preserves class distribution.

---

## 🏆 Experimental Results

### 🎯 Classification Performance by Color Space

Table below presents accuracy results for each classifier-color space combination:

| Color Space | KNN | Random Forest | XGBoost | SVM |
|-------------|------|---------------|---------|------|
| RGB | **94.14%** | 91.93% | 92.29% | 84.36% |
| HSV | **94.43%** | 92.29% | 91.00% | 81.93% |
| YCbCr | **95.14%** | 92.43% | 91.36% | 82.14% |
| **LAB** | **🌟 95.36%** | 92.93% | 91.29% | 83.57% |

**Key Observation**: LAB + KNN achieves the highest accuracy (95.36%), confirming that perceptually uniform color spaces better capture subtle yellowish skin tone shifts correlated with elevated bilirubin levels.

### 📊 Detailed Performance Metrics (Best Results by Color Space)

| Color Space | Best Model | K Value | Accuracy | Precision | Recall | F1-Score |
|-------------|------------|---------|----------|-----------|--------|----------|
| 🥇 **LAB** | KNN | k=3 | **95.36%** | **96.49%** | **94.14%** | **95.30%** |
| 🥈 HSV | KNN | k=3 | 94.43% | 95.33% | 91.00% | 94.37% |
| 🥉 YCbCr | KNN | k=3 | 95.14% | 95.80% | 94.43% | 95.11% |
| RGB | KNN | k=3 | 94.14% | 95.98% | 92.14% | 94.02% |

### 🤖 Model Comparison (LAB Color Space)

| Model | Accuracy | Training Time | Strengths |
|-------|----------|---------------|----------|
| **KNN (k=3)** | **95.36%** ⭐ | Fast | Simple, interpretable |
| Random Forest | 92.93% | Medium | Robust to overfitting |
| XGBoost | 91.29% | Medium | Good generalization |
| SVM | 83.57% | Slow | Works with high dimensions |
| ResNet50 | 81.21% | Slow | Deep feature learning |

**Analysis**: KNN outperforms tree-based ensembles and deep learning models, demonstrating that local neighborhood-based learning is well-suited for biomedical image tasks where class boundaries are subtle.

### 🎯 Confusion Matrix (LAB + KNN)

**Best Model Performance**:
```
              Predicted
            ≤10 mg/dL  >10 mg/dL
Actual ≤10   676        24
       >10    41        659

True Positives:  676  |  False Positives:  24
False Negatives: 41   |  True Negatives:  659

Accuracy:  95.36%
Precision: 96.49%
Recall:    94.14%
F1-Score:  95.30%
```

**Clinical Significance**:
- ✅ **Low False Negatives (41)**: Critical for neonatal safety - minimizes missed jaundice cases
- ✅ **Low False Positives (24)**: Reduces unnecessary clinical interventions  
- ✅ **High True Positives (676)**: Excellent detection of elevated bilirubin cases
- ✅ **High Recall (94.14%)**: Prioritizes catching jaundiced infants

### 📈 Multi-K Robustness Test

Tested KNN stability across different K values:

| Color Space | K=4 | K=5 | K=6 |
|-------------|-----|-----|-----|
| RGB | 94.00% | 94.14% | 93.43% |
| HSV | 94.36% | 94.43% | 93.14% |
| YCbCr | 95.50% | 95.14% | 95.43% |
| **LAB** | **95.10%** | **95.36%** | **95.00%** |

**Conclusion**: LAB consistently performs above 95% across different K values, demonstrating robustness to hyperparameter changes without extensive tuning.

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
Python 3.8+
Google Colab
GPU (optional, for deep learning models)
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Explicable11/JAUNDICE_PREDICTOR.git
cd JAUNDICE_PREDICTOR
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Open in Google Colab**
   - Upload `jaundice_prediction.ipynb` to Google Colab
   - Or open directly from Google Drive

4. **Mount Google Drive** (if using Google Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Quick Start Example

```python
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load and preprocess image
image = cv2.imread('path/to/neonate_image.jpg')
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image_resized = cv2.resize(image_lab, (224, 224))

# Flatten and predict
features = image_resized.flatten().reshape(1, -1)
prediction = knn_model.predict(features)

result = "High Jaundice (>10 mg/dL)" if prediction == 1 else "Low/Normal (≤10 mg/dL)"
print(f"Prediction: {result}")
```

---

## 📝 Key Findings & Conclusion

### ✅ What Worked Best

1. **LAB Color Space**: Achieved highest accuracy (95.36%) due to perceptual uniformity
2. **K-Nearest Neighbors**: Simple yet effective, outperformed complex deep learning models
3. **Skin ROI Extraction**: Improved focus on relevant features
4. **Data Augmentation**: Better generalization with balanced 7,000 samples
5. **Incremental PCA**: Efficient dimensionality reduction to 100 components

### 📊 Clinical Impact

- **95.36% Accuracy**: Reliable enough for clinical screening
- **High Recall (94.14%)**: Minimizes false negatives - critical for infant safety
- **Fast Inference**: Suitable for real-time mobile screening
- **Interpretable**: No black-box CNNs, clear decision process
- **Low-Resource Friendly**: Can run on mobile devices

### 🎯 Conclusion

This study successfully demonstrates a **lightweight, interpretable machine learning framework** for non-invasive neonatal jaundice detection. By leveraging **color space-aware preprocessing** (LAB+CLAHE), **HSV-based segmentation**, **incremental PCA**, and **distance-weighted KNN**, our pipeline achieves **95.36% accuracy** - outperforming previous studies with smaller datasets or complex deep networks.

The proposed method offers a **practical, scalable solution for mobile health applications** and **resource-constrained clinical settings**, making early jaundice screening accessible in underserved regions.

---

## 📚 Project Structure

```
JAUNDICE_PREDICTOR/
├── 📓 jaundice_prediction.ipynb    # Main notebook with experiments
├── 📋 requirements.txt              # Python dependencies  
├── 📖 README.md                     # Project documentation
└── 📂 Data Structure (Google Drive)
    ├── images/                      # Raw neonatal images
    ├── sample_roi_output/           # Extracted skin ROIs
    ├── chd_jaundice_published_2.csv # Labels & metadata
    ├── X_aug_knn_lab.npy            # Processed features (LAB)
    └── y_aug_knn.npy                # Binary labels
```

---

## 🤝 Contributing

Contributions are welcome! Ideas for enhancement:
- 🎨 Improve UI/UX for predictions
- 📱 Create mobile app version
- 🌐 Add web deployment  
- 🔍 Implement multi-class classification
- 📊 Add more visualization tools

---

## 🙏 Acknowledgments

- **Dataset**: Neo Natal Jaundice Dataset from Xuzhou Central Hospital
- **Inspiration**: Improving neonatal healthcare through AI
- **Tools**: TensorFlow, scikit-learn, OpenCV, PyTorch, Jupyter
- **Research Paper**: "Enhancing Neonatal Jaundice Detection: A Color Space-Aware PCA-KNN Approach"

---

## 📧 Contact

**Authors**: Aryan Singh, Manish Pratap Singh, Dev Ayush, Rohit Kumar Tiwari, Sushil Kumar Saroj  
**Corresponding Authors**: rohitkushinagar@gmail.com  
**Repository**: [JAUNDICE_PREDICTOR](https://github.com/Explicable11/JAUNDICE_PREDICTOR)

For questions or collaborations, please open an issue!

---

### ⭐ Star this repo if you found it helpful!

**Made with ❤️ and 🤖 for better neonatal healthcare**
