<div align="center">

# 🏥 Jaundice Predictor 🔬

### *AI-Powered Neonatal Jaundice Detection System*

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=F7B93E&center=true&vCenter=true&width=600&lines=95.36%25+Accuracy+Achieved!;LAB+Color+Space+%2B+KNN;Multiple+ML+Models+Tested;Real-time+Image+Processing" alt="Typing SVG" />

</div>

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

A comprehensive machine learning system for detecting neonatal jaundice through image analysis. This project implements multiple ML algorithms and color space transformations to achieve **95.36% accuracy** in classifying jaundice severity levels.

### ✨ Key Highlights

- 🎯 **95.36% Accuracy** - LAB color space with K-Nearest Neighbors (k=3)
- 🌈 **4 Color Spaces Tested** - RGB, HSV, YCbCr, and LAB
- 🤖 **5 ML Models Compared** - KNN, Random Forest, XGBoost, SVM, ResNet50
- 📸 **Advanced Image Processing** - CLAHE, Skin ROI extraction, Data augmentation
- 📊 **7000 Balanced Samples** - Binary classification (≤10 mg/dL vs >10 mg/dL)

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
│ Preprocessing   │
│  • CLAHE        │
│  • Skin ROI     │
│  • Resize 224×  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Color Space     │
│ Transformation  │
│  • RGB/HSV/     │
│    YCbCr/LAB    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Augmentation  │
│  • Rotation      │
│  • Crop          │
│  • Gentle Noise  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature         │
│ Extraction      │
│  • PCA (100)    │
│  • Flatten      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ML Models       │
│  • KNN          │
│  • Random Forest│
│  • XGBoost      │
│  • SVM/ResNet50 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prediction     │
│  ≤10 or >10     │
│   mg/dL         │
└─────────────────┘
```

---

## 🏆 Model Performance Comparison

### 📈 Best Results by Color Space

<table align="center">
<thead>
  <tr>
    <th>Color Space</th>
    <th>Best Model</th>
    <th>K Value</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-Score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><b>🥇 LAB</b></td>
    <td>KNN</td>
    <td>k=3</td>
<td><b>95.36%</b></td>      <td>96.49%</td>      <td>94.14%</td>    <td>95.30%</td>
  </tr>
  <tr>
    <td><b>🥈 HSV</b></td>
    <td>KNN</td>
    <td>k=3</td>
<td><b>94.43%</b></td>      <td>95.33%</td>      <td>91.00%</td>    <td>95.34%</td>
    <td>93.43%</td>
    <td>94.37%</td>
  </tr>
  <tr>
    <td><b>🥉 YCbCr</b></td>
    <td>KNN</td>
    <td>k=3</td>
<td><b>95.14%</b></td>    <td>95.80%</td>
    <td>94.43%</td>
    <td>95.11%</td>
  </tr>
  <tr>
    <td>RGB</td>
    <td>KNN</td>
    <td>k=3</td>
<td><b>94.14%</b></td>    <td>95.98%</td>
  95.21%  <td>92.14%</td>
    <td>94.02%</td>
  </tr>
</tbody>
</table>

### 🤖 Model Comparison (LAB Color Space)

| Model | Accuracy | Training Time | Strengths |
|-------|----------|--------------|----------|
| **KNN (k=3)** | **95.36%*** ⭐ | Fast | Simple, interpretable |
| Random Forest | 92.93% | Medium | Robust to overfitting |
| XGBoost | 91.29% | Medium | Good generalization |
| SVM | 83.57% | Slow | Works with high dimensions |
| ResNet50 | 81.21% | Slow | Deep feature learning |

---

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
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

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook jaundice_prediction.ipynb
   ```

4. **Mount Google Drive** (if using Google Colab)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

---

## 🚀 Usage

### Quick Start

```python
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load trained model
knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance')

# Process image
image = cv2.imread('path/to/image.jpg')
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image_resized = cv2.resize(image_lab, (224, 224))

# Predict
prediction = knn_model.predict(image_resized.reshape(1, -1))
result = "High Jaundice (>10 mg/dL)" if prediction == 1 else "Low/Normal (≤10 mg/dL)"
print(result)
```

### Full Pipeline Example

Refer to the `jaundice_prediction.ipynb` notebook for:
- Complete data preprocessing
- Model training with different color spaces
- Hyperparameter tuning
- Performance evaluation
- Confusion matrix visualization

---

## 📁 Project Structure

```
JAUNDICE_PREDICTOR/
│
├── 📓 jaundice_prediction.ipynb    # Main notebook with all experiments
├── 📋 requirements.txt              # Python dependencies
├── 📖 README.md                     # Project documentation
│
└── 📂 Data Structure (Google Drive)
    ├── images/                      # Raw neonatal images
    ├── sample_roi_output/           # Extracted skin ROIs
    ├── chd_jaundice_published_2.csv # Labels & metadata
    ├── X_aug_knn_lab.npy           # Processed features (LAB)
    └── y_aug_knn.npy               # Binary labels
```

---

## 🔬 Methodology

### 1. Image Preprocessing
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances image contrast
- **Skin ROI Extraction**: Isolates skin regions using color thresholding
- **Resize**: Standardizes images to 224×224 pixels

### 2. Color Space Transformations
Tested 4 color spaces to find optimal representation:
- **RGB**: Standard color space
- **HSV**: Separates hue, saturation, value
- **YCbCr**: Luminance and chrominance separation
- **LAB**: Perceptually uniform color space (BEST)

### 3. Data Augmentation
- Rotation (±3°)
- Random crop
- Gentle Gaussian noise (σ=0.005)
- Balanced to 3500 samples per class

### 4. Feature Engineering
- **Flattening**: 224×224×3 → 150,528 features
- **Standardization**: Zero mean, unit variance
- **PCA**: Dimensionality reduction to 100 components

### 5. Classification
Trained and evaluated 5 different algorithms with various hyperparameters

---

## 📊 Key Findings

### ✅ What Worked Best
1. **LAB Color Space**: Best accuracy (96.21%)
2. **K-Nearest Neighbors**: Simple yet effective
3. **Skin ROI Extraction**: Improved focus on relevant features
4. **Data Augmentation**: Better generalization
5. **PCA Preprocessing**: Faster training, similar performance

### ❌ Challenges & Limitations
- ResNet50 performed poorly (81.21%) - likely needs more data
- SVM struggled with high dimensionality
- Color calibration not implemented (lighting variations)
- Binary classification only (could be multi-class)

---

## 🎓 Technical Details

### Dataset Information
- **Total Samples**: 2,235 original images
- **Augmented Dataset**: 7,000 balanced samples
- **Class Distribution**: 3,500 per class (≤10 mg/dL vs >10 mg/dL)
- **Train/Test Split**: 80/20 stratified
- **Image Size**: 224×224×3

### Confusion Matrix (Best Model - LAB + KNN)

```
                Predicted
              ≤10    >10
Actual ≤10    676    24
       >10     41   659

Accuracy: 95.36%
Precision (≤10): 94.86%
Recall (≤10): 96.57%
F1-Score (≤10): 95.70%
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- 🎨 Improve UI/UX for predictions
- 📱 Create a mobile app version
- 🌐 Add web deployment
- 🔍 Implement multi-class classification
- 📊 Add more visualization tools
- 🧪 Test with different datasets

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is available for educational and research purposes.

---

## 🙏 Acknowledgments

- Dataset: CHD Jaundice Published Dataset
- Inspiration: Improving neonatal healthcare through AI
- Tools: TensorFlow, scikit-learn, OpenCV, PyTorch

---

## 📧 Contact

**Author**: Explicable11  
**Repository**: [JAUNDICE_PREDICTOR](https://github.com/Explicable11/JAUNDICE_PREDICTOR)

For questions or collaborations, please open an issue or reach out!

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

<img src="https://github-readme-stats.vercel.app/api/pin/?username=Explicable11&repo=JAUNDICE_PREDICTOR&theme=tokyonight" />

### 📈 Repository Stats

![GitHub last commit](https://img.shields.io/github/last-commit/Explicable11/JAUNDICE_PREDICTOR?style=flat-square)
![GitHub code size](https://img.shields.io/github/languages/code-size/Explicable11/JAUNDICE_PREDICTOR?style=flat-square)
![GitHub top language](https://img.shields.io/github/languages/top/Explicable11/JAUNDICE_PREDICTOR?style=flat-square)

---

**Made with ❤️ and 🤖 for better neonatal healthcare**

</div>
