# 🫁 Pneumonia Detection using Deep Learning

Deep learning project for detecting **pneumonia** from chest X-ray images using three architectures: **Custom CNN**, **ResNet50**, and **Vision Transformer (ViT)**.

---

## 📋 Project Overview

- Binary classification of chest X-rays: Normal vs Pneumonia  
- Preprocessed and augmented dataset (~60K images) with stratified train/val/test splits  
- Comparative evaluation of three state-of-the-art models  

---

## 🏗️ Model Architectures

| Model | Parameters | Key Feature |
|-------|-----------|-------------|
| Custom CNN | ~1.2M | Lightweight, fast inference, edge-friendly |
| ResNet50 | ~25M | Transfer learning, robust features, strong generalization |
| Vision Transformer (ViT) | ~86M | Attention-based, state-of-the-art accuracy |

---

## 📊 Dataset & Preprocessing

- **Sources**: Kaggle Chest X-Ray Datasets (~60K images)  
- **Preprocessing**:
  - Standardized to 256×256 px, PNG format  
  - Removed duplicates, achieved class balance (~1:1 ratio)  
  - Data augmentation: rotations, flips, brightness/contrast, zoom, shearing  

---

## 🔧 Tools & Libraries

- **Deep Learning**: PyTorch, torchvision, transformers  
- **Data Handling**: PIL, OpenCV, pandas  
- **Visualization**: matplotlib, seaborn  
- **Evaluation**: scikit-learn  
- **Environment**: Google Colab (GPU-enabled)  

---

## 📈 Results

| Model | Test Accuracy | Avg F1-Score |
|-------|--------------|-------------|
| Custom CNN | 95.28% | 0.952 |
| ResNet50 | 98.84% | 0.989 |
| ViT | 98.38% | 0.983 |

**Insights:**  
- **ResNet50** – Best overall balance of accuracy, precision, recall  
- **ViT** – High accuracy with attention mechanisms  
- **Custom CNN** – Lightweight, suitable for resource-constrained devices  

---

## 🚀 Recommendations

- **Production/Clinical Use:** ResNet50  
- **Edge/Mobile Deployment:** Custom CNN  
- **Research/High Accuracy Experiments:** ViT  

---

## 🔍 Key Takeaways

1. **Data Quality Matters** – Cleaning and preprocessing improved performance significantly  
2. **Class Balance is Critical** – Prevented bias toward Pneumonia class  
3. **Transfer Learning Works** – ResNet50 and ViT outperformed the custom CNN  
4. **Regularization** – Dropout and early stopping prevented overfitting  
5. **Comprehensive Evaluation** – Confusion matrices, per-class F1, misclassified examples  

---

## 🛠️ Installation & Usage

**Steps:**
1. Open `Pneumonia-Project.ipynb` in Google Colab or Jupyter  
2. Download datasets or mount Google Drive  
3. Run cells sequentially to train and evaluate models  
4. Models are saved automatically after training  
5. Training history, confusion matrices, and misclassified examples are displayed in the notebook  

```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn pandas opencv-python pillow tqdm
