# CSET343 — AI in Healthcare: Coursework Labs

This directory contains standalone Python implementations for practical laboratory assignments in **CSET343: Artificial Intelligence in Healthcare**. The labs cover fundamental machine learning, multiclass medical diagnosis, deep learning with handcrafted computer vision features, and neural network hyperparameter optimization.

---

## 📚 Lab Assignments Overview

| Lab | Title | Task / Modality | Primary Models & Techniques |
| :--- | :--- | :--- | :--- |
| **Lab 4** | **Logistic Regression on Breast Cancer** | Tabular / Diagnostic Binary Classification | Logistic Regression, StandardScaler, ROC-AUC, Precision/Recall |
| **Lab 5** | **Multiclass Classification on Dermatology** | Tabular / 6-Class Differential Diagnosis | Logistic Regression, k-NN, Random Forest, SVM, Decision Tree, Imputation |
| **Lab 6** | **Hybrid CNN & Handcrafted Features on Chest X-Rays** | Medical Imaging / Pneumonia vs. Normal | Custom CNN, HOG, LBP, GLCM Texture Descriptors, Stratified K-Fold |
| **Lab 7** | **Grid Search on Deep Learning Model** | Tabular / Diabetes Risk Classification | Keras Sequential, SciKeras `KerasClassifier`, GridSearchCV, Implausible Zero Remediation |

---

## 🔬 Lab Details & Execution

### Lab 4: Logistic Regression on Breast Cancer Wisconsin Dataset
- **File**: `lab4_logistic_regression_breast_cancer.py`
- **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC) loaded directly via `scikit-learn` (no internet required).
- **Description**: Evaluates benign vs. malignant tumor classification, computes statistical summaries, handles feature scaling, and plots decision thresholds, ROC curves, and confusion matrices.
- **Run**:
  ```bash
  python lab4_logistic_regression_breast_cancer.py
  ```
- **Outputs**: High-resolution diagnostic plots saved to `./lab4_outputs/`.

---

### Lab 5: Multiclass Classification on Dermatology Dataset
- **File**: `lab5_multiclass_dermatology.py`
- **Dataset**: UCI Dermatology Dataset (366 records, 34 clinical/histopathological features, 6 disease classes).
- **Description**: Performs missing value imputation (age attribute), standardizes clinical metrics, and benchmarks 5 classifiers across multi-class precision, recall, F1-score, and One-vs-Rest ROC curves.
- **Run**:
  ```bash
  python lab5_multiclass_dermatology.py
  ```
- **Outputs**: Class-wise performance summaries and confusion matrices saved to `./lab5_outputs/`.

---

### Lab 6: Hybrid CNN & Handcrafted Computer Vision on Chest X-Rays
- **File**: `lab6_chest_xray_cnn.py` (alias: `lab6.py`)
- **Dataset**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (~5,863 pediatric chest radiographs).
- **Description**:
  1. Trains an end-to-end Convolutional Neural Network (CNN) with dropout and batch normalization.
  2. Extracts traditional handcrafted medical imaging texture features:
     - **HOG**: Histogram of Oriented Gradients (structural contours).
     - **LBP**: Local Binary Patterns (micro-texture irregularities).
     - **GLCM**: Gray-Level Co-occurrence Matrix (contrast, dissimilarity, homogeneity, energy, correlation).
  3. Evaluates classical classifiers (SVM, RF, k-NN, Logistic Regression) on handcrafted features and compares against the CNN.
- **Setup**: Place or extract the dataset such that the directory matches `chest_xray/train/`, `chest_xray/val/`, `chest_xray/test/`.
- **Run**:
  ```bash
  python lab6_chest_xray_cnn.py
  ```
- **Outputs**: Loss curves, accuracy plots, and ROC figures saved to `./lab6_outputs/`.

---

### Lab 7: Grid Search Hyperparameter Tuning on Keras Neural Network
- **File**: `lab7_grid_search_pima.py` (alias: `lab7.py`)
- **Dataset**: Pima Indians Diabetes Dataset (auto-downloaded via raw URL).
- **Description**: Implements clinical zero-as-missing preprocessing for physiological indicators (Glucose, BloodPressure, SkinThickness, Insulin, BMI) followed by median imputation. Tunes deep neural network architecture parameters (hidden neurons, activations, weight initializers, dropout rates, batch sizes, epochs) using `scikeras` and `GridSearchCV`.
- **Run**:
  ```bash
  python lab7_grid_search_pima.py
  ```
- **Outputs**: Console logs detailing mean test scores across candidate grids and test evaluation metrics.

---

## 🛠 Installation & Setup

1. Navigate to the `labs` directory:
   ```bash
   cd labs
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run any lab script directly with `python <script_name>.py`.
