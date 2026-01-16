# Lymphography Disease Prediction using Explainable AI (SHAP)

## 📌 Project Overview
This project presents an end-to-end machine learning framework for predicting
lymphatic diseases using the UCI Lymphography dataset.
To ensure transparency and trust in predictions, Explainable AI (XAI) techniques
using SHAP (SHapley Additive exPlanations) are integrated.

The project is based on a research-oriented approach and focuses on both
predictive performance and interpretability, which is crucial for healthcare
applications.

---

## 📊 Dataset Information
- **Dataset Name:** Lymphography Dataset
- **Source:** UCI Machine Learning Repository
- **Samples:** 148
- **Features:** 18 categorical clinical attributes
- **Target Classes:**
  - Normal
  - Metastases
  - Malignant Lymphoma
  - Fibrosis

---

## 🤖 Machine Learning Models Used
The following supervised learning models are implemented and compared:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree (DT)
- Multi-Layer Perceptron (MLP)
- Random Forest (RF)

Model evaluation is performed using stratified k-fold cross-validation.

---

## 📈 Evaluation Metrics
Models are evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Geometric Mean (GM)

A multi-criteria decision-making technique (TOPSIS) is used to rank models
based on overall performance.

---

## 🔍 Explainable AI (SHAP)
SHAP is used to provide:
- Global feature importance
- Local explanations for individual predictions
- Model transparency and interpretability

Key influential features such as **number of nodes**, **block of lymph c**, and
**changes in structure** are identified through SHAP analysis.

---

## 🏆 Best Model
Based on evaluation metrics and TOPSIS ranking:
- **Random Forest** achieved the best overall performance.

---

## 🗂️ Project Structure
```
Lymphography-XAI-Project/
│
├── data/ # Dataset
├── notebooks/ # Jupyter notebooks (EDA, training, evaluation, SHAP)
├── src/ # Modular source code
├── models/ # Saved trained models
├── results/ # Metrics and plots
├── app.py #  Deployment script
├── requirements.txt # Dependencies
└── README.md
```
---

## ⚙️ How to Run the Project
1. Clone the repository
```bash
git clone https://github.com/ad8034/Lymphography-XAI-Project.git
```
2. Setup venv
```
python -m venv venv
venv\scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run notebooks in order:
```
01_data_exploration.ipynb

02_model_training.ipynb

03_model_evaluation.ipynb

04_shap_explanation.ipynb
```