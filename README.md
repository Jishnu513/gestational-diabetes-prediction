# Gestational Diabetes Prediction with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-yellow.svg)](https://shap.readthedocs.io/)

Machine Learning system for Gestational Diabetes Mellitus (GDM) prediction using XGBoost with SHAP explainability. Developed during ML internship at **NIT Puducherry (Apr-May 2024)**.

## 🎯 Project Overview

Predicts gestational diabetes risk using 6 maternal health indicators with 94.6% accuracy and explainable AI for clinical trust.

## 🏆 Key Achievements

- ✅ Implemented **5+ ML models** (XGBoost, Random Forest, SVM variants)
- ✅ Achieved **94.6% testing accuracy** with 18.6% improvement from baseline
- ✅ Processed **1,000+ medical data samples** using Python and Pandas
- ✅ Integrated **SHAP explainability** for clinical interpretability
- ✅ Applied feature engineering and metric-driven evaluation

## 🛠️ Technology Stack

- **Language**: Python 3.11+
- **ML Framework**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: SHAP

## 🚀 Quick Start

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Run the Interactive Web Application:
```bash
python -m streamlit run app.py
```

### Train the model (Optional, model is already provided):
```bash
python train_model.py
```

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Maternal age (years) |
| Pregnancy No | Numerical | Number of pregnancies |
| Weight | Numerical | Body weight (kg) |
| Height | Numerical | Height (cm) |
| BMI | Numerical | Body Mass Index |
| Heredity | Binary (0/1) | Family history of diabetes |

**Dataset**: 1,012 samples, perfectly balanced classes

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 97.03% |
| Testing Accuracy | **94.58%** |
| Precision | 95% |
| Recall | 95% |
| F1-Score | 95% |

### Model Comparison

| Model | Accuracy |
|-------|----------|
| **XGBoost (Optimized)** | **94.58%** |
| Random Forest | 92.12% |
| SVM (RBF) | 89.66% |
| XGBoost (Default) | 88.67% |
| SVM (Linear) | 86.21% |

**Improvement**: 18.6% from baseline (76.0% → 94.6%)

## 🔍 Feature Importance (SHAP)

1. **BMI** - 34.2% (strongest predictor)
2. **Heredity** - 28.7% (family history)
3. **Age** - 19.8% (maternal age)

## 🏗️ Model Architecture
```python
Pipeline([
    ('preprocessor', StandardScaler()),
    ('classifier', XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8
    ))
])
```

## 💡 Key Features

- **5+ Algorithm Comparison**: XGBoost, Random Forest, SVM variants
- **SHAP Explainability**: Transparent predictions for clinical use
- **Optimized Hyperparameters**: Systematic tuning for best performance
- **Production Pipeline**: StandardScaler + XGBoost for consistent preprocessing
- **Interactive Predictions**: Beautiful Streamlit web interface for real-time risk assessment and SHAP visualization

## 📁 Project Structure
```
├── app.py                  # Main Streamlit Web Application
├── train_model.py          # Model training & evaluation script
├── gdm_model.pkl           # Pre-trained XGBoost Model
├── requirements.txt        # Dependencies
├── data/
│   └── Gestational_Diabetes.csv
└── README.md
```

## 🎯 Usage Example
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('gdm_model.pkl')

# Patient data
patient = pd.DataFrame({
    'Age': [28],
    'Pregnancy No': [2],
    'Weight': [68.5],
    'Height': [165],
    'BMI': [25.2],
    'Heredity': [1]
})

# Predict
prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0]

print(f"Risk: {'High' if prediction == 1 else 'Low'}")
print(f"Confidence: {probability[prediction]*100:.1f}%")
```

## 🔬 Methodology

1. **Data Preprocessing**: StandardScaler normalization, missing value handling
2. **Feature Engineering**: BMI calculation, heredity encoding
3. **Model Training**: Stratified 80-20 train-test split
4. **Hyperparameter Tuning**: Optimized for accuracy and generalization
5. **Evaluation**: Multiple metrics (accuracy, precision, recall, F1)
6. **Explainability**: SHAP values for feature importance

## 📊 Results & Visualizations

- Confusion Matrix: 96% correct predictions
- SHAP Summary Plot: Feature importance visualization
- Model Comparison Chart: Performance across 5+ algorithms

## 🚀 Future Enhancements

- [x] Web interface with Flask/Streamlit (Completed)
- [ ] REST API for production deployment
- [ ] Cross-validation for robust estimates
- [ ] Additional features (glucose levels, blood pressure)
- [ ] Mobile app integration


## 👨‍💻 Author

**Jishnu**  
B.Tech Computer Science, Amrita Vishwa Vidyapeetham  
Internship: National Institute of Technology Puducherry


## 🙏 Acknowledgments

- NIT Puducherry for internship opportunity
- Scikit-learn and XGBoost communities
- SHAP developers for explainability framework

---
