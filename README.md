# Hospital Readmission Prediction

## Overview
End-to-end machine learning project to predict whether a diabetic 
patient will be **readmitted to hospital within 30 days** using 
real clinical data from 130 US hospitals.

## Results
| Model | Recall | F1 | AUC |
|---|---|---|---|
| Dummy Classifier (baseline) | 0.000 | 0.000 | — |
| Logistic Regression | 0.546 | 0.271 | 0.675 |
| Random Forest | 0.062 | 0.107 | — |
| XGBoost (tuned) | 0.604 | 0.272 | 0.673 |
| **LightGBM (tuned)** | **0.610** | **0.265** | **0.666** |

## Why Recall?
In hospital readmission detection **Recall is more important 
than Accuracy** — missing a readmission is far more costly 
than a false alarm.

## Project Structure

hospital-readmission/
├── data/
│   └── diabetic_data.csv
├── notebooks/
│   └── hospital_readmission.ipynb
├── models/
│   └── hospital_readmission_model.pkl
├── requirements.txt
└── README.md

## Key Findings
- **Previous inpatient visits** is the strongest predictor 
  of readmission — confirmed by all 3 importance methods
- **Discharge destination** is critical — where a patient 
  goes after hospital significantly affects readmission risk
- **Disease complexity** — more medications, lab procedures 
  and diagnoses = higher readmission risk
- **Diabetes management** — insulin and medication changes 
  are consistently important predictors

## Steps
-  Exploratory Data Analysis (EDA)
-  Complex feature engineering & encoding
-  Handled severe class imbalance (89/11)
-  Trained and compared 5 models + DummyClassifier baseline
-  Hyperparameter tuning — LightGBM & XGBoost
- Threshold tuning analysis
-  Feature importance (3 methods)
-  Permutation importance
-  SHAP values for clinical explainability
-  Saved model as sklearn Pipeline

## Tools Used
- Python
- Scikit-Learn
- LightGBM
- XGBoost
- SHAP
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook
- GitHub

## Dataset
UCI Diabetes 130-US Hospitals Dataset
- **100,244** real patient records
- **112** features after preprocessing
- **130** US hospitals
- **Years:** 1999-2008

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
import joblib
import pandas as pd

# Load pipeline
model = joblib.load("hospital_readmission_model.pkl")

# Predict on new data — no scaling needed!
prediction = model.predict(new_patient_data)
```

## Clinical Disclaimer
This is an academic ML project — **not a clinically 
deployable system.** Real hospital readmission prediction 
requires clinical validation, regulatory approval and 
integration with EHR systems.

## Author
**Paul Oyeleke**
- GitHub: https://github.com/Oyeleke01
- LinkedIn: https://www.linkedin.com/in/paul-oyeleke-6a425b37b/

## Acknowledgements
Dataset: UCI Machine Learning Repository
Inspired by: Zero to Mastery ML Course — Daniel Bourke
