# Alzheimer's Disease Risk Classifier

A machine learning pipeline predicting Alzheimer's disease diagnosis from
clinical, lifestyle, and cognitive assessment data — built from scratch using
scikit-learn and deployed as a live demo on Hugging Face Spaces.

---

## Dataset

**Source:** Alzheimer's Disease Dataset (El Kharoua, 2024) via Kaggle
([link](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset))

- 2,149 patient records
- 33 features: demographic, lifestyle, medical history, clinical measurements,
  cognitive assessments, and symptom indicators
- Binary target: Alzheimer's diagnosis (0 = No, 1 = Yes)
- Class distribution: 1,389 negative / 760 positive (~65/35 split)

To reproduce this project:
1. Download the dataset from the link above
2. Place `alzheimers_disease_data.csv` from the source in the same directory as the notebook
3. Run all cells top to bottom
---

## Methodology

1. Dropped non-informative columns (`PatientID`, `DoctorInCharge`)
2. Confirmed all features already numeric — no encoding required
3. Stratified 70/30 train/test split (preserves class ratio)
4. Trained a Random Forest Classifier (100 estimators)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## Results

**Overall test accuracy: 93%** (n = 645 test samples)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| No Alzheimer's (0) | 0.93 | 0.98 | 0.95 |
| Alzheimer's (1) | 0.95 | 0.86 | 0.90 |

### Top 5 most important features (Gini importance)

| Feature | Importance |
|---|---|
| Functional Assessment | 0.177 |
| ADL (Activities of Daily Living) | 0.155 |
| MMSE (Mini-Mental State Exam) | 0.121 |
| Memory Complaints | 0.077 |
| Behavioral Problems | 0.044 |

These align exactly with the predictors identified in published Alzheimer's
literature — the model independently converged on clinically validated markers
without being told which features mattered.

---

## Live demo

Deployed on Hugging Face Spaces using Gradio — enter a patient's clinical
profile and get a model prediction in real time.

> [Alzheimer's Prediction Model](https://huggingface.co/spaces/Suj1234/alzheimersmodel1234)


---

## Tools & libraries

Python · scikit-learn · Pandas · NumPy · joblib · Hugging Face Spaces

---

## Files

| File | Description |
|---|---|
| `Alzeimers_detection.ipynb` | Full notebook with code and real outputs |
| `alzheimers_model.joblib` | Trained Random Forest model (saved with joblib) |
| `app.py` | Gradio web app deployed on Hugging Face |
| `requirements.txt` | Dependencies for the Hugging Face deployment |
