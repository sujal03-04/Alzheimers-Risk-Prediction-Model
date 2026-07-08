# Alzheimer's Disease Risk Classifier

A machine learning pipeline to predict Alzheimer's disease diagnosis from
clinical, lifestyle, and cognitive assessment data.

## Dataset
2,149 patient records, 33 features, sourced from a public clinical dataset
(El Kharoua, 2024, Kaggle). Target: binary Alzheimer's diagnosis.

Dataset: [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)
by Rabie El Kharoua, available on Kaggle under CC BY 4.0.

To reproduce this project:
1. Download the dataset from the link above
2. Place `alzheimers_disease_data.csv` in the same directory as the notebook
3. Run all cells top to bottom
   
## Model
Random Forest Classifier (scikit-learn), 100 estimators, stratified 70/30
train/test split.

## Results
- **Overall accuracy: 93%**
- Seizure detection precision/recall: 0.93/0.98 (no Alzheimer's), 0.95/0.86 (Alzheimer's)

## Top predictors (by Gini importance)
1. Functional Assessment (0.18)
2. ADL score — Activities of Daily Living (0.15)
3. MMSE score — Mini-Mental State Exam (0.12)
4. Memory Complaints (0.08)
5. Behavioral Problems (0.04)

These align with predictors identified as significant in published Alzheimer's
literature — the model independently converged on clinically validated markers.

## Live demo
Deployed on Hugging Face Spaces: [Alzheimer's Prediction Model](https://huggingface.co/spaces/Suj1234/alzheimersmodel1234)

## Tools
Python · scikit-learn · Pandas · NumPy · joblib
