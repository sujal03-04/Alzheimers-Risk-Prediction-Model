import gradio as gr
import joblib
import numpy as np
import pandas as pd

model = joblib.load("alzheimers_model.joblib")

FEATURE_NAMES = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
    'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
    'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
    'Confusion', 'Disorientation', 'PersonalityChanges',
    'DifficultyCompletingTasks', 'Forgetfulness'
]

def yn(val):
    return 1 if val == "Yes" else 0

def predict(age, gender, ethnicity, education, bmi, smoking, alcohol, physical_activity,
            diet_quality, sleep_quality, family_history, cardiovascular, diabetes,
            depression, head_injury, hypertension, systolic_bp, diastolic_bp,
            chol_total, chol_ldl, chol_hdl, chol_trig, mmse, functional_assessment,
            memory_complaints, behavioral_problems, adl, confusion, disorientation,
            personality_changes, difficulty_tasks, forgetfulness):

    row = [
        age, yn(gender) if gender in ["Male","Female"] else (0 if gender=="Male" else 1),
        ethnicity, education, bmi, yn(smoking), alcohol, physical_activity,
        diet_quality, sleep_quality, yn(family_history), yn(cardiovascular), yn(diabetes),
        yn(depression), yn(head_injury), yn(hypertension), systolic_bp, diastolic_bp,
        chol_total, chol_ldl, chol_hdl, chol_trig, mmse, functional_assessment,
        yn(memory_complaints), yn(behavioral_problems), adl, yn(confusion),
        yn(disorientation), yn(personality_changes), yn(difficulty_tasks), yn(forgetfulness)
    ]

    input_df = pd.DataFrame([row], columns=FEATURE_NAMES)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        result = f"Model output: Higher-risk pattern\nConfidence: {probability[1]*100:.1f}%"
    else:
        result = f"Model output: Lower-risk pattern\nConfidence: {probability[0]*100:.1f}%"

    return result

with gr.Blocks(title="Alzheimer's Risk Pattern Classifier — Demo") as demo:
    gr.Markdown("# Alzheimer's Risk Pattern Classifier")
    gr.Markdown(
        "**Educational demo only — not a diagnostic tool.** "
        "This interface runs a Random Forest classifier trained on a public clinical dataset "
        "(2,149 patients, 32 features). It is intended to demonstrate a machine learning "
        "pipeline, not to provide medical advice. Always consult a qualified clinician."
    )

    with gr.Tabs():
        with gr.Tab("Prediction"):
            gr.Markdown("### Demographics")
            with gr.Row():
                age = gr.Slider(60, 90, value=75, label="Age")
                gender = gr.Radio(["Male", "Female"], value="Male", label="Gender")
                ethnicity = gr.Slider(0, 3, value=0, step=1, label="Ethnicity (coded 0-3, dataset-defined categories)")
                education = gr.Slider(0, 3, value=1, step=1, label="Education level (coded 0-3)")

            gr.Markdown("### Lifestyle")
            with gr.Row():
                bmi = gr.Slider(15, 40, value=27.6, label="BMI")
                smoking = gr.Radio(["No", "Yes"], value="No", label="Smoking")
                alcohol = gr.Slider(0, 20, value=10, label="Alcohol consumption (units/week)")
            with gr.Row():
                physical_activity = gr.Slider(0, 10, value=5, label="Physical activity (hours/week)")
                diet_quality = gr.Slider(0, 10, value=5, label="Diet quality score")
                sleep_quality = gr.Slider(4, 10, value=7, label="Sleep quality score")

            gr.Markdown("### Medical history")
            with gr.Row():
                family_history = gr.Radio(["No", "Yes"], value="No", label="Family history of Alzheimer's")
                cardiovascular = gr.Radio(["No", "Yes"], value="No", label="Cardiovascular disease")
                diabetes = gr.Radio(["No", "Yes"], value="No", label="Diabetes")
                depression = gr.Radio(["No", "Yes"], value="No", label="Depression")
            with gr.Row():
                head_injury = gr.Radio(["No", "Yes"], value="No", label="History of head injury")
                hypertension = gr.Radio(["No", "Yes"], value="No", label="Hypertension")

            gr.Markdown("### Clinical measurements")
            with gr.Row():
                systolic_bp = gr.Slider(90, 180, value=135, label="Systolic BP (mmHg)")
                diastolic_bp = gr.Slider(60, 120, value=90, label="Diastolic BP (mmHg)")
            with gr.Row():
                chol_total = gr.Slider(150, 300, value=225, label="Total cholesterol (mg/dL)")
                chol_ldl = gr.Slider(50, 200, value=125, label="LDL cholesterol (mg/dL)")
                chol_hdl = gr.Slider(20, 100, value=60, label="HDL cholesterol (mg/dL)")
                chol_trig = gr.Slider(50, 400, value=228, label="Triglycerides (mg/dL)")

            gr.Markdown("### Cognitive and functional assessment")
            with gr.Row():
                mmse = gr.Slider(0, 30, value=15, label="MMSE score (0-30, higher = better cognition)")
                functional_assessment = gr.Slider(0, 10, value=5, label="Functional assessment (0-10, higher = more independent)")
                adl = gr.Slider(0, 10, value=5, label="ADL score (0-10, higher = better daily function)")

            gr.Markdown("### Symptoms")
            with gr.Row():
                memory_complaints = gr.Radio(["No", "Yes"], value="No", label="Memory complaints")
                behavioral_problems = gr.Radio(["No", "Yes"], value="No", label="Behavioral problems")
                confusion = gr.Radio(["No", "Yes"], value="No", label="Confusion")
            with gr.Row():
                disorientation = gr.Radio(["No", "Yes"], value="No", label="Disorientation")
                personality_changes = gr.Radio(["No", "Yes"], value="No", label="Personality changes")
                difficulty_tasks = gr.Radio(["No", "Yes"], value="No", label="Difficulty completing tasks")
                forgetfulness = gr.Radio(["No", "Yes"], value="No", label="Forgetfulness")

            predict_btn = gr.Button("Run prediction", variant="primary")
            output = gr.Textbox(label="Result")

            predict_btn.click(
                fn=predict,
                inputs=[age, gender, ethnicity, education, bmi, smoking, alcohol, physical_activity,
                        diet_quality, sleep_quality, family_history, cardiovascular, diabetes,
                        depression, head_injury, hypertension, systolic_bp, diastolic_bp,
                        chol_total, chol_ldl, chol_hdl, chol_trig, mmse, functional_assessment,
                        memory_complaints, behavioral_problems, adl, confusion, disorientation,
                        personality_changes, difficulty_tasks, forgetfulness],
                outputs=output
            )

        with gr.Tab("Methodology & performance"):
            gr.Markdown("""
### Dataset
2,149 patient records, 32 clinical/demographic/lifestyle features, sourced from a
public Kaggle dataset (El Kharoua, 2024). Target: binary Alzheimer's diagnosis
(1,389 negative, 760 positive).

### Model
Random Forest Classifier (scikit-learn), 100 estimators, trained on a 70/30
stratified train/test split (random_state=42).

### Test set performance
| Metric | Class 0 (No AD) | Class 1 (AD) |
|---|---|---|
| Precision | 0.93 | 0.95 |
| Recall | 0.98 | 0.86 |
| F1-score | 0.95 | 0.90 |

**Overall accuracy: 93%** (n=645 test samples)

### Top 5 most important features (by Gini importance)
1. Functional Assessment (0.18)
2. ADL score (0.15)
3. MMSE score (0.12)
4. Memory Complaints (0.08)
5. Behavioral Problems (0.04)

These align with the predictors identified as most significant in published
literature on this dataset (functional assessment, ADL, MMSE, memory complaints,
and behavioral problems were independently found to be the top predictors).

### Limitations
- Trained on a single dataset; not validated on independent populations
- Not peer-reviewed, not clinically validated, not regulatory-approved
- For educational demonstration only
            """)

demo.launch()