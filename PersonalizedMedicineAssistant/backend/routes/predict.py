from fastapi import APIRouter, HTTPException
import pickle
import pandas as pd
import numpy as np

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Load Model
model_path = "models/svc.pkl"
svc = pickle.load(open(model_path, "rb"))

# Load Symptoms List
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
    'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10,
    'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15,
    'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma'
}

# Prediction Function
@router.post("/")
async def predict_disease(symptoms: list):
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
        else:
            raise HTTPException(status_code=400, detail=f"Invalid symptom: {symptom}")

    input_df = pd.DataFrame([input_vector], columns=symptoms_dict.keys())
    predicted_label = svc.predict(input_df)[0]

    return {"predicted_disease": diseases_list.get(predicted_label, "Unknown disease")}
