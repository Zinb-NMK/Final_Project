import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
dataset = pd.read_csv("Data Sets/Training.csv")

# Encoding target variable
le = LabelEncoder()
dataset["prognosis"] = le.fit_transform(dataset["prognosis"])

# Splitting Data
x = dataset.drop("prognosis", axis=1)
y = dataset["prognosis"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)

# Train SVC Model
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)

# Save Model if Not Exists
model_path = "models/svc.pkl"
if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    pickle.dump(svc, open(model_path, 'wb'))

# Load Saved Model
svc = pickle.load(open(model_path, 'rb'))

# Load Additional Data
sym_des = pd.read_csv("Data Sets/symtoms_df.csv")
precautions = pd.read_csv("Data Sets/precautions_df.csv")
workout = pd.read_csv("Data Sets/workout_df.csv")
description = pd.read_csv("Data Sets/description.csv")
medications = pd.read_csv("Data Sets/medications.csv")
diets = pd.read_csv("Data Sets/diets.csv")

# Symptoms and Disease Mappings
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6,
    'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
    'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31,
    'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer disease',
    1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension', 30: 'Migraine',
    7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox',
    11: 'Dengue', 37: 'Typhoid', 40: 'Hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E',
    3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemorrhoids (piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthritis',
    5: 'Arthritis', 0: '(vertigo) Paroxysmal Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis',
    27: 'Impetigo'
}

reverse_diseases_list = {v: k for k, v in diseases_list.items()}

# Helper Function
def get_recommendations(disease):
    desc = description[description['Disease'] == disease]['Description'].values
    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    meds = medications[medications['Disease'] == disease]['Medication'].values
    diet = diets[diets['Disease'] == disease]['Diet'].values
    work = workout[workout['disease'] == disease]['workout'].values

    return desc, pre, meds, diet, work

# Prediction Function
def predict_disease(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
        else:
            print(f"Warning: '{symptom}' is not recognized.")

    input_df = pd.DataFrame([input_vector], columns=x.columns)

    predicted_label = svc.predict(input_df)[0]

    return diseases_list.get(predicted_label, "Unknown disease")

# User Input
symptoms = input("Enter your symptoms separated by commas: ")
user_symptoms = [s.strip() for s in symptoms.split(',') if s.strip() in symptoms_dict]

if not user_symptoms:
    print("No valid symptoms entered. Please try again.")
else:
    predicted_disease = predict_disease(user_symptoms)
    desc, pre, meds, diet, work = get_recommendations(predicted_disease)

    # Display Results
    print("\n========== Predicted Disease ==========")
    print(predicted_disease)

    print("\n========== Description ==========")
    if len(desc) > 0:
        print(desc[0])
    else:
        print("No description available.")

    print("\n========== Precautions ==========")
    if len(pre) > 0:
        for i, p in enumerate(pre[0], 1):
            print(f"{i}. {p}")
    else:
        print("No precautions available.")

    print("\n========== Medications ==========")
    if len(meds) > 0:
        for i, m in enumerate(meds, 1):
            print(f"{i}. {m}")
    else:
        print("No medications available.")

    print("\n========== Workout ==========")
    if len(work) > 0:
        for i, w in enumerate(work, 1):
            print(f"{i}. {w}")
    else:
        print("No workout recommendations available.")

    print("\n========== Diets ==========")
    if len(diet) > 0:
        for i, d in enumerate(diet, 1):
            print(f"{i}. {d}")
    else:
        print("No diet recommendations available.")

# Print Sklearn Version (Debugging)
import sklearn
print("\nScikit-learn version:", sklearn.__version__)
