import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

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
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x.columns)}

diseases_list = dict(enumerate(le.classes_))

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
    print("\n================= Predicted Disease =============")
    print(predicted_disease)

    print("================= Description ==================")
    print(desc[0] if len(desc) > 0 else "No description available.")

    print("================= Precautions ==================")
    if len(pre) > 0:
        for i, p in enumerate(pre[0], 1):
            print(f"{i} :  {p}")
    else:
        print("No precautions available.")

    print("================= Medications ==================")
    if len(meds) > 0:
        for i, m in enumerate(meds, 1):
            print(f"{i} :  {m}")
    else:
        print("No medications available.")

    print("================= Workout ==================")
    if len(work) > 0:
        for i, w in enumerate(work, 1):
            print(f"{i} :  {w}")
    else:
        print("No workout recommendations available.")

    print("================= Diets ==================")
    if len(diet) > 0:
        for i, d in enumerate(diet, 1):
            print(f"{i} :  {d}")
    else:
        print("No diet recommendations available.")
