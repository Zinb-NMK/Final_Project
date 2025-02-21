import os
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from backend.utils import load_general_disease_data, load_cancer_data  # Ensure utils is correct

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to datasets
general_disease_data_path = "backend/Data Sets/general"
cancer_data_path = "backend/Data Sets/cancer"

# Function to load dataset
def load_general_disease_data(file_path):
    """
    Load a dataset and handle missing values.
    """
    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path)
            data.fillna('Unknown', inplace=True)  # Handle missing values
            logging.info(f"Loaded dataset: {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return None
    else:
        logging.warning(f"Warning: {file_path} not found.")
        return None

# Function to load all general disease data
def load_all_general_disease_data(base_path):
    """
    Load all general disease datasets from the base path.
    """
    general_data_files = [
        "description.csv",
        "diets.csv",
        "medications.csv",
        "precautions_df.csv",
        "symtoms_df.csv",
        "workout_df.csv",
        "Training.csv",
    ]

    data = {}
    for file in general_data_files:
        file_path = os.path.join(base_path, file)  # Path to general folder
        data[file.split('.')[0]] = load_general_disease_data(file_path)

    return data

# Load General Disease Dataset (Training.csv)
dataset_path = os.path.join(general_disease_data_path, "Training.csv")
dataset = load_general_disease_data(dataset_path)

# Check if dataset is loaded
if dataset is None:
    logging.error(f"General disease dataset '{dataset_path}' could not be loaded.")
    raise SystemExit("Dataset loading failed, please check your files.")

# Print column names to debug KeyError for 'prognosis'
logging.info(f"Dataset columns: {dataset.columns}")

# Ensure 'prognosis' column exists
if 'prognosis' not in dataset.columns:
    logging.error("'prognosis' column is missing from the dataset. Please check the dataset format.")
    raise SystemExit("Missing 'prognosis' column in the dataset.")

# Encoding target variable
le = LabelEncoder()
dataset["prognosis"] = le.fit_transform(dataset["prognosis"])

# Splitting Data
x = dataset.drop("prognosis", axis=1)
y = dataset["prognosis"]

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)

# Train SVC Model
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)

# Save Model if Not Exists
model_path = os.path.join(os.path.dirname(__file__), "models", "svc.pkl")
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(svc, open(model_path, 'wb'))
    logging.info(f"Model saved at {model_path}")

# Load Saved Model
svc = pickle.load(open(model_path, 'rb'))

# Load Additional Data using utils (ensure paths are correct)
description = load_general_disease_data(os.path.join(general_disease_data_path, "description.csv"))
precautions = load_general_disease_data(os.path.join(general_disease_data_path, "precautions_df.csv"))
workout = load_general_disease_data(os.path.join(general_disease_data_path, "workout_df.csv"))
medications = load_general_disease_data(os.path.join(general_disease_data_path, "medications.csv"))
diets = load_general_disease_data(os.path.join(general_disease_data_path, "diets.csv"))

# Load cancer data (ensure cancer data path is correct)
symptom_severity = load_cancer_data(os.path.join(cancer_data_path, "Symptom-severity.csv"))
final_augmented = load_cancer_data(os.path.join(cancer_data_path, "Final_Augmented_dataset_Diseases_and_Symptoms.csv"))
symptom_description = load_cancer_data(os.path.join(cancer_data_path, "symptom_Description.csv"))
symptom_precaution = load_cancer_data(os.path.join(cancer_data_path, "symptom_precaution.csv"))

# Symptoms and Disease Mappings
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x.columns)}
diseases_list = dict(enumerate(le.classes_))

# Helper Function for Recommendations
def get_recommendations(disease):
    try:
        desc = description[description['Disease'] == disease]['Description'].values
        pre = precautions[precautions['Disease'] == disease][
            ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
        meds = medications[medications['Disease'] == disease]['Medication'].values
        diet = diets[diets['Disease'] == disease]['Diet'].values
        work = workout[workout['disease'] == disease]['workout'].values
        return desc, pre, meds, diet, work
    except KeyError as e:
        logging.error(f"Error retrieving data for disease '{disease}': {e}")
        return None, None, None, None, None

# Prediction Function
def predict_disease(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
        else:
            logging.warning(f"Warning: '{symptom}' is not recognized. {symptom}")

    input_df = pd.DataFrame([input_vector], columns=x.columns)
    predicted_label = svc.predict(input_df)[0]
    return diseases_list.get(predicted_label, "Unknown disease")

# User Input
symptoms = input("Enter your symptoms separated by commas: ")
user_symptoms = [s.strip() for s in symptoms.split(',') if s.strip() in symptoms_dict]

if not user_symptoms:
    logging.warning("No valid symptoms entered. Please try again.")
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
