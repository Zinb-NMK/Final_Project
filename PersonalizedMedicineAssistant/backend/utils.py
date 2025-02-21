import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)


def load_dataset(file_path):
    """
    Load a dataset and handle missing values.
    """
    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path)
            if data.empty:
                logging.warning(f"Dataset at {file_path} is empty.")
            # Handle missing values by filling with 'Unknown' or 0
            data.fillna('Unknown', inplace=True)
            logging.info(f"Loaded dataset: {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return None
    else:
        logging.warning(f"Warning: {file_path} not found.")
        return None


def load_general_disease_data(base_path):
    """
    Load general disease datasets from the given base path with validation.
    """
    general_data_files = [
        "description.csv",  # General disease description
        "diets.csv",  # Diets for general diseases
        "medications.csv",  # Medications for general diseases
        "precautions_df.csv",  # Precautions for general diseases
        "symtoms_df.csv",  # Symptoms dataset for general diseases
        "workout_df.csv",  # Workouts for general diseases
        "Training.csv",  # General disease training dataset
    ]

    data = {}
    for file in general_data_files:
        file_path = os.path.join(base_path, "general", file)  # Path to general folder
        dataset = load_dataset(file_path)
        if dataset is not None:
            # Example validation (you can add more checks as needed)
            if 'prognosis' not in dataset.columns:
                logging.warning(f"Column 'prognosis' missing in {file}.")
            data[file.split('.')[0]] = dataset

    return data


def load_cancer_data(base_path):
    """
    Load cancer-related datasets from the given base path with validation.
    """
    cancer_data_files = [
        "Symptom-severity.csv",  # Symptom severity for cancer
        "Final_Augmented_dataset_Diseases_and_Symptoms.csv",  # Augmented cancer dataset
        "symptom_Description.csv",  # Symptom descriptions for cancer
        "symptom_precaution.csv",  # Cancer symptom precautions
    ]

    data = {}
    for file in cancer_data_files:
        # Correct the path construction for cancer files
        file_path = os.path.join(base_path, "cancer", file)  # Ensure path is correctly formed
        dataset = load_dataset(file_path)
        if dataset is not None:
            # Validate columns if necessary (e.g., check for 'Cancer' or 'Symptoms' column)
            if 'Cancer' not in dataset.columns:
                logging.warning(f"Column 'Cancer' missing in {file}.")
            data[file.split('.')[0]] = dataset

    return data


def clean_user_input(user_input):
    """
    Clean user input by removing extra spaces and making all words lowercase.
    """
    symptoms = [symptom.strip().lower() for symptom in user_input.split(',')]
    return symptoms


if __name__ == "__main__":
    # Base path where your datasets are stored
    base_path = "backend/Data Sets"

    # Load general disease data
    general_disease_data = load_general_disease_data(base_path)

    # Print the columns of the general disease training dataset as a check
    if general_disease_data and 'Training' in general_disease_data:
        logging.info(f"Dataset columns: {general_disease_data['Training'].columns}")
    else:
        logging.error("General disease dataset 'Training' is missing or could not be loaded.")

    # Load cancer-related data
    cancer_data = load_cancer_data(base_path)

    # If cancer data is loaded successfully, log the dataset columns
    if cancer_data:
        for cancer_file, dataset in cancer_data.items():
            if dataset is not None:
                logging.info(f"Loaded cancer dataset: {cancer_file}")
                logging.info(f"Columns: {dataset.columns}")
            else:
                logging.error(f"Cancer dataset '{cancer_file}' could not be loaded.")

    # Now ask the user for input (e.g., symptoms)
    user_input = input("Enter your symptoms separated by commas: ")
    symptoms = clean_user_input(user_input)
    logging.info(f"User symptoms: {symptoms}")
