from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages, auth
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import joblib

from .models import Medical, User, Ment, Profile


def home(request):
    return render(request, 'home.html')


def registerView(request):
    return render(request, 'register.html')


def registerUser(request):
    if request.method == "POST":
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '').strip()

        if not username or not email or not password:
            messages.error(request, "All fields are required.")
            return redirect('reg')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken. Please choose a different one.")
            return redirect('reg')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered. Try logging in.")
            return redirect('reg')

        User.objects.create(
            username=username,
            email=email,
            password=make_password(password),
            is_patient=True
        )
        messages.success(request, "Account created successfully. You can now log in.")
        return redirect('reg')

    messages.error(request, "Invalid request method.")
    return redirect('reg')


def loginView(request):
    if request.method == "POST":
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_active:
            auth.login(request, user)
            return redirect('patient' if user.is_patient else 'doctor' if user.is_doctor else 'login')
        else:
            messages.error(request, "Invalid Username or Password.")
            return redirect('login')
    return render(request, 'login.html')


@login_required
def patient_home(request):
    doctor_count = User.objects.filter(is_doctor=True).count()
    patient_count = User.objects.filter(is_patient=True).count()
    appointment_count = Ment.objects.filter(approved=True).count()
    medical_total = Medical.objects.all().count()
    medical_pending = Medical.objects.filter(medicine='See Doctor').count()
    medical_completed = medical_total - medical_pending
    
    user_profile = Profile.objects.filter(user_id=request.user.id).exists()
    
    context = {
        "status": "1" if user_profile else "0",
        "doctor": doctor_count,  
        "ment": appointment_count,  
        "patient": patient_count,  
        "drug": medical_completed,
        "profile_status": "Please Create Profile To Continue" if not user_profile else ""
    }
    
    return render(request, 'patient/home.html', context)


@login_required
def create_profile(request):
    if request.method == "POST":
        birth_date = request.POST.get('birth_date', '').strip()
        region = request.POST.get('region', '').strip()
        country = request.POST.get('country', '').strip()
        gender = request.POST.get('gender', '').strip()

        if not birth_date or not region or not country or not gender:
            messages.error(request, "All fields are required.")
            return redirect('create_profile')

        if Profile.objects.filter(user_id=request.user.id).exists():
            messages.error(request, "Profile already exists. You can update it instead.")
            return redirect('patient')

        Profile.objects.create(
            user=request.user, 
            birth_date=birth_date, 
            gender=gender, 
            region=region,
            country=country
        )
        messages.success(request, "Your Profile Was Created Successfully.")
        return redirect('patient')

    context = {
        "user_profile": Profile.objects.filter(user=request.user).first(),
        "gender": ["Male", "Female", "Other"]
    }
    return render(request, 'patient/create_profile.html', context)


def diagnosis(request):
	symptoms = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']
	symptoms = sorted(symptoms)
	context = {'symptoms':symptoms, 'status':'1'}
	return render(request, 'patient/diagnosis.html', context)

  
@csrf_exempt
@csrf_exempt
def MakePredict(request):
    s1 = request.POST.get('s1')
    s2 = request.POST.get('s2')
    s3 = request.POST.get('s3')
    s4 = request.POST.get('s4')
    s5 = request.POST.get('s5')
    id = request.POST.get('id')

    list_b = [s1, s2, s3, s4, s5]
    print(list_b)

    list_a = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
              'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
              'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain',
              'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
              'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
              'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
              'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
              'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
              'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
              'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
              'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
              'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
              'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
              'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
              'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
              'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
              'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
              'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
              'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine',
              'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
              'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
              'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history',
              'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
              'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
              'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
              'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
              'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
              'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

    # Convert symptoms into binary form
    list_c = [1 if symptom in list_b else 0 for symptom in list_a]

    test = np.array(list_c).reshape(1, -1)
    print(test.shape)

    # Dictionary to map predicted numbers to disease names
    disease_map = {
        0: "Fungal Infection",
        1: "Allergy",
        2: "GERD",
        3: "Chronic Cholestasis",
        4: "Drug Reaction",
        5: "Peptic Ulcer Disease",
        6: "AIDS",
        7: "Diabetes",
        8: "Gastroenteritis",
        9: "Bronchial Asthma",
        10: "Hypertension",
        11: "Migraine",
        12: "Cervical Spondylosis",
        13: "Paralysis",
        14: "Jaundice",
        15: "Malaria",
        16: "Chickenpox",
        17: "Dengue",
        18: "Typhoid",
        19: "Hepatitis A",
        20: "Hepatitis B",
        21: "Hepatitis C",
        22: "Hepatitis D",
        23: "Hepatitis E",
        24: "Alcoholic Hepatitis",
        25: "Tuberculosis",
        26: "Common Cold",
        27: "Pneumonia",
        28: "Dimorphic Hemorrhoids (Piles)",
        29: "Heart Attack",
        30: "Varicose Veins",
        31: "Hypothyroidism",
        32: "Hyperthyroidism",
        33: "Hypoglycemia",
        34: "Osteoarthritis",
        35: "Arthritis",
        36: "Vertigo",
        37: "Acne",
        38: "Urinary Tract Infection",
        39: "Psoriasis",
        40: "Impetigo"
    }

    try:
        clf = joblib.load('model/lr_classifier.pkl')
        prediction = clf.predict(test)
        predicted_number = int(prediction[0])  # Convert to integer for mapping
        result = disease_map.get(predicted_number, "Unknown Disease")  # Get disease name
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    # Save to database
    Medical.objects.create(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, disease=result, patient_id=id)

    return JsonResponse({'status': result})
