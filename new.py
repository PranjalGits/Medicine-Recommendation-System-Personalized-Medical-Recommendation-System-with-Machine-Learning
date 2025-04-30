import streamlit as st
import pandas as pd
import numpy as np
import pickle
import speech_recognition as sr
import tempfile
import os
import base64
from pydub import AudioSegment
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import streamlit_option_menu

import subprocess

# === Custom Imports === #
from Utils.Agent import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
from Utils.brain_of_doctor import encode_image, analyze_image_with_query
from Utils.voice_of_patient import transcribe_with_groq
from Utils.voice_of_doctor import text_to_speech_with_gtts
from groq import Groq

# === Configuration === #
st.set_page_config(page_title="🩺 Personalized Healthcare System", layout="wide")

# === Load Data === #
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")
svc = pickle.load(open('svc.pkl', 'rb'))

# Symptoms and diseases dictionary
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}


diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# === Lottie Animation Loader === #
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_health = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_mjlh3hcy.json")

# === Sidebar Navigation === #
with st.sidebar:
    st_lottie(lottie_health, height=150)
    st.title("🩺 Health Menu")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Symptom Checker", "Report Analyzer", "Voice & Vision Chatbot", "AI Doctor Chat"],
        icons=["house", "activity", "cloud-upload", "mic", "robot"],
        default_index=0
    )

# === Utility Functions === #
def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'])
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    med = [m for m in medications[medications['Disease'] == dis]['Medication'].values]
    die = [d for d in diets[diets['Disease'] == dis]['Diet'].values]
    wrkout = workout[workout['disease'] == dis]['workout'].values
    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

def extract_symptoms_from_text(text):
    extracted = []
    text = text.lower()
    for key in symptoms_dict.keys():
        label = key.replace("_", " ").strip()
        if label in text:
            extracted.append(key)
    return extracted

def record_symptoms():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Speak now...")
        audio = recognizer.listen(source, phrase_time_limit=5)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"📝 Detected symptoms: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("😕 Could not understand audio.")
            return ""
        except sr.RequestError as e:
            st.error(f"❌ API error: {e}")
            return ""

# === App Pages === #
if selected == "Home":
    col1, col2 = st.columns(2)
    with col1:
        st.title("🏠 Welcome to Your AI Healthcare Companion")
        st.markdown("""
        This smart assistant uses **ML**, **AI**, **speech**, and **image processing** to help patients:
        - Diagnose symptoms
        - Analyze medical reports
        - Chat with virtual doctors
        - Get recommendations (diet, exercise, medication)
        """)
    with col2:
        st_lottie(lottie_health, height=300)

elif selected == "Symptom Checker":
    st.subheader("🔍 Symptom-Based Disease Predictor")
    user_input = st.text_input("Enter your symptoms (comma-separated)", placeholder="e.g. headache, nausea, fever")

    if st.button("🎙 Record Symptoms"):
        spoken = record_symptoms()
        if spoken:
            matched_symptoms = extract_symptoms_from_text(spoken)
            st.session_state["spoken_symptoms"] = ", ".join(matched_symptoms)
            if not matched_symptoms:
                st.warning("😕 No matching symptoms found in your speech.")
            else:
                st.success(f"✅ Matched Symptoms: {', '.join(matched_symptoms)}")

    symptom_input = st.text_input("📝 Or use extracted symptoms", value=st.session_state.get("spoken_symptoms", ""))

    if st.button("🚨 Predict Disease"):
        if not symptom_input:
            st.warning("⚠️ Please enter valid symptoms.")
        else:
            user_symptoms = [s.strip().lower() for s in symptom_input.split(',')]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, my_pre, meds, diet, wrkout = helper(predicted_disease)

            st.markdown(f"## 🦠 Predicted Disease: `{predicted_disease}`")
            st.markdown(f"**📖 Description:** {dis_des}")
            st.markdown("### 💊 Medications")
            st.write(", ".join(meds))
            st.markdown("### 🍎 Diet Plan")
            st.write(", ".join(diet))
            st.markdown("### 🏋️ Exercise Routine")
            st.write(", ".join(wrkout))
            st.markdown("### 🚑 Precautions")
            st.write(", ".join(my_pre[0]))

elif selected == "Report Analyzer":
    st.subheader("📑 Upload Medical Report for AI Analysis")
    uploaded_file = st.file_uploader("Upload .txt report", type="txt")

    if uploaded_file:
        report_text = uploaded_file.read().decode("utf-8")
        if st.button("🤖 Analyze with Doctors"):
            with st.spinner("Running expert analysis..."):
                agents = {
                    "Cardiologist": Cardiologist(report_text),
                    "Psychologist": Psychologist(report_text),
                    "Pulmonologist": Pulmonologist(report_text)
                }
                responses = {}
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(agent.run): name for name, agent in agents.items()}
                    for future in as_completed(futures):
                        agent_name = futures[future]
                        responses[agent_name] = future.result()

                team_agent = MultidisciplinaryTeam(
                    cardiologist_report=responses["Cardiologist"],
                    psychologist_report=responses["Psychologist"],
                    pulmonologist_report=responses["Pulmonologist"]
                )
                final_diagnosis = team_agent.run()
                st.success("✅ Final Report Generated")
                st.markdown("### 🧠 Team Diagnosis")
                st.markdown(final_diagnosis)

elif selected == "Voice & Vision Chatbot":
    st.subheader("👂🧠 Voice + Vision Diagnosis")
    audio_file = st.file_uploader("🎤 Upload an audio file", type=["mp3", "wav"])
    image_file = st.file_uploader("🖼 Upload a patient image", type=["jpg", "jpeg", "png"])

    if st.button("🩺 Diagnose from Audio + Image"):
        if not audio_file:
            st.warning("⚠️ Please upload audio.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                tmp_audio.write(audio_file.read())
                audio_path = tmp_audio.name
            transcription = transcribe_with_groq(
                stt_model="whisper-large-v3",
                audio_filepath=audio_path,
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
            )
            st.markdown("### 🗣 Transcription")
            st.write(transcription)

            if image_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                    tmp_img.write(image_file.read())
                    image_path = tmp_img.name
                encoded_img = encode_image(image_path)
                doctor_response = analyze_image_with_query(
                    query=f"Patient said: {transcription}",
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    encoded_image=encoded_img
                )
            else:
                doctor_response = f"Based on audio: {transcription}"

            st.markdown("### 👨‍⚕️ Diagnosis")
            st.write(doctor_response)
            output_path = os.path.join("results", "doctor_response.mp3")
            text_to_speech_with_gtts(doctor_response, output_path)
            st.audio(output_path, format="audio/mp3")

elif selected == "AI Doctor Chat":
    st.subheader("💬 Ask Your AI Doctor")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    if user_input := st.chat_input("Ask a medical question..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        chat_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        formatted_history = [{"role": role, "content": msg} for role, msg in st.session_state.chat_history]
        response = chat_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=formatted_history,
            temperature=0.7
        )
        bot_reply = response.choices[0].message.content
        st.chat_message("assistant").markdown(bot_reply)
        st.session_state.chat_history.append(("assistant", bot_reply))
        output_voice_path = os.path.join("results", "chatbot_voice.mp3")
        text_to_speech_with_gtts(bot_reply, output_voice_path)
        st.audio(output_voice_path, format="audio/mp3")

# === Footer === #
with st.expander("ℹ️ About"):
    st.write("An AI-powered assistant to help patients with disease prediction, voice & vision diagnosis, and medical consultation.")
with st.expander("📬 Contact"):
    st.write("For queries, email: contact@healthapp.com")
with st.expander("👨‍💻 Developer"):
    st.write("Developed by: [Your Name | Your Team | GitHub]")
