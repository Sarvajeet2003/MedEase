import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import requests
import warnings

warnings.filterwarnings("ignore")

# FHIR server base URL
FHIR_SERVER_URL = "http://localhost:8080/fhir"  # Replace with your FHIR server's URL

# Load prediction model
def predict_readmission(Gender, Admission_Type, Diagnosis, Num_Lab_Procedures,
                        Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits,
                        Num_Emergency_Visits, Num_Diagnoses, A1C_Result):
    with open("Readmission_Model.pkl", "rb") as m:
        model = pickle.load(m)
    
    data = np.array([[Gender, Admission_Type, Diagnosis, Num_Lab_Procedures,
                      Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits,
                      Num_Emergency_Visits, Num_Diagnoses, A1C_Result]])
    prediction = model.predict(data)
    return prediction[0]

# Fetch data from FHIR server
def fetch_patient_data(patient_id):
    try:
        # Fetch Patient resource
        patient_response = requests.get(f"{FHIR_SERVER_URL}/Patient/{patient_id}")
        patient = patient_response.json() if patient_response.status_code == 200 else None

        # Fetch related Observation resource
        observation_response = requests.get(f"{FHIR_SERVER_URL}/Observation?subject=Patient/{patient_id}")
        observations = observation_response.json()["entry"] if observation_response.status_code == 200 else []

        # Fetch related Encounter resource
        encounter_response = requests.get(f"{FHIR_SERVER_URL}/Encounter?subject=Patient/{patient_id}")
        encounters = encounter_response.json()["entry"] if encounter_response.status_code == 200 else []

        return {
            "patient": patient,
            "observations": observations,
            "encounters": encounters
        }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Display fetched data in a structured format
def display_fetched_data(fetched_data):
    st.markdown("### Fetched Data")

    # Patient Information
    patient = fetched_data.get("patient")
    if patient:
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {patient['name'][0]['given'][0]} {patient['name'][0]['family']}")
            st.write(f"**Gender:** {patient['gender'].capitalize()}")
        with col2:
            st.write(f"**Birth Date:** {patient['birthDate']}")

    # Observation Data
    observations = fetched_data.get("observations")
    if observations:
        st.subheader("Observation Data")
        for obs in observations:
            resource = obs["resource"]
            st.write(f"**Code:** {resource['code']['coding'][0]['display']}")
            st.write(f"**Value:** {resource['valueCodeableConcept']['text']}")
            st.divider()

    # Encounter Data
    encounters = fetched_data.get("encounters")
    if encounters:
        st.subheader("Encounter Data")
        encounter = encounters[0]["resource"]
        st.write(f"**Admission Type:** {encounter['class']['code'].capitalize()}")
        st.markdown("**Details:**")
        details = {
            "Lab Procedures": None,
            "Medications": None,
            "Outpatient Visits": None,
            "Inpatient Visits": None,
            "Emergency Visits": None,
            "Diagnoses Count": None
        }
        for ext in encounter.get("extension", []):
            key = ext["url"].split("/")[-1]
            details[key] = ext["valueInteger"]

        for key, value in details.items():
            if value is not None:
                st.write(f"- **{key.replace('_', ' ').capitalize()}:** {value}")

# Predict and display the result
def populate_and_predict(fetched_data):
    patient = fetched_data.get("patient")
    observations = fetched_data.get("observations")
    encounters = fetched_data.get("encounters")

    if not patient or not observations or not encounters:
        st.error("Incomplete data fetched. Please ensure all required fields are present in the FHIR server.")
        return

    # Gender
    gender_map = {"female": 0, "male": 1, "other": 2}
    Gender = gender_map.get(patient["gender"], 2)

    # A1C Result
    a1c_result = next((obs for obs in observations if obs["resource"]["code"]["coding"][0]["code"] == "4548-4"), None)
    if not a1c_result:
        st.error("A1C observation not found for this patient.")
        return
    A1C_Result = 1 if a1c_result["resource"]["valueCodeableConcept"]["text"] == "Normal" else 0

    # Encounter details
    encounter = encounters[0]["resource"]
    admission_type_map = {"emergency": 1, "urgent": 2, "elective": 0}
    Admission_Type = admission_type_map.get(encounter["class"]["code"], 0)

    # Extensions
    extensions = {ext["url"].split("/")[-1]: ext["valueInteger"] for ext in encounter.get("extension", [])}
    Num_Lab_Procedures = extensions.get("labProcedures", 1)
    Num_Medications = extensions.get("medications", 1)
    Num_Outpatient_Visits = extensions.get("outpatientVisits", 0)
    Num_Inpatient_Visits = extensions.get("inpatientVisits", 0)
    Num_Emergency_Visits = extensions.get("emergencyVisits", 0)
    Num_Diagnoses = extensions.get("diagnosesCount", 1)

    # Prediction
    result = predict_readmission(Gender, Admission_Type, 1,  # Assuming Diagnosis = 1 for simplicity
                                 Num_Lab_Procedures, Num_Medications, Num_Outpatient_Visits,
                                 Num_Inpatient_Visits, Num_Emergency_Visits, Num_Diagnoses, A1C_Result)

    st.subheader("Prediction Result:")
    if result == 1:
        st.success("## :red[Readmission is Required]")
    else:
        st.success("## :green[Readmission is Not Required]")

# Streamlit app layout
st.set_page_config(page_title="Predicting Hospital Readmissions",
                   layout="wide",
                   menu_items={'About': "### This page is created by Desilva!"})

st.markdown("<h1 style='text-align: center; color: #fa6607;'>Predicting Hospital Readmissions</h1>", unsafe_allow_html=True)
st.write("")

select = option_menu(None, ["Home", "Readmission"], 
                     icons=["hospital-fill", "ticket-detailed"], orientation="horizontal",
                     styles={"container": {"padding": "0!important", "background-color": "#fafafa"},
                             "icon": {"color": "#fdfcfb", "font-size": "20px"},
                             "nav-link": {"font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
                             "nav-link-selected": {"background-color": "#fa6607"}})

if select == "Home":
    st.title("Welcome to the Hospital Readmissions Prediction Project!")
    st.write("This project predicts hospital readmissions using patient data fetched from a FHIR server.")

elif select == "Readmission":
    st.write("Enter a Patient ID to fetch data and predict readmission.")
    patient_id = st.text_input("Enter Patient ID:")
    if st.button("Fetch and Predict"):
        if patient_id:
            fetched_data = fetch_patient_data(patient_id)
            if fetched_data:
                display_fetched_data(fetched_data)
                populate_and_predict(fetched_data)
            else:
                st.error("No data found for the entered Patient ID.")
        else:
            st.error("Please enter a Patient ID.")
