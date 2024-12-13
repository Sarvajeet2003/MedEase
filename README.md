# Predicting Hospital Readmissions

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Solution Steps](#solution-steps)
4. [Workflow](#workflow)
5. [Data Description](#data-description)
6. [Challenges](#challenges)
7. [Technologies Used](#technologies-used)
9. [How to Run](#how-to-run)

## Project Overview

The primary goal of this project is to build a predictive model that can identify patients who are at high risk of hospital readmission within 30 days after their initial discharge. This predictive model will assist healthcare providers in proactively intervening and providing targeted interventions to reduce readmission rates and improve patient outcomes.

## Problem Statement

The healthcare industry faces challenges in predicting hospital readmissions accurately, which can lead to increased healthcare costs and reduced patient satisfaction. This project addresses this issue by leveraging machine learning techniques to develop a predictive model based on patient data to identify individuals at high risk of hospital readmission.

## Solution Steps

1. **Data Preprocessing:** Clean and prepare the healthcare data, handling missing values, encoding categorical variables, and ensuring data quality and consistency.

2. **Feature Engineering:** Create relevant features from the available data, including patient demographics, medical history, previous hospitalizations, and other clinically relevant factors.

3. **Model Building:** Develop a machine learning or statistical model capable of predicting the likelihood of hospital readmission within 30 days. 

4. **Model Evaluation:** Assess the performance of the predictive model using standard binary classification evaluation metrics such as accuracy, precision, recall, F1-score, ROC curve, and AUC.

## Workflow

![Hospital](https://github.com/asdesilva3/Predicting-Hospital-Readmissions/assets/148002331/d3a2f1e6-ecaa-4b90-9d1b-a1e5e81e722d)



## Data Description

The dataset contains the following columns:

- **Patient_ID**: Unique identifier for each patient.
- **Age**: Age of the patient in years.
- **Gender**: Gender of the patient (e.g., Male, Female, Other).
- **Admission_Type**: Type of admission (e.g., Emergency, Urgent, Elective).
- **Diagnosis**: Primary diagnosis of the patient upon admission (e.g., Heart Disease, Diabetes, Injury, Infection).
- **Num_Lab_Procedures**: Number of laboratory procedures performed during the hospital stay.
- **Num_Medications**: Number of medications prescribed to the patient.
- **Num_Outpatient_Visits**: Number of outpatient visits prior to the current hospital admission.
- **Num_Inpatient_Visits**: Number of inpatient visits prior to the current hospital admission.
- **Num_Emergency_Visits**: Number of emergency room visits prior to the current hospital admission.
- **Num_Diagnoses**: Number of diagnoses.
- **A1C_Result**: Result of the A1C test, if available (e.g., Normal, Abnormal).
- **Readmitted**: Indicates whether the patient was readmitted to the hospital within a certain time frame (e.g., Yes, No).


## Challenges

One specific challenge encountered in this project was the presence of many null values in the `A1C_Result` column. To address this, a classification model was developed to predict and fill in the missing values in the `A1C_Result` column based on other available features. Once the missing values were imputed, a predictive model for hospital readmission status was built using the completed dataset.


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Streamlit


## How to Run

1. Install the required dependencies using `pip install -r requirements.txt`

2. Run the Streamlit application using `streamlit run app.py`
