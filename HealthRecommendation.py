import openai
import pandas as pd
import pandas as pd
from fuzzywuzzy import fuzz

def filter_diseases(disease_name, df, max_distance=1):
    def is_within_distance(disease):
        return fuzz.ratio(disease_name.lower(), disease.lower()) >= (100 - max_distance * 100 / len(disease_name))
    
    return df[df['disease'].apply(is_within_distance)]

def generate_summary_and_tips_with_openai(patient_summary,disease):
    
    # Initialize the OpenAI client
    openai.api_key = 'sk-Rb463GMh6YnDAUErt64cP-oTFxYC6Jdb7QPJvVXD0UT3BlbkFJ09UT4GmnCHPPW0KZw3RpAj1wC5EnXdybX3r6KcEAoA'
    data = pd.read_csv("healifyLLM_answer_dataset.csv")
    filtered_rows = filter_diseases(disease.lower(), data)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical assistant AI. Your task is to analyze the provided patient health summary "
                "and recommend detailed, actionable health tips for better management of their condition."
            ),
        },
        {
            "role": "user",
            "content": f"Patient Health Summary:\n{patient_summary}\n\n  Information about the disease person had - {filtered_rows}. He has been recently released from this hopsital after this diseas. Provide an analysis and recommendations. Dont give medicine names to avoid legal issues.",
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= messages
    )

    return response['choices'][0]['message']['content']

print(generate_summary_and_tips_with_openai("Name:Abhinav, Gender:Male,Age:20","asthma"))