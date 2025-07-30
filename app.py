# Streamlit Interface for Salary Prediction
import streamlit as st
import pandas as pd
import pickle

# Load dataset and model (mock loading - replace with actual paths)
@st.cache_data
def load_data():
    return pd.read_csv("data/ds_salaries.csv")  # Replace with actual path

@st.cache_resource
def load_model():
    with open("salary_predictor.pkl", "rb") as file:
        return pickle.load(file)

data = load_data()
loaded_model = load_model()

st.title("💼 Employee Salary Prediction")

with st.form("prediction_form"):
    experience_level = st.selectbox("Experience Level", data['experience_level'].unique())
    employment_type = st.selectbox("Employment Type", data['employment_type'].unique())
    job_title = st.selectbox("Job Title", data['job_title'].unique())
    employee_residence = st.selectbox("Employee Residence", data['employee_residence'].unique())
    remote_ratio = st.slider("Remote Ratio", 0, 100, 50)
    company_location = st.selectbox("Company Location", data['company_location'].unique())
    company_size = st.selectbox("Company Size", data['company_size'].unique())
    work_year = st.selectbox("Work Year", sorted(data['work_year'].unique()))

    submit = st.form_submit_button("Predict Salary")

if submit:
    input_df = pd.DataFrame.from_dict([{
        'work_year': work_year,
        'experience_level': experience_level,
        'employment_type': employment_type,
        'job_title': job_title,
        'employee_residence': employee_residence,
        'remote_ratio': remote_ratio,
        'company_location': company_location,
        'company_size': company_size
    }])

    prediction = loaded_model.predict(input_df)
    st.success(f"🎯 Predicted Salary (USD): ${prediction[0]:,.2f}")
