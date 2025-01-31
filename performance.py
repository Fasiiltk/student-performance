import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb')




# Add custom CSS to change the background color
st.markdown(
    """
    <style>
    /* Change the background color of the entire app */
    .stApp {
        background-color: #87CEEB;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title and subheader
st.title('STUDENT PERFORMANCE PREDICTION ⌛')
st.subheader('Enter Your Data 📊')

# Input fields
Hours_Studied = st.number_input('Enter hours studied')
Attendance = st.number_input('Enter the attendance of the student')
Access_to_Resources_m = st.selectbox('Access to Resources', ['Low', 'Medium', 'High'])
Motivation_Level_m = st.selectbox('Motivation Level', ['Low', 'Medium', 'High'])

# Create a dictionary of input data
input_data = {
    'Hours_Studied': Hours_Studied,
    'Attendance': Attendance,
    'Access_to_Resources_m': Access_to_Resources_m,
    'Motivation_Level_m': Motivation_Level_m
}

# Convert input data to a DataFrame
new_data = pd.DataFrame([input_data])

# Load preprocessed data and ensure column matching
df = pd.read_csv('student_performance_preprocessed_data.csv')
columns_list = df.columns.to_list()

# Map categorical variables to numerical values
new_data['Access_to_Resources_m'] = new_data['Access_to_Resources_m'].map({'Low': 1, 'Medium': 2, 'High': 3})
new_data['Motivation_Level_m'] = new_data['Motivation_Level_m'].map({'Low': 1, 'Medium': 2, 'High': 3})

# Reindex the new data
df_preprocessed = new_data.reindex(columns=columns_list, fill_value=0)

# Prediction
if st.button('PREDICT'):
    prediction = model.predict(df_preprocessed)
    st.write(f'The Predicted Score Is: `{prediction[0]}`')
    if prediction[0] > 40:
        st.balloons()
