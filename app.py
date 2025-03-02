
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_liver_disease(input_data):
    sample_data = pd.DataFrame([input_data])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = np.max(classifier.predict_proba(scaled_data)[0])
    return pred, prob

# Streamlit UI components
st.title("Liver Disease Prediction")

# Input fields with updated variable names and mapping to dataframe columns
age = st.number_input("Age", min_value=4, max_value=90, value=4, step=1)  # Map to 'Age' column
gender = st.number_input("Gender (1 = Male, 0 = Female)", min_value=0, max_value=1, value=0, step=1)  # Map to 'Gender' column
total_bilirubin = st.number_input("Total Bilirubin", min_value=0.4, max_value=70.0, value=0.4, step=0.1)
direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.1, max_value=19.7, value=0.1, step=0.1)
alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=63, max_value=2110, value=63, step=1)
alanine_aminotransferase = st.number_input("Alanine Aminotransferase", min_value=10, max_value=2000, value=10, step=1)
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=10, max_value=4929, value=10, step=1)
total_proteins = st.number_input("Total Proteins", min_value=2.7, max_value=9.6, value=2.7, step=0.1)
albumin = st.number_input("Albumin", min_value=0.9, max_value=5.5, value=0.9, step=0.1)
albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.3, max_value=2.8, value=0.3, step=0.1)

# Create the input dictionary for prediction
input_data = {
    'Age': age,
    'Gender': gender,
    'Total_Bilirubin': total_bilirubin,
    'Direct_Bilirubin': direct_bilirubin,
    'Alkaline_Phosphotase': alkaline_phosphotase,
    'Alamine_Aminotransferase': alanine_aminotransferase,
    'Aspartate_Aminotransferase': aspartate_aminotransferase,
    'Total_Protiens': total_proteins,
    'Albumin': albumin,
    'Albumin_and_Globulin_Ratio': albumin_and_globulin_ratio,
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_liver_disease(input_data)

        if pred == 1:
            st.error(f"Prediction: Liver Disease with probability {prob:.2f}")
        else:
            st.success(f"Prediction: No Liver Disease with probability {prob:.2f}")
