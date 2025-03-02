
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')
df = pd.read_csv('indian_liver_patient.csv')
# Define the prediction function
def predict_diabetes(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = np.max(classifier.predict_proba(scaled_data)[0])
    return pred,prob

# Streamlit UI components
st.title("Diabetes Prediction")


# Input fields with updated variable names and mapping to dataframe columns
age = st.number_input("Age", min_value=4, max_value=90, value=4, step=1)  # Map to 'Age' column
gender = st.number_input("Gender' (1 = Male, 0 = Female)", min_value=0, max_value=1, value=0, step=1)  # Map to 'Gender' column
Total_Bilirubin = st.number_input("Total_Bilirubin", min_value=0.4, max_value=70.0, value=0.4, step=0.1)
Direct_Bilirubin = st.number_input("Direct_Bilirubin", min_value=0.1, max_value=19.7, value=0.1, step=0.1)
Alkaline_Phosphotase = st.number_input("Alkaline_Phosphotase", min_value=63, max_value=2110, value=63, step=1)
Alamine_Aminotransferase = st.number_input("Alamine_Aminotransferase", min_value=10, max_value=2000, value=10, step=1)
Aspartate_Aminotransferase = st.number_input("Aspartate_Aminotransferase", min_value=10, max_value=4929, value=10, step=1)
Total_Protiens = st.number_input("Total_Protiens", min_value=2.7, max_value=9.6, value=2.7, step=0.1)
Albumin = st.number_input("Albumin", min_value=0.9, max_value=5.5, value=0.9, step=0.1)
Albumin_and_Globulin_Ratio = st.number_input("Albumin_and_Globulin_Ratio", min_value=0.3, max_value=2.8, value=0.3, step=0.1)


# Create the input dictionary for prediction
input_data = {
  'Age': 65,
 'Gender': 1,
 'Total_Bilirubin	': 0.7,
 'Direct_Bilirubin': 0.1,
 'Alkaline_Phosphotase': 187,
 'Alamine_Aminotransferase': 16,
 'Aspartate_Aminotransferase': 18,
 'Total_Protiens':6.8,
 'Albumin':3.3,   
 'Albumin_and_Globulin_Ratio':0.90,
   }

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_diabetes(input_data)

        if pred == 1:
            st.error(f"Prediction: Diabetes with probability {prob:.2f}")
        else:
            st.success(f"Prediction: No Diabetes with probability {prob:.2f}")
