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
age = st.number_input("Age", min_value=df['Age'].min(), max_value=df['Age'].max(), value=df['Age'].median(), step=1.0)  # Map to 'Age' column
sex = st.number_input("Sex (1 = Male, 0 = Female)", min_value=df['Gender'].min(), max_value=df['Gender'].max(), value=df['Gender'].median(), step=1)  # Map to 'Gender' column
Total_Bilirubin = st.number_input("Total_Bilirubin", min_value=df['Total_Bilirubin'].min(), max_value=df['Total_Bilirubin'].max(), value=df['Total_Bilirubin'].median(), step=0.1)
Direct_Bilirubin = st.number_input("Direct_Bilirubin", min_value=df['Direct_Bilirubin'].min(), max_value=df['Direct_Bilirubin'].max(), value=df['Direct_Bilirubin'].median(), step=0.1)
Alkaline_Phosphotase = st.number_input("Alkaline_Phosphotase", min_value=df['Alkaline_Phosphotase'].min(), max_value=df['Alkaline_Phosphotase'].max(), value=df['Alkaline_Phosphotase'].median(), step=1)
Alamine_Aminotransferase = st.number_input("Alamine_Aminotransferase", min_value=df['Alamine_Aminotransferase'].min(), max_value=df['Alamine_Aminotransferase'].max(), value=df['Alamine_Aminotransferase'].median(), step=1)
Aspartate_Aminotransferase = st.number_input("Aspartate_Aminotransferase", min_value=df['Aspartate_Aminotransferase'].min(), max_value=df['Aspartate_Aminotransferase'].max(), value=df['Aspartate_Aminotransferase'].median(), step=1)
Total_Protiens = st.number_input("Total_Protiens", min_value=df['Total_Protiens'].min(), max_value=df['Total_Protiens'].max(), value=df['Total_Protiens'].median(), step=0.1)
Albumin = st.number_input("Albumin", min_value=df['Albumin'].min(), max_value=df['Albumin'].max(), value=df['Albumin'].median(), step=0.1)
Albumin_and_Globulin_Ratio = st.number_input("Albumin_and_Globulin_Ratio", min_value=df['Albumin_and_Globulin_Ratio'].min(), max_value=df['Albumin_and_Globulin_Ratio'].max(), value=df['Albumin_and_Globulin_Ratio'].median(), step=0.1)


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
