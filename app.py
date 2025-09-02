import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("heart_knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction Model", layout="centered")

st.title("❤️ Heart Disease Prediction Model")

st.write("Fill in the details below to check the risk of heart disease.")

# User inputs
age = st.slider("Age", 20, 100, 30)
sex = st.radio("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalch = st.slider("Maximum Heart Rate Achieved", 70, 220, 150)
exang = st.radio("Exercise Induced Angina (exang)", ("Yes", "No"))
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise (slope)", [0, 1, 2])
ca = st.slider("Number of Major Vessels (ca)", 0, 4, 0)
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert inputs
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalch, exang, oldpeak, slope, ca, thal]])

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease!")
    else:
        st.success("✅ No significant risk detected.")
