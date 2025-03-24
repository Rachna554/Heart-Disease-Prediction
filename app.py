import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os
import base64

# Set page config with custom icon and wide layout
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Add custom CSS for background and styling
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as file:
            img_data = file.read()
        b64_img = base64.b64encode(img_data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{b64_img}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Background image not found. Using default styling.")

add_bg_from_local('background.jpg')  # Ensure the file exists in your working directory

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Heart Disease Prediction"])

if page == "Home":
    # Home page content
    st.title("Welcome to the Heart Disease Prediction System")
    st.write("""
        This application uses machine learning to predict the likelihood of heart disease based on user-provided data.
        
        ### How It Works
        - Enter patient information under the **Heart Disease Prediction** section.
        - The application uses a trained XGBoost model to analyze the data and predict the likelihood of heart disease.
        - This system is built to assist medical professionals and individuals in early risk detection, though it should not replace professional medical advice.
        
        ### Disclaimer
        This tool provides an estimation based on patterns from historical data and is not a substitute for clinical diagnosis. Please consult a healthcare provider for any health concerns.
    """)
    
elif page == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction System")
    st.write("This application predicts the presence of heart disease based on patient data.")

    # Input fields
    age = st.number_input("🔢 Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("🚻 Sex", ["Male", "Female"])
    cp = st.selectbox("💔 Chest Pain Type", 
                    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("💉 Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
    chol = st.number_input("🍔 Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("🩺 Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("🏃‍♂️ Maximum Heart Rate", min_value=70, max_value=220, value=150)
    exang = st.selectbox("🚶 Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=6.2, value=0.0)
    slope = st.selectbox("🧭 Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Function to preprocess input data
    def preprocess_input():
        # Encoding categorical features
        sex_n = 1 if sex == "Male" else 0
        cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
        cp_n = cp_map[cp]
        fbs_n = 1 if fbs == "Yes" else 0
        restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        restecg_n = restecg_map[restecg]
        exang_n = 1 if exang == "Yes" else 0
        slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
        slope_n = slope_map[slope]
        
        # Input data array
        input_data = np.array([[age, sex_n, cp_n, trestbps, chol, fbs_n, restecg_n, thalach, exang_n, oldpeak, slope_n]])
        
        # Scale the input data using pre-fitted StandardScaler
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_data)  # Replace fit_transform with transform for production
        
        return input_scaled

    # Prediction button
    if st.button("🔍 Predict"):
        try:
            # Load the trained XGBoost model
            with open("xgboost_model.pkl", "rb") as file:
                model = pickle.load(file)

            # Preprocess input data
            input_scaled = preprocess_input()

            # Make prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Display result
            st.header("🩺 Prediction Result")
            if prediction[0] == 0:
                st.success("✅ No Heart Disease Detected")
                st.write(f"Probability of Heart Disease: {probability:.2%}")
            else:
                st.error("⚠️ Heart Disease Detected")
                st.write(f"Probability of Heart Disease: {probability:.2%}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
