import streamlit as st
import pandas as pd
import pickle
import os

# Model file path
model_filename = '../../Models/model.pkl'

# Load model
if not os.path.exists(model_filename):
    st.error("Model file not found. Please check the path.")
else:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

# Load mean and std values
mean_std_file = '../../Models/mean_std_values.pkl'
if not os.path.exists(mean_std_file):
    st.error("Mean and std values file not found. Please check the path.")
else:
    with open(mean_std_file, 'rb') as f:
        mean_std_values = pickle.load(f)

# Main function
def main():
    st.title('Heart Disease Prediction')
    # Inputs
    age = st.slider('Age', 18, 100, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    sex_num = 1 if sex == 'Male' else 0
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    cp_num = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 250)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    fbs_num = 1 if fbs == 'True' else 0
    restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
    restecg_num = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    exang_num = 1 if exang == 'Yes' else 0
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    slope_num = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    thal_num = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)

    # Prediction
    if st.button('Predict'):
        try:
            user_input = pd.DataFrame(data={
                'age': [age],
                'sex': [sex_num],
                'cp': [cp_num],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [fbs_num],
                'restecg': [restecg_num],
                'thalach': [thalach],
                'exang': [exang_num],
                'oldpeak': [oldpeak],
                'slope': [slope_num],
                'ca': [ca],
                'thal': [thal_num]
            })

            # Standardize user input
            user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']

            # Make predictions
            prediction = model.predict(user_input)
            prediction_proba = model.predict_proba(user_input)

            # Display results
            if prediction[0] == 1:
                bg_color = 'red'
                prediction_result = 'Positive'
            else:
                bg_color = 'green'
                prediction_result = 'Negative'

            confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

            st.markdown(
                f"<div style='background-color:{bg_color}; color:white; padding:10px;'>"
                f"<h3>Prediction: {prediction_result}</h3>"
                f"<p>Confidence: {((confidence * 10000) // 1) / 100}%</p>"
                "</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
