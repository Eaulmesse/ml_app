import streamlit as st
import joblib

# Load the model and scaler
loaded_model = joblib.load("linear_regression_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

# Streamlit app
st.title("Linear Regression Model")
st.write("Enter the number of hours studied and get the predicted test score")

# User input
hours_studied = st.number_input("Enter the number of hours studied", min_value=0, value=0)

if st.button("Predict"):
    try:
        new_data = [[hours_studied]]
        scaled_data = loaded_scaler.transform(new_data)
        prediction = loaded_model.predict(scaled_data)
        st.write(f"Predicted test score: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")