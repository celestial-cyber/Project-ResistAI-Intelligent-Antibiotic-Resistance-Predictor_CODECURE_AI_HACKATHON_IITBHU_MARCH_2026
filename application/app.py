
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the saved components ---
# Ensure these files are in the same directory as your Streamlit app.py
try:
    loaded_model = joblib.load('best_model.joblib')
    loaded_le_location = joblib.load('le_location.joblib')
    loaded_le_antibiotic = joblib.load('le_antibiotic.joblib')
    loaded_le_result = joblib.load('le_result.joblib')
    loaded_scaler = joblib.load('scaler.joblib')
    st.success("Model, encoders, and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Make sure 'best_model.joblib', 'le_location.joblib', 'le_antibiotic.joblib', 'le_result.joblib', and 'scaler.joblib' are in the same directory.")
    st.stop() # Stop the app if files are not found

# Define the feature names for scaling, as used during training
feature_cols = ['Location_Encoded', 'Antibiotic_Encoded']

# --- 2. Define the prediction function ---
def predict_resistance(location, antibiotic):
    """
    Predicts the resistance ('R', 'I', or 'S') for a given location and antibiotic.

    Args:
        location (str): The location string.
        antibiotic (str): The antibiotic string.

    Returns:
        str: The predicted resistance label ('R', 'I', or 'S'), or an error message if encoding fails.
    """
    try:
        # Strip whitespace from input before encoding
        encoded_location = loaded_le_location.transform([location.strip()])[0]
    except ValueError:
        return f"Error: Location '{location}' not seen during training. Please select a known location."

    try:
        # Strip whitespace from input before encoding
        encoded_antibiotic = loaded_le_antibiotic.transform([antibiotic.strip()])[0]
    except ValueError:
        return f"Error: Antibiotic '{antibiotic}' not seen during training. Please select a known antibiotic."

    # Combine encoded features into a DataFrame with feature names for correct scaling
    features_df = pd.DataFrame([[encoded_location, encoded_antibiotic]], columns=feature_cols)

    # Scale the features using the loaded scaler
    scaled_features = loaded_scaler.transform(features_df)

    # Make prediction
    prediction = loaded_model.predict(scaled_features)

    # Decode the numerical prediction back to original label
    decoded_prediction = loaded_le_result.inverse_transform(prediction)[0]

    return decoded_prediction

# --- 3. Streamlit Application Layout ---
st.title('Antibiotic Resistance Prediction App')
st.write('Select a location and an antibiotic to predict the resistance level (R/I/S).')

# Get unique values for dropdowns from the loaded LabelEncoders
locations = loaded_le_location.classes_
antibiotics = loaded_le_antibiotic.classes_

# Dropdown for Location
selected_location = st.selectbox(
    'Select Location:',
    options=locations
)

# Dropdown for Antibiotic
selected_antibiotic = st.selectbox(
    'Select Antibiotic:',
    options=antibiotics
)

# Prediction button
if st.button('Predict Resistance'):
    if selected_location and selected_antibiotic:
        # Make prediction
        result = predict_resistance(selected_location, selected_antibiotic)

        # Display the result
        if "Error" in result:
            st.error(result)
        else:
            st.subheader('Prediction Result:')
            if result == 'R':
                st.warning(f"The predicted resistance for {selected_antibiotic} at {selected_location} is: **{result} (Resistant)**")
            elif result == 'I':
                st.info(f"The predicted resistance for {selected_antibiotic} at {selected_location} is: **{result} (Intermediate)**")
            elif result == 'S':
                st.success(f"The predicted resistance for {selected_antibiotic} at {selected_location} is: **{result} (Susceptible)**")
            else:
                st.write(f"The predicted resistance for {selected_antibiotic} at {selected_location} is: **{result}**")
    else:
        st.warning("Please select both a location and an antibiotic.")

