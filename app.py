import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("safety_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🔒 Women Safety Prediction Dashboard")

st.write("Enter the details of a place to check if it is **Safe or Unsafe**.")

# User inputs
location = st.selectbox("📍 Location Type", ["Street", "Market", "Bus_Stop", "Office", "Home", "Park", "Public_Transport", "Mall", "Highway", "School"])
time_of_day = st.selectbox("⏰ Time of Day", ["Morning", "Afternoon", "Evening", "Night", "Late_Night"])
crowd = st.selectbox("👥 Crowd Density", ["Low", "Medium", "High"])
lighting = st.selectbox("💡 Lighting", ["Well_Lit", "Dim", "Dark"])
police = st.selectbox("👮 Police Presence", ["Yes", "No"])
emergency = st.selectbox("🚑 Emergency Access", ["Easy", "Moderate", "Hard"])
past_incidents = st.slider("📊 Past Incidents in Area", 0, 5, 1)
weather = st.selectbox("🌦️ Weather", ["Clear", "Rainy", "Foggy", "Stormy"])

# Prepare input for model
input_data = {
    "Location_Type": location,
    "Time_of_Day": time_of_day,
    "Crowd_Density": crowd,
    "Lighting": lighting,
    "Police_Presence": police,
    "Emergency_Services_Access": emergency,
    "Past_Incidents": past_incidents,
    "Weather": weather
}

# Convert categorical to numeric using saved encoders
for col in input_data:
    if col in label_encoders:  # apply encoder if available
        input_data[col] = label_encoders[col].transform([input_data[col]])[0]

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("🔍 Predict Safety"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ This place is predicted as **SAFE**.")
    else:
        st.error("⚠️ This place is predicted as **UNSAFE**.")

