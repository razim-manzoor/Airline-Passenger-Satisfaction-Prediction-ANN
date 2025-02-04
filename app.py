# Import necessary libraries
import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Load the preprocessing pipeline and model
preprocessor = joblib.load('preprocessor.joblib')
model = load_model('passenger_satisfaction_model.h5')

# Function to preprocess input data and make predictions
def predict_satisfaction(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply preprocessing
    processed_data = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(processed_data)
    return "Satisfied" if prediction > 0.5 else "Neutral or Dissatisfied"

# Streamlit app title
st.title("Airline Passenger Satisfaction Prediction")

# Sidebar for user inputs
st.sidebar.header("Passenger Details")

# Input fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
customer_type = st.sidebar.selectbox("Customer Type", ["First-time", "Returning"])
travel_type = st.sidebar.selectbox("Type of Travel", ["Personal", "Business"])
travel_class = st.sidebar.selectbox("Class", ["Economy", "Business"])
flight_distance = st.sidebar.number_input("Flight Distance", min_value=0, value=500)
departure_delay = st.sidebar.number_input("Departure Delay (minutes)", min_value=0, value=5)
arrival_delay = st.sidebar.number_input("Arrival Delay (minutes)", min_value=0, value=10)

# Ratings (sliders from 1 to 5)
time_convenience = st.sidebar.slider("Departure and Arrival Time Convenience", 1, 5, 3)
online_booking = st.sidebar.slider("Ease of Online Booking", 1, 5, 3)
checkin_service = st.sidebar.slider("Check-in Service", 1, 5, 3)
online_boarding = st.sidebar.slider("Online Boarding", 1, 5, 3)
gate_location = st.sidebar.slider("Gate Location", 1, 5, 3)
onboard_service = st.sidebar.slider("On-board Service", 1, 5, 3)
seat_comfort = st.sidebar.slider("Seat Comfort", 1, 5, 3)
leg_room = st.sidebar.slider("Leg Room Service", 1, 5, 3)
cleanliness = st.sidebar.slider("Cleanliness", 1, 5, 3)
food_drink = st.sidebar.slider("Food and Drink", 1, 5, 3)
inflight_service = st.sidebar.slider("In-flight Service", 1, 5, 3)
wifi_service = st.sidebar.slider("In-flight Wifi Service", 1, 5, 3)
entertainment = st.sidebar.slider("In-flight Entertainment", 1, 5, 3)
baggage_handling = st.sidebar.slider("Baggage Handling", 1, 5, 3)

# Predict button
if st.sidebar.button("Predict Satisfaction"):
    # Collect input data
    input_data = {
        'Gender': gender,
        'Age': age,
        'Customer Type': customer_type,
        'Type of Travel': travel_type,
        'Class': travel_class,
        'Flight Distance': flight_distance,
        'Departure Delay': departure_delay,
        'Arrival Delay': arrival_delay,
        'Departure and Arrival Time Convenience': time_convenience,
        'Ease of Online Booking': online_booking,
        'Check-in Service': checkin_service,
        'Online Boarding': online_boarding,
        'Gate Location': gate_location,
        'On-board Service': onboard_service,
        'Seat Comfort': seat_comfort,
        'Leg Room Service': leg_room,
        'Cleanliness': cleanliness,
        'Food and Drink': food_drink,
        'In-flight Service': inflight_service,
        'In-flight Wifi Service': wifi_service,
        'In-flight Entertainment': entertainment,
        'Baggage Handling': baggage_handling
    }
    
    # Make prediction
    result = predict_satisfaction(input_data)
    
    # Display result
    st.subheader("Prediction Result:")
    if result == "Satisfied":
        st.success("The passenger is likely to be **Satisfied** with their experience.")
    else:
        st.error("The passenger is likely to be **Neutral or Dissatisfied** with their experience.")


# Add a footer
st.markdown("""
---
**Developed by:** [Razim Manzoor](https://www.linkedin.com/in/razim-manzoor)
            
**GitHub Repository:** [Link](https://github.com/razim-manzoor/Airline-Passenger-Satisfaction-Prediction-ANN)
            
**Contact:** [Email to: manzoorrazim@gmail.com](mailto:manzoorrazim@gmail.com)
""")