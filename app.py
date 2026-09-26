
import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------------------
# Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Wellness Tourism Package Predictor")
st.write(
    "Enter the customer details below to predict "
    "whether the customer is likely to purchase a package."
)

# -----------------------------------------
# Load trained model
# -----------------------------------------

import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------------------
# Customer inputs
# -----------------------------------------

st.header("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Inquiry"]
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2
    )

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

with col3:
    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=50,
        value=3
    )

    passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    number_of_children = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

st.header("Sales Interaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    designation = st.selectbox(
        "Designation",
        ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=25000
    )

with col2:
    pitch_satisfaction = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )

with col3:
    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0,
        max_value=20,
        value=3
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        max_value=100,
        value=15
    )

# -----------------------------------------
# Create input dataframe
# -----------------------------------------

input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income],
    "PitchSatisfactionScore": [pitch_satisfaction],
    "ProductPitched": [product_pitched],
    "NumberOfFollowups": [number_of_followups],
    "DurationOfPitch": [duration_of_pitch]
})

# -----------------------------------------
# Prediction
# -----------------------------------------

if st.button("Predict Purchase", type="primary"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            f"Customer is predicted to purchase the package. "
            f"Estimated probability: {probability:.2%}"
        )
    else:
        st.info(
            f"Customer is predicted not to purchase the package. "
            f"Estimated probability: {probability:.2%}"
        )
