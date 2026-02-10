import streamlit as st
import pickle
import numpy as np

# st.set_page_config(
#     page_title="Car Price Prediction",
#     layout="centered"
# )

with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🚗 Car Price Prediction Project")
st.caption("Lasso Regression Model")

st.divider()

year = st.number_input("Car Year", min_value=1990, max_value=2025, value=2015)

present_price = st.number_input(
    "Present Price (in Lakhs ₹)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=50000,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    ("Petrol", "Diesel", "CNG")
)

seller_type = st.selectbox(
    "Seller Type",
    ("Dealer", "Individual")
)

transmission = st.selectbox(
    "Transmission",
    ("Manual", "Automatic")
)

owner = st.selectbox(
    "Owner",
    (0, 1, 3)
)

st.divider()

fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
transmission_map = {"Manual": 0, "Automatic": 1}

fuel_type = fuel_map[fuel_type]
seller_type = seller_map[seller_type]
transmission = transmission_map[transmission]

if st.button("Predict Selling Price", use_container_width=True):
    input_data = np.array([[
        year,
        present_price,
        kms_driven,
        fuel_type,
        seller_type,
        transmission,
        owner
    ]])

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
