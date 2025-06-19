import streamlit as st
import numpy as np
import pickle

# Load pre-trained models and scalers
with open('DT.pkl', 'rb') as f:
    crop_model = pickle.load(f)
with open('scc.pkl', 'rb') as f:
    crop_scaler = pickle.load(f)
with open('model.pkl', 'rb') as f:
    fertilizer_model = pickle.load(f)
with open('sc.pkl', 'rb') as f:
    fertilizer_scaler = pickle.load(f)

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Soil dictionary
soil_dict = {
    1: "Sandy", 2: "Loamy", 3: "Black", 4: "Red", 5: "Clayey"
}

# Crop type dictionary
crop_type_dict = {
    1: "Maize", 2: "Sugarcane", 3: "Cotton", 4: "Tobacco", 5: "Paddy",
    6: "Barley", 7: "Wheat", 8: "Millets", 9: "Oil seeds", 10: "Pulses"
}

# Fertilizer dictionary
fert_dict = {
    1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28', 5: '17-17-17', 6: '20-20', 7: '10-26-26'
}

# Streamlit UI
st.title("Crop and Fertilizer Recommendation System")
option = st.sidebar.selectbox("Choose an option", ["Crop Recommendation", "Fertilizer Recommendation"])

if option == "Crop Recommendation":
    st.header("Crop Recommendation System")
    N = st.number_input("Nitrogen", min_value=0, max_value=100, value=50)
    P = st.number_input("Phosphorus", min_value=0, max_value=100, value=50)
    K = st.number_input("Potassium", min_value=0, max_value=100, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

    if st.button("Predict Crop"):
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        transformed_features = crop_scaler.transform(features)
        prediction = crop_model.predict(transformed_features)[0]
        st.success(f"{crop_dict[prediction]} is the best crop to be cultivated.")

elif option == "Fertilizer Recommendation":
    st.header("Fertilizer Recommendation System")
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity", min_value=0.0, max_value=1.0, value=0.5)
    moisture = st.number_input("Moisture", min_value=0.0, max_value=1.0, value=0.5)
    soil_type = st.selectbox("Soil Type", list(soil_dict.values()))
    crop_type = st.selectbox("Crop Type", list(crop_type_dict.values()))
    soil_type = list(soil_dict.keys())[list(soil_dict.values()).index(soil_type)]
    crop_type = list(crop_type_dict.keys())[list(crop_type_dict.values()).index(crop_type)]
    nitrogen = st.number_input("Nitrogen", min_value=0, max_value=100, value=10)
    potassium = st.number_input("Potassium", min_value=0, max_value=100, value=10)
    phosphorus = st.number_input("Phosphorus", min_value=0, max_value=100, value=10)

    if st.button("Recommend Fertilizer"):
        features = np.array([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorus]])
        transformed_features = fertilizer_scaler.transform(features)
        prediction = fertilizer_model.predict(transformed_features)[0]
        st.success(f"{fert_dict[prediction]} is the best fertilizer for the given conditions.")
