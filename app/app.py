import streamlit as st
import pandas as pd
import joblib
import numpy as np

def load_models():
    model = joblib.load('models/city_air_quality_model.pkl')
    encoder = joblib.load('models/city_encoder.pkl')
    return model, encoder

def preprocess_data(city_selected, encoder):
    input_data = pd.DataFrame([{
        'City': city_selected,
        'AQI': None,
        'CO': None,
        'Benzene': None,
    }])
    input_data['City'] = encoder.transform(input_data[['City']])
    return input_data

def align_features(input_data, model):
    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in input_data.columns:
            input_data[feature] = 0
    return input_data[model_features]

def predict_air_quality(processed_data, model):
    prediction = model.predict(processed_data)
    return prediction[0]

def main():
    st.title('Air Quality Prediction System')
    model, encoder = load_models()
    city_selected = st.selectbox('Select a City:', encoder.categories_[0])

    if st.button('Predict Air Quality'):
        try:
            preprocessed_data = preprocess_data(city_selected, encoder)
            aligned_data = align_features(preprocessed_data, model)
            prediction = predict_air_quality(aligned_data, model)
            st.subheader(f"Predicted AQI: {prediction}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
