import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Paths to the model files
city_model_path = "C:/Air-Pollution-Prediction-System/models/city_air_quality_model.pkl"
station_model_path = "C:/Air-Pollution-Prediction-System/models/station_air_quality_model.pkl"

# Load the models
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)

# Function to encode categorical features
def encode_categorical_features(data, categorical_columns):
    encoder = LabelEncoder()
    for col in categorical_columns:
        if col in data.columns:
            data[col] = encoder.fit_transform(data[col])
    return data

# Inference function
def perform_inference(model, input_data, feature_names):
    print("Performing inference...")
    
    # Ensure the input data matches the model's expected feature order
    input_data = input_data[feature_names]
    
    # Encode categorical features
    categorical_columns = ['City', 'AQI_Category', 'Date']
    input_data = encode_categorical_features(input_data, categorical_columns)
    
    # Make predictions
    predictions = model.predict(input_data)
    return predictions

# Main function
def main():
    try:
        # Load city and station models
        city_model = load_model(city_model_path)
        station_model = load_model(station_model_path)
        print("Models loaded successfully.")

        # Retrieve feature names from the models
        city_feature_names = city_model.feature_names_in_
        station_feature_names = station_model.feature_names_in_

        # Example input data for testing (update as needed)
        # Ensure input features match the trained model requirements
        sample_city_data = pd.DataFrame({
            'PM2.5': [60.0],
            'PM10': [80.0],
            'NO': [12.0],
            'NO2': [20.0],
            'NOx': [25.0],
            'NH3': [5.0],
            'CO': [1.0],
            'SO2': [6.0],
            'O3': [30.0],
            'Benzene': [1.2],
            'Toluene': [0.5],
            'Xylene': [0.3],
            'AQI': [120],                # Placeholder value
            'AQI_Category': ["Moderate"], # Placeholder value
            'City': ["City_Name"],       # Placeholder value
            'Date': ["2025-01-01"]       # Placeholder value
        })

        sample_station_data = pd.DataFrame({
            'PM2.5': [70.0],
            'PM10': [90.0],
            'NO': [10.0],
            'NO2': [15.0],
            'NOx': [20.0],
            'NH3': [4.0],
            'CO': [1.1],
            'SO2': [5.0],
            'O3': [25.0],
            'Benzene': [1.0],
            'Toluene': [0.4],
            'Xylene': [0.2],
            'AQI': [140],                # Placeholder value
            'AQI_Category': ["Unhealthy"], # Placeholder value
            'City': ["Station_Name"],    # Placeholder value
            'Date': ["2025-01-01"],      # Placeholder value
            'StationId': [12345]         # Placeholder value for StationId
        })

        # Perform inference
        city_predictions = perform_inference(city_model, sample_city_data, city_feature_names)
        station_predictions = perform_inference(station_model, sample_station_data, station_feature_names)

        # Output predictions
        print("City Predictions:", city_predictions)
        print("Station Predictions:", station_predictions)

    except FileNotFoundError as e:
        print("Error:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)

if __name__ == "__main__":
    main()
