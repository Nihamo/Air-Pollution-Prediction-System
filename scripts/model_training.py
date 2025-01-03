import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load processed data
def load_processed_data():
    print("Loading processed data...")
    city_data = pd.read_csv('C:/Air-Pollution-Prediction-System/data/processed_city_day.csv')
    station_data = pd.read_csv('C:/Air-Pollution-Prediction-System/data/processed_station_day.csv')
    print("Processed data loaded successfully.")
    return city_data, station_data

# Preprocess data
# Preprocess data
def preprocess_data(data, target_column):
    print("Preprocessing data...")
    
    # Encode categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column != target_column:
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column].astype(str))
    
    # Fill missing values separately for numeric and categorical columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        mode_value = data[column].mode()[0]
        data[column] = data[column].fillna(mode_value)

    print("Data preprocessed successfully.")
    return data, label_encoders


# Train the model
def train_model(data, target_column, model_path):
    print("Training model...")
    # Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model trained and saved at {model_path}")
    return X_test, y_test, model

def main():
    # Load and preprocess data
    city_data, station_data = load_processed_data()
    target_column = 'AQI_Bucket'
    model_path_city = 'C:/Air-Pollution-Prediction-System/models/city_air_quality_model.pkl'
    model_path_station = 'C:/Air-Pollution-Prediction-System/models/station_air_quality_model.pkl'
    
    # Preprocess city data
    print("Processing City Data...")
    city_data, city_label_encoders = preprocess_data(city_data, target_column)
    # Train model for city data
    X_test_city, y_test_city, city_model = train_model(city_data, target_column, model_path_city)
    
    # Preprocess station data
    print("Processing Station Data...")
    station_data, station_label_encoders = preprocess_data(station_data, target_column)
    # Train model for station data
    X_test_station, y_test_station, station_model = train_model(station_data, target_column, model_path_station)

if __name__ == "__main__":
    main()
