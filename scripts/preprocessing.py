import pandas as pd
import os

def load_data():
    print("Loading datasets...")
    city_day_data = pd.read_csv('C:/Air-Pollution-Prediction-System/data/city_day.csv')
    station_day_data = pd.read_csv('C:/Air-Pollution-Prediction-System/data/station_day.csv')
    print("Datasets loaded successfully.")
    return city_day_data, station_day_data

def preprocess_city_day_data(city_day_data):
    print("Preprocessing city_day dataset...")
    city_day_data.dropna(subset=['AQI'], inplace=True)
    city_day_data.fillna(0, inplace=True)
    city_day_data['AQI_Category'] = pd.cut(
        city_day_data['AQI'],
        bins=[0, 50, 100, 200, 300, 400, 500, float('inf')],
        labels=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe', 'Hazardous'],
        include_lowest=True
    )
    print("city_day dataset preprocessed successfully.")
    return city_day_data

def preprocess_station_day_data(station_day_data):
    print("Preprocessing station_day dataset...")
    station_day_data.dropna(subset=['AQI'], inplace=True)
    station_day_data.fillna(0, inplace=True)
    station_day_data['AQI_Category'] = pd.cut(
        station_day_data['AQI'],
        bins=[0, 50, 100, 200, 300, 400, 500, float('inf')],
        labels=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe', 'Hazardous'],
        include_lowest=True
    )
    print("station_day dataset preprocessed successfully.")
    return station_day_data

def save_processed_data(city_day_data, station_day_data):
    print("Saving processed data...")
    city_processed_path = 'C:/Air-Pollution-Prediction-System/data/processed_city_day.csv'
    station_processed_path = 'C:/Air-Pollution-Prediction-System/data/processed_station_day.csv'
    os.makedirs(os.path.dirname(city_processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(station_processed_path), exist_ok=True)

    city_day_data.to_csv(city_processed_path, index=False)
    station_day_data.to_csv(station_processed_path, index=False)
    print(f"Processed city_day data saved to {city_processed_path}.")
    print(f"Processed station_day data saved to {station_processed_path}.")

def main():
    city_day_data, station_day_data = load_data()
    processed_city_day_data = preprocess_city_day_data(city_day_data)
    processed_station_day_data = preprocess_station_day_data(station_day_data)
    save_processed_data(processed_city_day_data, processed_station_day_data)

if __name__ == "__main__":
    main()
