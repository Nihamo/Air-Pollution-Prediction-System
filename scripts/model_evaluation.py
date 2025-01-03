import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

def load_data(city_file, station_file):
    try:
        city_data = pd.read_csv(city_file)
        print(f"Processed data loaded successfully from {city_file}.")
    except Exception as e:
        print(f"Error loading city data: {e}")
        return None, None

    try:
        station_data = pd.read_csv(station_file)
        print(f"Processed data loaded successfully from {station_file}.")
    except Exception as e:
        print(f"Error loading station data: {e}")
        return None, None

    return city_data, station_data

def preprocess_data(data, feature_columns, target_column):
    X = data[feature_columns].copy()
    y = data[target_column].copy()

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(exclude=['object']).columns

    # Handle missing values
    X.loc[:, numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # One-hot encode categorical features
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_columns = encoder.fit_transform(X[categorical_cols])
        encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_cols))
        X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)

    return X, y

def evaluate_model(X, y, model):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {acc}")
    print("Classification Report:")
    print(classification_report(y, y_pred))

def main():
    city_file = "C:/Air-Pollution-Prediction-System/data/processed_city_day.csv"
    station_file = "C:/Air-Pollution-Prediction-System/data/processed_station_day.csv"

    # Load data
    city_data, station_data = load_data(city_file, station_file)
    if city_data is None or station_data is None:
        return

    # Define feature columns and target column
    feature_columns_city = ['City', 'Date', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
                            'Benzene', 'Toluene', 'Xylene']
    target_column_city = 'AQI_Category'

    # Preprocess city data
    X_city, y_city = preprocess_data(city_data, feature_columns_city, target_column_city)

    # Split data
    X_train_city, X_test_city, y_train_city, y_test_city = train_test_split(X_city, y_city, test_size=0.2, random_state=42)

    # Train model for city data
    model_city = RandomForestClassifier(random_state=42)
    model_city.fit(X_train_city, y_train_city)

    # Evaluate city model
    print("Evaluating City Model...")
    evaluate_model(X_test_city, y_test_city, model_city)

    # Similar preprocessing and evaluation can be done for station data

if __name__ == "__main__":
    main()
