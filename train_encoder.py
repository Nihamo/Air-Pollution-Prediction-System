import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the processed city data (ensure the path is correct)
city_data = pd.read_csv('C:/Air-Pollution-Prediction-System/data/processed_city_day.csv')

# Initialize the OneHotEncoder for the 'City' column
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Changed 'sparse' to 'sparse_output'

# Train the encoder on the 'City' column
encoder.fit(city_data[['City']])

# Save the trained encoder model to a file
joblib.dump(encoder, 'C:/Air-Pollution-Prediction-System/models/city_encoder.pkl')

print("City encoder has been trained and saved successfully!")
