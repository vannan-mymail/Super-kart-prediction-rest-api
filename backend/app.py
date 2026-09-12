# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model and scaler
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API


# Initialize the Flask application
superkart_predictor_api = Flask("Superkart Price Predictor")

# Load the trained machine learning model
model = joblib.load("backend_files/tuned_random_forest_model.joblib")

# Load preprocessing components
scaler = joblib.load("backend_files/scaler.joblib")
categorical_cols = joblib.load("backend_files/categorical_cols.joblib")
numerical_cols_to_scale = joblib.load("backend_files/numerical_cols_to_scale.joblib")
training_columns = joblib.load("backend_files/training_columns.joblib")

# Define the current year for Store_Age_Years calculation
current_year = 2026

# Define a route for the home page (GET request)
@superkart_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Superkart Model Prediction API!"

# Helper function to preprocess incoming data
def preprocess_data(raw_data_df):
    # Make a copy to avoid modifying the original input DataFrame
    processed_df = raw_data_df.copy()

    # 1. Feature Engineering: Product_Id_Prefix and Store_Age_Years
    # Assuming Product_Id is present in raw_data_df if needed for prefix
    if 'Product_Id' in processed_df.columns:
        processed_df['Product_Id_Prefix'] = processed_df['Product_Id'].apply(lambda x: x[:2])

    # Assuming Store_Establishment_Year is present
    if 'Store_Establishment_Year' in processed_df.columns:
        processed_df['Store_Age_Years'] = current_year - processed_df['Store_Establishment_Year']

    # 2. One-Hot Encoding for categorical features
    # Ensure only columns in categorical_cols that exist in processed_df are used
    cols_to_encode_existing = [col for col in categorical_cols if col in processed_df.columns]
    processed_df = pd.get_dummies(processed_df, columns=cols_to_encode_existing, drop_first=True)

    # Drop original 'Product_Id' and 'Store_Establishment_Year' if they exist
    processed_df = processed_df.drop(columns=['Product_Id', 'Store_Establishment_Year'], errors='ignore')

    # Drop 'Store_Id' if it exists in the input data
    processed_df = processed_df.drop(columns=['Store_Id'], errors='ignore')

    # 3. Scaling numerical features
    # Ensure only columns in numerical_cols_to_scale that exist in processed_df are used
    cols_to_scale_existing = [col for col in numerical_cols_to_scale if col in processed_df.columns]
    processed_df[cols_to_scale_existing] = scaler.transform(processed_df[cols_to_scale_existing])

    # 4. Align columns to match the training data's feature set
    # This is crucial for models that are sensitive to feature order and presence
    processed_df = processed_df.reindex(columns=training_columns, fill_value=0)

    return processed_df

# Define an endpoint for single prediction (POST request)
@superkart_predictor_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted sales as a JSON response.
    """
    # Get the JSON data from the request body
    raw_data = request.get_json()

    # Convert the raw input data to a Pandas DataFrame
    input_df = pd.DataFrame([raw_data])

    # Preprocess the input data
    processed_input_df = preprocess_data(input_df)

    # Make prediction
    predicted_sales = model.predict(processed_input_df)[0]

    # Return the predicted sales
    return jsonify({'Predicted Product Store Sales Total': round(float(predicted_sales), 2)})


# Define an endpoint for batch prediction (POST request)
@superkart_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product and store details for multiple entries
    and returns the predicted sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    raw_batch_df = pd.read_csv(file)

    # Preprocess the batch data
    processed_batch_df = preprocess_data(raw_batch_df)

    # Make predictions for all entries in the DataFrame
    predicted_sales_list = model.predict(processed_batch_df).tolist()

    # Create a dictionary of predictions with a generic index as keys
    output_dict = {i: round(float(sale), 2) for i, sale in enumerate(predicted_sales_list)}

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_predictor_api.run(debug=True)
