import streamlit as st
import pandas as pd
import requests
import json

# --- Configuration --- #
# Replace with your actual Codespace forwarded URL for port 7860
# Example: "https://organic-space-abcd1234-7860.app.github.dev"
model_root_url = "_____"

# --- API Endpoints --- #
predict_url = f"{model_root_url}/v1/predict"
predict_batch_url = f"{model_root_url}/v1/predictbatch"

# --- Streamlit UI --- #
st.set_page_config(page_title="Superkart Sales Predictor", layout="wide")
st.title("🛒 Superkart Product Sales Predictor")

st.markdown("Predict the total sales for a product in a given store.")

st.sidebar.header("Configuration")
model_root_url_input = st.sidebar.text_input("Backend API URL", value=model_root_url, key="model_url_input")
if model_root_url_input and model_root_url_input != "_____":
    model_root_url = model_root_url_input
    predict_url = f"{model_root_url}/v1/predict"
    predict_batch_url = f"{model_root_url}/v1/predictbatch"
    st.sidebar.success("API URL updated!")
elif model_root_url_input == "_____":
    st.sidebar.warning("Please update the Backend API URL in the sidebar for predictions to work.")

# --- Single Prediction Tab --- #
st.header("Single Product Sales Prediction")

with st.form("single_prediction_form"):
    st.subheader("Product Details")
    product_id = st.text_input("Product ID (e.g., FD6114, NC1180)", value="FD6114")
    product_weight = st.number_input("Product Weight", value=12.66, min_value=0.1, max_value=100.0, step=0.1)
    product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar', 'reg'])
    product_allocated_area = st.number_input("Product Allocated Area (ratio)", value=0.027, min_value=0.001, max_value=1.0, step=0.001, format="%.3f")
    product_type = st.selectbox("Product Type", [
        'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Snack Foods',
        'Household', 'Fruits and Vegetables', 'Breakfast', 'Hard Drinks', 'Breads', 'Meat',
        'Soft Drinks', 'Others', 'Starchy Foods', 'Seafood'
    ])
    product_mrp = st.number_input("Product MRP", value=117.08, min_value=10.0, max_value=300.0, step=0.01)

    st.subheader("Store Details")
    # Store_Id is dropped before model training and should not be an input
    store_establishment_year = st.number_input("Store Establishment Year", value=2009, min_value=1900, max_value=2025, step=1)
    store_size = st.selectbox("Store Size", ['Medium', 'High', 'Small'])
    store_location_city_type = st.selectbox("Store Location City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
    store_type = st.selectbox("Store Type", ['Supermarket Type1', 'Departmental Store', 'Supermarket Type2', 'Food Mart'])

    submitted = st.form_submit_button("Predict Sales")
    if submitted:
        if model_root_url == "_____":
            st.error("Please set the Backend API URL in the sidebar before making predictions.")
        else:
            payload = {
                "Product_Id": product_id,
                "Product_Weight": product_weight,
                "Product_Sugar_Content": product_sugar_content,
                "Product_Allocated_Area": product_allocated_area,
                "Product_Type": product_type,
                "Product_MRP": product_mrp,
                "Store_Establishment_Year": store_establishment_year,
                "Store_Size": store_size,
                "Store_Location_City_Type": store_location_city_type,
                "Store_Type": store_type
            }
            try:
                response = requests.post(predict_url, json=payload)
                if response.status_code == 200:
                    prediction = response.json()
                    st.success(f"Predicted Product Store Sales Total: **${prediction['Predicted Product Store Sales Total']:.2f}**")
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API. Please ensure the URL is correct and the API is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- Batch Prediction Tab --- #
st.header("Batch Product Sales Prediction")
st.write("Upload a CSV file containing multiple product and store entries for batch prediction.")
st.info("The CSV file should contain the following columns: `Product_Id`, `Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, `Product_Type`, `Product_MRP`, `Store_Establishment_Year`, `Store_Size`, `Store_Location_City_Type`, `Store_Type`.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    if model_root_url == "_____":
        st.error("Please set the Backend API URL in the sidebar before making predictions.")
    else:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head())

            # Convert DataFrame to CSV bytes for the API request
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            files = {'file': ('batch_data.csv', csv_data, 'text/csv')}

            response = requests.post(predict_batch_url, files=files)

            if response.status_code == 200:
                predictions = response.json()
                st.success("Batch predictions received!")

                # Convert dictionary of predictions to a DataFrame for better display
                predictions_df = pd.DataFrame(list(predictions.items()), columns=['Entry Index', 'Predicted Sales'])
                st.dataframe(predictions_df)

            else:
                st.error(f"Error from API: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Please ensure the URL is correct and the API is running.")
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
        except Exception as e:
            st.error(f"An unexpected error occurred during batch prediction: {e}")
