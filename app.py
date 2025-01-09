import streamlit as st
import pandas as pd
import joblib
import time
import os
   
# Load the trained model
model = joblib.load('C:/Users/souha/Downloads/model/model.pkl')

# Define the selected features
FEATURES = ['amt', 'merchant_freq', 'state_freq', 'amt_merchant_freq', 'amt_state_freq', 'category']

# Real-time dataset file
DATASET_PATH = 'C:/Users/souha/Downloads/fraudTest.csv'


# Output file for detected fraud
FRAUD_LOG_FILE = 'C:/Users/souha/Downloads/model/detected_frauds.csv'

# Initialize a set to track processed transactions
processed_transactions = set()


# Function to load and preprocess new data
@st.cache_data
def load_data():
    data = pd.read_csv(DATASET_PATH)
    
    # Feature Engineering
    merchant_counts = data['merchant'].value_counts()
    data['merchant_freq'] = data['merchant'].map(merchant_counts)
    
    state_counts = data['state'].value_counts()
    data['state_freq'] = data['state'].map(state_counts)
    
    data['amt_merchant_freq'] = data['amt'] * data['merchant_freq']
    data['amt_state_freq'] = data['amt'] * data['state_freq']
    
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le = LabelEncoder()
    data['merchant'] = le.fit_transform(data['merchant'])
    data['state'] = le.fit_transform(data['state'])
    data['category'] = le.fit_transform(data['category'])
    scaler = StandardScaler()
    numeric_features = ['amt', 'merchant_freq', 'state_freq', 'amt_merchant_freq', 'amt_state_freq','category']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    return data[FEATURES], data  # Return both engineered features and original data


# Function to log fraud to a file
def log_fraud(fraudulent_transactions):
    try:
        # Append to file or create if it doesn't exist
        if not os.path.exists(FRAUD_LOG_FILE):
            fraudulent_transactions.to_csv(FRAUD_LOG_FILE, index=False)
        else:
            fraudulent_transactions.to_csv(FRAUD_LOG_FILE, mode='a', header=False, index=False)
    except Exception as e:
        st.error(f"Error writing to log file: {e}")


# Predict fraud on new transactions
def predict_fraud(new_data, full_data):
    global processed_transactions
    predictions = model.predict(new_data)
    full_data['is_fraud'] = predictions

    # Filter for new transactions and frauds
    new_transactions = full_data[~full_data['trans_num'].isin(processed_transactions)]
    frauds = new_transactions[new_transactions['is_fraud'] == 1]

    # Update processed transactions
    processed_transactions.update(new_transactions['trans_num'])

    if not frauds.empty:
        log_fraud(frauds)  # Log frauds to a file

    return frauds


# Streamlit Interface
st.title("Real-Time Fraud Detection")

st.write("""
### Monitoring Transactions for Fraud in Real-Time
This dashboard processes transactions and alerts if fraud is detected. Fraudulent transactions are logged to a file.
""")

# Real-time monitoring section
st.sidebar.header("⚙️ Settings")
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)

# Display dataset preview
st.subheader("📊 Dataset Preview")
processed_data, full_data = load_data()
st.write(full_data.head())

# Real-time fraud detection
st.subheader("🔄 Real-Time Fraud Detection")
placeholder = st.empty()

while True:
    processed_data, full_data = load_data()
    frauds = predict_fraud(processed_data, full_data)
    
    # Check if frauds detected
    if not frauds.empty:
        st.error("⚠️ FRAUD DETECTED! ⚠️")
        st.write("🚨 Fraudulent Transactions Detected:")
        st.write(frauds)
        st.write(f"✅ Fraud logged to `{FRAUD_LOG_FILE}`.")
    else:
        placeholder.write("No fraud detected in the current batch.")

    time.sleep(refresh_interval)