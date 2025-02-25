# anomaly_detection.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from forecast_utils import generate_synthetic_data

def detect_anomalies(df, contamination=0.05):
    """
    Use IsolationForest to detect anomalies in the 'y' column of the DataFrame.
    Returns the DataFrame with an additional column 'anomaly' (1 for normal, -1 for anomaly).
    """
    # Ensure no missing values in 'y'
    data = df[['y']].dropna()
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(data)
    df = df.copy()
    df['anomaly'] = predictions
    return df

def show_anomaly_detection():
    st.header("Anomaly Detection")
    st.write("This section detects anomalies in historical demand data using IsolationForest.")

    # Use synthetic data for demonstration
    df = generate_synthetic_data()
    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Detect anomalies (using a default contamination of 5%)
    df_anomaly = detect_anomalies(df, contamination=0.05)
    
    num_anomalies = (df_anomaly['anomaly'] == -1).sum()
    st.metric("Number of Anomalies Detected", num_anomalies)
    
    st.subheader("Anomaly Detection Chart")
    # Plot the time series and overlay anomalies as red dots
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_anomaly['ds'], df_anomaly['y'], label="Demand", color="skyblue")
    anomalies = df_anomaly[df_anomaly['anomaly'] == -1]
    ax.scatter(anomalies['ds'], anomalies['y'], color='red', label="Anomaly", zorder=5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend()
    st.pyplot(fig)
