import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from forecast import (
    generate_synthetic_data,
    get_forecast,         # Prophet approach
    train_xgboost,        # XGBoost approach
    train_lightgbm        # LightGBM approach
)

st.set_page_config(page_title="RxData Inventory Forecast Dashboard", layout="wide")
st.title("RxData Inventory Forecast Demo")

# File uploader for CSV/Excel files
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

def load_user_data(uploaded_file):
    """Load CSV/Excel data with 'ds' parsed as dates."""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, parse_dates=['ds'])
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file, parse_dates=['ds'])
            else:
                st.error("Unsupported file format!")
                return None
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None

# Let the user choose which forecasting model to use
model_option = st.selectbox("Select Forecasting Model", ("Prophet", "XGBoost", "LightGBM"))

if st.button("Generate Forecast"):
    # Load or generate data
    if uploaded_file:
        df_user = load_user_data(uploaded_file)
        if df_user is None:
            st.error("Error processing file, using synthetic data.")
            df = generate_synthetic_data()
        else:
            df = df_user
    else:
        df = generate_synthetic_data()

    if model_option == "Prophet":
        # Prophet forecasting
        st.subheader("Prophet Forecast")
        model = Prophet(yearly_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=90, freq='D')
        forecast_df = model.predict(future)
        st.write("Prophet Forecast Data (Last 5 Rows):")
        st.write(forecast_df.tail())
        fig = model.plot(forecast_df)
        st.pyplot(fig)
        fig2 = model.plot_components(forecast_df)
        st.pyplot(fig2)

    elif model_option == "XGBoost":
        # XGBoost forecasting
        st.subheader("XGBoost Forecast")
        model_xgb, future_df = train_xgboost(df, days_to_predict=90)
        st.write("XGBoost Forecast Data (Next 90 Days):")
        st.write(future_df.tail())
        st.line_chart(data=future_df.set_index('ds')['xgb_pred'], use_container_width=True)

    elif model_option == "LightGBM":
        # LightGBM forecasting
        st.subheader("LightGBM Forecast")
        model_lgb, future_df = train_lightgbm(df, days_to_predict=90)
        st.write("LightGBM Forecast Data (Next 90 Days):")
        st.write(future_df.tail())
        st.line_chart(data=future_df.set_index('ds')['lgb_pred'], use_container_width=True)
