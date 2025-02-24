import streamlit as st
import pandas as pd
from prophet import Prophet
from forecast import generate_synthetic_data

# Caching the file loading to avoid reprocessing on each run
@st.cache_data
def load_user_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format!")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Cache the model fitting if the data doesn't change
@st.cache_resource
def fit_prophet_model(df):
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    return model

st.title("RxData Forecast Demo")

# File uploader widget
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

if st.button("Generate Forecast"):
    # Determine data source: user-uploaded file or synthetic data
    if uploaded_file:
        df = load_user_data(uploaded_file)
        if df is None:
            st.error("Error processing the uploaded file. Using synthetic data instead.")
            df = generate_synthetic_data()
    else:
        df = generate_synthetic_data()

    # Forecasting with a spinner for user feedback
    with st.spinner("Fitting the model and generating forecast..."):
        try:
            model = fit_prophet_model(df)
            future = model.make_future_dataframe(periods=90, freq='D')
            forecast_df = model.predict(future)
            st.success("Forecast generated successfully!")
            st.write("Forecast (last 5 rows):", forecast_df.tail())
            fig = model.plot(forecast_df)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during forecasting: {e}")

