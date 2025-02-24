import streamlit as st
import pandas as pd
from prophet import Prophet
from forecast import generate_synthetic_data

def load_user_data(uploaded_file):
    """Load a CSV or Excel file and return a DataFrame."""
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

st.title("RxData Inventory Forecast Dashboard")

# File uploader for CSV or Excel files
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

if st.button("Generate Forecast"):
    # Load user data if available; otherwise, use synthetic data
    if uploaded_file:
        df = load_user_data(uploaded_file)
        if df is None:
            st.error("Error processing file, using synthetic data instead.")
            df = generate_synthetic_data()
    else:
        df = generate_synthetic_data()

    # Forecasting using Prophet
    with st.spinner("Generating forecast..."):
        model = Prophet(yearly_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=90, freq='D')
        forecast_df = model.predict(future)

    # Calculate KPIs from the forecast period (last 90 days)
    forecast_period = forecast_df.tail(90)
    total_forecast_demand = forecast_period['yhat'].sum()
    average_forecast_demand = forecast_period['yhat'].mean()
    peak_forecast_demand = forecast_period['yhat'].max()
    peak_day = forecast_period.loc[forecast_period['yhat'].idxmax(), 'ds']

    # Display KPIs using Streamlit's metric components
    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Total Forecast Demand", value=f"{total_forecast_demand:,.0f}")
    col2.metric(label="Average Forecast Demand", value=f"{average_forecast_demand:,.0f}")
    col3.metric(label="Peak Forecast Demand", value=f"{peak_forecast_demand:,.0f}")
    col4.metric(label="Peak Day", value=str(peak_day.date()))

    # Display forecast data (optional)
    st.subheader("Forecast Data (Last 5 Rows)")
    st.write(forecast_df.tail())

    # Plot the forecast
    st.subheader("Forecast Plot")
    fig1 = model.plot(forecast_df)
    st.pyplot(fig1)

    # Plot forecast components (trend, yearly, etc.)
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast_df)
    st.pyplot(fig2)
