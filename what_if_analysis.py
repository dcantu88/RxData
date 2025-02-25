# what_if_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from forecast_utils import generate_synthetic_data

def show_what_if_analysis():
    st.header("What-If Analysis")
    st.write("Simulate scenarios by adjusting synthetic data parameters.")
    
    base_demand = st.number_input("Base Demand", value=50)
    noise_std = st.slider("Noise Standard Deviation", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
    
    def generate_custom_synthetic_data(days=365, start_date="2024-01-01"):
        dates = pd.date_range(start=start_date, periods=days, freq="D")
        trend = np.linspace(0, 10, days)
        noise = np.random.normal(0, noise_std, days)
        demand_values = base_demand + trend + noise
        demand_values = np.clip(demand_values, a_min=0, a_max=None)
        df_custom = pd.DataFrame({"ds": dates, "y": demand_values})
        df_custom["target_inventory"] = df_custom["y"] * 1.1 + np.random.normal(0, 2, days)
        df_custom["actual_inventory"] = df_custom["y"] + np.random.normal(0, 2, days)
        df_custom["cost_of_goods_sold"] = df_custom["y"] * 20 + np.random.normal(0, 10, days)
        df_custom["90_day_fills"] = np.random.randint(0, 50, days)
        df_custom["brand_fills"] = np.random.randint(0, 30, days)
        df_custom["generic_fills"] = np.random.randint(0, 20, days)
        df_custom["partial_fills"] = np.random.randint(0, 10, days)
        df_custom["reserved_inventory"] = np.random.randint(0, 10, days)
        df_custom["obsolete_inventory"] = np.random.randint(0, 5, days)
        return df_custom
    
    df_custom = generate_custom_synthetic_data()
    st.write("Custom Synthetic Data Preview:", df_custom.head())
    
    from forecast_utils import prophet_forecast
    model_custom, forecast_df_custom = prophet_forecast(df_custom, days_to_predict=90)
    forecast_period = forecast_df_custom.tail(90)
    
    st.markdown("#### Forecast KPIs (Custom Scenario)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Forecast Demand", f"{forecast_period['yhat'].sum():,.0f}")
    col2.metric("Average Forecast Demand", f"{forecast_period['yhat'].mean():,.0f}")
    col3.metric("Peak Forecast Demand", f"{forecast_period['yhat'].max():,.0f}")
    
    st.write("Custom Forecast Data (Last 5 Rows):", forecast_df_custom.tail())
    fig = model_custom.plot(forecast_df_custom)
    st.pyplot(fig)
    st.info("This scenario simulates changes in base demand and variability.")
