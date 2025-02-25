# business_impact.py

import streamlit as st
from forecast_utils import generate_synthetic_data, prophet_forecast

def compute_business_impact(df, forecast_df, pred_col='yhat'):
    # Example calculation: assume each unit represents $50 cost and a 30% reduction in overstock yields savings.
    forecast_period = forecast_df.tail(90)
    total_forecast = forecast_period[pred_col].sum()
    inventory_cost = 50  # example cost per unit
    potential_savings = 0.3 * total_forecast * inventory_cost
    average_forecast = forecast_period[pred_col].mean()
    peak_forecast = forecast_period[pred_col].max()
    return potential_savings, average_forecast, peak_forecast

def show_business_impact():
    st.header("Business Impact")
    st.write("This section illustrates the business value of our AI/ML solution based on forecast data.")
    
    # Use synthetic data for demonstration (or load actual data if available)
    df = generate_synthetic_data()
    model, forecast_df = prophet_forecast(df, days_to_predict=90)
    
    potential_savings, avg_forecast, peak_forecast = compute_business_impact(df, forecast_df, pred_col='yhat')
    
    st.subheader("Key Business Impact Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Potential Savings ($)", f"${potential_savings:,.0f}")
    col2.metric("Average Forecast Demand", f"{avg_forecast:,.0f} units")
    col3.metric("Peak Forecast Demand", f"{peak_forecast:,.0f} units")
    
    st.markdown(
        """
        **Business Value:**
        - A 30% reduction in overstock can free up significant working capital.
        - Improved forecast accuracy leads to fewer stockouts and higher customer satisfaction.
        - Optimized inventory management translates into lower costs and increased revenue.
        """
    )
    
    try:
        fig = model.plot(forecast_df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying forecast plot: {e}")
