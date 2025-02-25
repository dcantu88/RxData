# model_comparison.py
import streamlit as st
from forecast_utils import generate_synthetic_data, prophet_forecast, xgboost_forecast, lightgbm_forecast

def show_model_comparison():
    st.header("Model Comparison")
    st.write("Compare forecasts from different models using the same input data.")
    df = generate_synthetic_data()  # For demonstration; you can later allow CSV upload.
    
    model_prophet, forecast_df_prophet = prophet_forecast(df, days_to_predict=90)
    _, future_df_xgb = xgboost_forecast(df, days_to_predict=90)
    _, future_df_lgb = lightgbm_forecast(df, days_to_predict=90)
    
    # Calculate KPIs for each model
    prophet_kpis = {
        "Total Forecast Demand": f"{forecast_df_prophet.tail(90)['yhat'].sum():,.0f}",
        "Average Forecast Demand": f"{forecast_df_prophet.tail(90)['yhat'].mean():,.0f}",
        "Peak Forecast Demand": f"{forecast_df_prophet.tail(90)['yhat'].max():,.0f}"
    }
    xgb_kpis = {
        "Total Forecast Demand": f"{future_df_xgb['xgb_pred'].sum():,.0f}",
        "Average Forecast Demand": f"{future_df_xgb['xgb_pred'].mean():,.0f}",
        "Peak Forecast Demand": f"{future_df_xgb['xgb_pred'].max():,.0f}"
    }
    lgb_kpis = {
        "Total Forecast Demand": f"{future_df_lgb['lgb_pred'].sum():,.0f}",
        "Average Forecast Demand": f"{future_df_lgb['lgb_pred'].mean():,.0f}",
        "Peak Forecast Demand": f"{future_df_lgb['lgb_pred'].max():,.0f}"
    }
    
    st.markdown("### Prophet Forecast KPIs")
    for key, value in prophet_kpis.items():
        st.write(f"**{key}:** {value}")
        
    st.markdown("### XGBoost Forecast KPIs")
    for key, value in xgb_kpis.items():
        st.write(f"**{key}:** {value}")
        
    st.markdown("### LightGBM Forecast KPIs")
    for key, value in lgb_kpis.items():
        st.write(f"**{key}:** {value}")
