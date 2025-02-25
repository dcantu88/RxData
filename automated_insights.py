# automated_insights.py

import streamlit as st

def generate_insights(forecast_kpis, historical_kpis, business_metrics):
    insights = f"""
    **Forecast Overview:**
    - Total Forecast Demand (Next 90 Days): {forecast_kpis['total']:,} units
    - Average Demand: {forecast_kpis['average']:,} units
    - Peak Demand: {forecast_kpis['peak']:,} units

    **Historical Context:**
    - Inventory Gap: {historical_kpis['inventory_gap']:,} units

    **Business Impact:**
    - Potential Savings from Optimized Inventory: ${business_metrics['savings']:,} per period

    By optimizing inventory levels, the pharmacy can significantly reduce wasted capital and improve service levels, leading to higher profitability.
    """
    return insights

def show_automated_insights(forecast_kpis, historical_kpis, business_metrics):
    st.header("Automated Insights")
    insights = generate_insights(forecast_kpis, historical_kpis, business_metrics)
    st.markdown(insights)
