import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from forecast import generate_synthetic_data  # Ensure this is in your project

# 1. Basic Page Configuration
st.set_page_config(
    page_title="RxData Forecast Dashboard",
    layout="wide"
)

# 2. Inject Custom CSS for Theming and Layout
st.markdown(
    """
    <style>
    /* Set the main app background color */
    [data-testid="stAppViewContainer"] {
        background-color: #1E1E1E;
    }

    /* Customize headings */
    h1, h2, h3, h4 {
        color: #FAFAFA;
        font-family: "Arial Black", sans-serif;
    }

    /* Hero section styling */
    .my-hero-section {
        background-color:#262730;
        padding:40px;
        border-radius:10px;
        text-align:center;
        margin-bottom:20px;
        margin-top: -1rem;
    }
    .my-hero-section h1 {
        color:#FAFAFA; 
        font-size:2.5em;
        margin-bottom:0;
    }
    .my-hero-section p {
        color:#F0F0F0; 
        font-size:1.2em; 
        margin-top:10px;
    }

    /* Make text readable on dark background */
    .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #F0F0F0 !important;
    }

    /* Style for buttons */
    .stButton button {
        background-color: #E94F37 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        border: none !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.2rem !important;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #D8432F !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Hero Section (Marketing-Style Header)
hero_html = """
<div class="my-hero-section">
    <h1>Avoid overstock, save in working capital</h1>
    <p>
        30% reduction in inventory costs | 20% fewer stock-outs | Profitable Rx 90 Day Fills | 15% boost in patient retention
    </p>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# 4. Main Title
st.title("RxData Inventory Forecast Dashboard")

# 5. File Uploader for CSV or Excel files
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

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

# 6. Button to Generate Forecast and Display All KPIs/Visuals
if st.button("Generate Forecast"):
    # 6a. Load user data if available; otherwise, use synthetic data
    if uploaded_file:
        df = load_user_data(uploaded_file)
        if df is None:
            st.error("Error processing file, using synthetic data instead.")
            df = generate_synthetic_data()
    else:
        df = generate_synthetic_data()

    # 6b. Forecasting with Prophet
    with st.spinner("Generating forecast..."):
        model = Prophet(yearly_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=90, freq='D')
        forecast_df = model.predict(future)

    # 6c. Basic Forecast KPIs from the last 90 days
    forecast_period = forecast_df.tail(90)
    total_forecast_demand = forecast_period['yhat'].sum()
    average_forecast_demand = forecast_period['yhat'].mean()
    peak_forecast_demand = forecast_period['yhat'].max()
    peak_day = forecast_period.loc[forecast_period['yhat'].idxmax(), 'ds']

    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Total Forecast Demand", value=f"{total_forecast_demand:,.0f}")
    col2.metric(label="Average Forecast Demand", value=f"{average_forecast_demand:,.0f}")
    col3.metric(label="Peak Forecast Demand", value=f"{peak_forecast_demand:,.0f}")
    col4.metric(label="Peak Day", value=str(peak_day.date()))

    # 6d. Additional Inventory KPIs: Targets vs Actual, Overstock, Stockout Savings
    if 'target_inventory' in df.columns and 'actual_inventory' in df.columns:
        forecast_period['inventory_diff'] = forecast_period['actual_inventory'] - forecast_period['target_inventory']
        forecast_period['overstock'] = forecast_period['inventory_diff'].apply(lambda x: x if x > 0 else 0)
        total_overstock = forecast_period['overstock'].sum()
        forecast_period['stockout_savings'] = forecast_period['inventory_diff'].apply(lambda x: -x if x < 0 else 0)
        total_stockout_savings = forecast_period['stockout_savings'].sum()
        total_inventory_gap = forecast_period['inventory_diff'].sum()

        st.subheader("Additional Inventory KPIs")
        col5, col6, col7 = st.columns(3)
        col5.metric(label="Inventory Targets vs Actual", value=f"{total_inventory_gap:,.0f}")
        col6.metric(label="Total Overstock", value=f"{total_overstock:,.0f}")
        col7.metric(label="Total Stockout Savings", value=f"{total_stockout_savings:,.0f}")
    else:
        st.info("Additional Inventory KPIs require 'target_inventory' and 'actual_inventory' columns in your data.")

    # 6e. Inventory Efficiency KPIs: Inventory Turnover Ratio & Days of Inventory on Hand
    if 'cost_of_goods_sold' in df.columns and 'actual_inventory' in df.columns:
        average_inventory = df['actual_inventory'].mean()
        total_cogs = df['cost_of_goods_sold'].sum()
        if average_inventory > 0:
            inventory_turnover_ratio = total_cogs / average_inventory
            days_of_inventory_on_hand = 365 / inventory_turnover_ratio if inventory_turnover_ratio != 0 else None
        else:
            inventory_turnover_ratio = None
            days_of_inventory_on_hand = None

        st.subheader("Inventory Efficiency KPIs")
        col8, col9 = st.columns(2)
        if inventory_turnover_ratio is not None:
            col8.metric(label="Inventory Turnover Ratio", value=f"{inventory_turnover_ratio:.2f}")
        else:
            col8.metric(label="Inventory Turnover Ratio", value="N/A")
        if days_of_inventory_on_hand is not None:
            col9.metric(label="Days of Inventory on Hand", value=f"{days_of_inventory_on_hand:.0f}")
        else:
            col9.metric(label="Days of Inventory on Hand", value="N/A")
    else:
        st.info("Inventory Efficiency KPIs require 'cost_of_goods_sold' and 'actual_inventory' columns.")

    # 6f. Additional Inventory Metrics: Turns, Days of Supply, Reserved/Obsolete Ratio
    if 'cost_of_goods_sold' in df.columns and 'actual_inventory' in df.columns:
        colA, colB, colC = st.columns(3)
        # Here, Turns is the same as the Inventory Turnover Ratio
        colA.metric(label="Turns", value=f"{inventory_turnover_ratio:.2f}" if inventory_turnover_ratio is not None else "N/A")
        # Days of Supply is similar to Days of Inventory on Hand
        colB.metric(label="Days of Supply", value=f"{days_of_inventory_on_hand:.0f}" if days_of_inventory_on_hand is not None else "N/A")
        if 'reserved_inventory' in df.columns and 'obsolete_inventory' in df.columns:
            reserved_total = df['reserved_inventory'].sum()
            obsolete_total = df['obsolete_inventory'].sum()
            ratio = reserved_total / obsolete_total if obsolete_total != 0 else None
            colC.metric(label="Reserved/Obsolete Ratio", value=f"{ratio:.2f}" if ratio is not None else "N/A")
        else:
            colC.info("Reserved/Obsolete KPIs not available")
    else:
        st.info("Additional Inventory Metrics require 'cost_of_goods_sold' and 'actual_inventory' columns.")

    # 6g. Forecast Accuracy Metrics: RMSE and MAPE if historical actual demand is available
    if 'y' in df.columns:
        forecast_with_actual = forecast_df.merge(df[['ds', 'y']], on='ds', how='left')
        historical_data = forecast_with_actual[forecast_with_actual['y'].notnull()]
        if not historical_data.empty:
            historical_data['error'] = historical_data['yhat'] - historical_data['y']
            rmse = np.sqrt((historical_data['error']**2).mean())
            historical_data_nonzero = historical_data[historical_data['y'] != 0]
            if not historical_data_nonzero.empty:
                mape = (abs(historical_data_nonzero['yhat'] - historical_data_nonzero['y']) / historical_data_nonzero['y']).mean() * 100
            else:
                mape = None
            st.subheader("Forecast Accuracy Metrics")
            col10, col11 = st.columns(2)
            if rmse is not None:
                col10.metric(label="RMSE", value=f"{rmse:.2f}")
            else:
                col10.metric(label="RMSE", value="N/A")
            if mape is not None:
                col11.metric(label="MAPE (%)", value=f"{mape:.2f}%")
            else:
                col11.metric(label="MAPE (%)", value="N/A")
        else:
            st.info("Not enough historical data available to calculate forecast accuracy metrics.")
    else:
        st.info("Forecast Accuracy Metrics require a 'y' column with actual demand data.")

    # 6h. Additional Fills KPIs: Total 90 Day Fills, Total Brand Fills, Total Generic Fills, Partial Fills
    if all(col in df.columns for col in ['90_day_fills', 'brand_fills', 'generic_fills', 'partial_fills']):
        total_90_day_fills = df['90_day_fills'].sum()
        total_brand_fills = df['brand_fills'].sum()
        total_generic_fills = df['generic_fills'].sum()
        total_partial_fills = df['partial_fills'].sum()

        st.subheader("Additional Fills KPIs")
        col12, col13, col14, col15 = st.columns(4)
        col12.metric(label="Total 90 Day Fills", value=f"{total_90_day_fills:,.0f}")
        col13.metric(label="Total Brand Fills", value=f"{total_brand_fills:,.0f}")
        col14.metric(label="Total Generic Fills", value=f"{total_generic_fills:,.0f}")
        col15.metric(label="Partial Fills", value=f"{total_partial_fills:,.0f}")
    else:
        st.info("Fills KPIs require '90_day_fills', 'brand_fills', 'generic_fills', and 'partial_fills' columns.")

    # 6i. Display Forecast Data & Plots
    st.subheader("Forecast Data (Last 5 Rows)")
    st.write(forecast_df.tail())

    st.subheader("Forecast Plot")
    fig1 = model.plot(forecast_df)
    st.pyplot(fig1)

    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast_df)
    st.pyplot(fig2)
