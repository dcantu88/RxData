# streamlit_app.py

import streamlit as st
from forecast_utils import generate_synthetic_data, load_user_data
from model_comparison import show_model_comparison
from parameter_tuning import show_parameter_tuning
from what_if_analysis import show_what_if_analysis

st.set_page_config(page_title="RxData Inventory Forecast Dashboard", layout="wide")

# Inject custom CSS for dark mode, card-like metrics, etc.
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #1E1E1E !important; }
    body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }
    [data-testid="metric-container"], [data-testid="stMetric"] {
        border: 1px solid #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(255,255,255,0.05);
        margin-bottom: 1rem;
    }
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
    h1, h2, h3, h4 {
        color: #FAFAFA !important;
        font-family: "Arial Black", sans-serif;
    }
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
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section
st.markdown(
    """
    <div class="my-hero-section">
        <h1>RxData Inventory Forecast & KPI Dashboard</h1>
        <p>Advanced AI/ML solutions to optimize your inventory and drive insights.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Overview", "Forecast", "Model Comparison", "Parameter Tuning", "What-If Analysis"])

def show_overview():
    st.header("Overview")
    st.write(
        "Welcome to the RxData Inventory Forecast & KPI Dashboard. "
        "This tool leverages AI/ML to provide inventory forecasts and key performance indicators (KPIs) "
        "to help optimize your supply chain. Use the sidebar to navigate between sections."
    )

def show_forecast_section():
    st.header("Forecast")
    st.write("Upload a CSV (with 'ds' and 'y' columns) or skip to use synthetic data. Then select a model and click **Generate Forecast** to see results.")

    # 1. File Uploader
    uploaded_file_local = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

    # 2. Model Selection
    model_choice = st.selectbox("Select a Forecasting Model", ["Prophet", "XGBoost", "LightGBM"])

    # 3. Generate Forecast Button
    if st.button("Generate Forecast"):
        # 3a. Load data or fallback to synthetic
        if uploaded_file_local:
            df_user = load_user_data(uploaded_file_local)
            if df_user is None:
                st.error("Error reading file. Using synthetic data instead.")
                df = generate_synthetic_data()
            else:
                df = df_user
        else:
            df = generate_synthetic_data()

        # 3b. Validate columns for chosen model
        if model_choice == "Prophet":
            if not {'ds','y'}.issubset(df.columns):
                st.error("Your data must have 'ds' and 'y' columns for Prophet. Using synthetic fallback.")
                df = generate_synthetic_data()
        else:
            if 'ds' not in df.columns:
                st.error("Your data must have 'ds' column for XGBoost/LightGBM. Using synthetic fallback.")
                df = generate_synthetic_data()

        # 3c. Run the selected model forecast
        if model_choice == "Prophet":
            from forecast_utils import prophet_forecast
            model, forecast_df = prophet_forecast(df, days_to_predict=90)
            pred_col = 'yhat'
        elif model_choice == "XGBoost":
            from forecast_utils import xgboost_forecast
            model, forecast_df = xgboost_forecast(df, days_to_predict=90)
            pred_col = 'xgb_pred'
        else:  # LightGBM
            from forecast_utils import lightgbm_forecast
            model, forecast_df = lightgbm_forecast(df, days_to_predict=90)
            pred_col = 'lgb_pred'

        # 3d. Forecast KPIs (Next 90 Days)
        st.subheader("Forecast KPIs (Next 90 Days)")
        if pred_col not in forecast_df.columns:
            st.error(f"Missing '{pred_col}' column in forecast. Cannot display forecast KPIs.")
        else:
            forecast_period = forecast_df.tail(90)
            total_forecast_demand = forecast_period[pred_col].sum()
            average_forecast_demand = forecast_period[pred_col].mean()
            peak_forecast_demand = forecast_period[pred_col].max()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Forecast Demand", f"{total_forecast_demand:,.0f}")
            col2.metric("Average Forecast Demand", f"{average_forecast_demand:,.0f}")
            col3.metric("Peak Forecast Demand", f"{peak_forecast_demand:,.0f}")

        # 3e. Historical KPIs
        st.subheader("Historical KPIs")
        if {'target_inventory','actual_inventory'}.issubset(df.columns):
            df['inventory_diff'] = df['actual_inventory'] - df['target_inventory']
            total_overstock = df[df['inventory_diff'] > 0]['inventory_diff'].sum()
            total_stockout_savings = -df[df['inventory_diff'] < 0]['inventory_diff'].sum()
            total_inventory_gap = df['inventory_diff'].sum()

            col4, col5, col6 = st.columns(3)
            col4.metric("Inventory Targets vs Actual", f"{total_inventory_gap:,.0f}")
            col5.metric("Total Overstock", f"{total_overstock:,.0f}")
            col6.metric("Total Stockout Savings", f"{total_stockout_savings:,.0f}")
        else:
            st.info("Historical Inventory KPIs require 'target_inventory' and 'actual_inventory' columns.")

        # Additional Inventory Efficiency KPIs
        st.subheader("Inventory Efficiency KPIs (Historical)")
        if 'cost_of_goods_sold' in df.columns and 'actual_inventory' in df.columns:
            average_inventory = df['actual_inventory'].mean()
            total_cogs = df['cost_of_goods_sold'].sum()
            if average_inventory > 0:
                inventory_turnover_ratio = total_cogs / average_inventory
                days_of_inventory_on_hand = 365 / inventory_turnover_ratio if inventory_turnover_ratio != 0 else None
            else:
                inventory_turnover_ratio = None
                days_of_inventory_on_hand = None

            col7, col8 = st.columns(2)
            if inventory_turnover_ratio is not None:
                col7.metric("Inventory Turnover Ratio", f"{inventory_turnover_ratio:.2f}")
            else:
                col7.metric("Inventory Turnover Ratio", "N/A")
            if days_of_inventory_on_hand is not None:
                col8.metric("Days of Inventory on Hand", f"{days_of_inventory_on_hand:.0f}")
            else:
                col8.metric("Days of Inventory on Hand", "N/A")
        else:
            st.info("Inventory Efficiency KPIs require 'cost_of_goods_sold' and 'actual_inventory' columns.")

        # Additional Inventory Metrics (Reserved/Obsolete)
        st.subheader("Additional Inventory Metrics (Historical)")
        if 'cost_of_goods_sold' in df.columns and 'actual_inventory' in df.columns:
            colA, colB, colC = st.columns(3)
            # If we calculated inventory_turnover_ratio above, reuse it; else None
            if 'inventory_turnover_ratio' not in locals():
                inventory_turnover_ratio = None
                days_of_inventory_on_hand = None

            colA.metric("Turns", f"{inventory_turnover_ratio:.2f}" if inventory_turnover_ratio else "N/A")
            colB.metric("Days of Supply", f"{days_of_inventory_on_hand:.0f}" if days_of_inventory_on_hand else "N/A")

            if 'reserved_inventory' in df.columns and 'obsolete_inventory' in df.columns:
                reserved_total = df['reserved_inventory'].sum()
                obsolete_total = df['obsolete_inventory'].sum()
                ratio = None
                if obsolete_total != 0:
                    ratio = reserved_total / obsolete_total
                colC.metric("Reserved/Obsolete Ratio", f"{ratio:.2f}" if ratio is not None else "N/A")
            else:
                colC.info("Reserved/Obsolete KPIs not available")
        else:
            st.info("Additional Inventory Metrics require 'cost_of_goods_sold' and 'actual_inventory' columns.")

        # Additional Fills KPIs
        st.subheader("Additional Fills KPIs (Historical)")
        if all(col in df.columns for col in ['90_day_fills', 'brand_fills', 'generic_fills', 'partial_fills']):
            total_90_day_fills = df['90_day_fills'].sum()
            total_brand_fills = df['brand_fills'].sum()
            total_generic_fills = df['generic_fills'].sum()
            total_partial_fills = df['partial_fills'].sum()

            colF1, colF2, colF3, colF4 = st.columns(4)
            colF1.metric("Total 90 Day Fills", f"{total_90_day_fills:,.0f}")
            colF2.metric("Total Brand Fills", f"{total_brand_fills:,.0f}")
            colF3.metric("Total Generic Fills", f"{total_generic_fills:,.0f}")
            colF4.metric("Partial Fills", f"{total_partial_fills:,.0f}")
        else:
            st.info("Fills KPIs require '90_day_fills', 'brand_fills', 'generic_fills', and 'partial_fills' columns.")

        # 3f. Forecast Accuracy (Prophet only)
        if model_choice == "Prophet":
            st.subheader("Forecast Accuracy Metrics (Prophet)")
            if 'y' not in df.columns:
                st.info("Forecast Accuracy Metrics require a 'y' column with actual demand data.")
            else:
                try:
                    forecast_merged = forecast_df.merge(df[['ds','y']], on='ds', how='left')
                    historical_data = forecast_merged[forecast_merged['y'].notnull()]
                    if not historical_data.empty:
                        historical_data['error'] = historical_data['yhat'] - historical_data['y']
                        rmse = np.sqrt((historical_data['error']**2).mean())

                        historical_data_nonzero = historical_data[historical_data['y'] != 0]
                        if not historical_data_nonzero.empty:
                            mape = (abs(historical_data_nonzero['yhat'] - historical_data_nonzero['y']) / historical_data_nonzero['y']).mean() * 100
                        else:
                            mape = None

                        col7, col8 = st.columns(2)
                        col7.metric("RMSE", f"{rmse:.2f}")
                        if mape is not None:
                            col8.metric("MAPE (%)", f"{mape:.2f}%")
                        else:
                            col8.metric("MAPE (%)", "N/A")
                    else:
                        st.info("Not enough overlapping historical data to compute accuracy metrics.")
                except Exception as e:
                    st.error(f"Accuracy merge error: {e}")
        else:
            st.info("Forecast Accuracy Metrics are only shown for Prophet in this demo.")

        # 3g. Show Forecast Data & Plots
        st.markdown("**Forecast Data (Last 5 Rows):**")
        if not forecast_df.empty:
            st.write(forecast_df.tail())
        else:
            st.warning("Forecast DataFrame is empty—no forecast to display.")

        # Plotting
        if model_choice == "Prophet":
            try:
                fig = model.plot(forecast_df)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Prophet main plot error: {e}")
            try:
                fig2 = model.plot_components(forecast_df)
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Prophet component plot error: {e}")
        else:
            # XGBoost or LightGBM line chart
            if pred_col in forecast_df.columns and 'ds' in forecast_df.columns:
                st.line_chart(data=forecast_df.set_index('ds')[pred_col], use_container_width=True)
            else:
                st.error("Cannot plot line chart: missing 'ds' or prediction column in forecast_df.")
    else:
        st.info("Select a model and click **Generate Forecast** to see results.")

if menu == "Overview":
    show_overview()
elif menu == "Forecast":
    show_forecast_section()
elif menu == "Model Comparison":
    show_model_comparison()
elif menu == "Parameter Tuning":
    show_parameter_tuning()
elif menu == "What-If Analysis":
    show_what_if_analysis()



