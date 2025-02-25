# streamlit_app.py

import streamlit as st
from forecast_utils import generate_synthetic_data, load_user_data
from model_comparison import show_model_comparison
from parameter_tuning import show_parameter_tuning
from what_if_analysis import show_what_if_analysis

# Set page configuration and inject custom CSS
st.set_page_config(page_title="RxData Inventory Forecast Dashboard", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #1E1E1E !important; }
    body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #FFFFFF !important; }
    [data-testid="metric-container"], [data-testid="stMetric"] {
        border: 1px solid #FFFFFF; padding: 1rem; border-radius: 0.5rem;
        background-color: rgba(255,255,255,0.05); margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #E94F37 !important; color: #FFFFFF !important;
        border-radius: 10px !important; border: none !important; font-size: 1rem !important;
        padding: 0.6rem 1.2rem !important; cursor: pointer;
    }
    .stButton button:hover { background-color: #D8432F !important; }
    h1, h2, h3, h4 { color: #FAFAFA !important; font-family: "Arial Black", sans-serif; }
    .my-hero-section {
        background-color:#262730; padding:40px; border-radius:10px;
        text-align:center; margin-bottom:20px; margin-top: -1rem;
    }
    .my-hero-section h1 { color:#FAFAFA; font-size:2.5em; margin-bottom:0; }
    .my-hero-section p { color:#F0F0F0; font-size:1.2em; margin-top:10px; }
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
menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Forecast",
        "Business Impact",
        "Automated Insights",
        "Model Comparison",
        "Parameter Tuning",
        "What-If Analysis"
    ]
)

# Overview Section
def show_overview():
    st.header("Overview")
    st.write(
        "Welcome to the RxData Inventory Forecast & KPI Dashboard. "
        "This tool leverages AI/ML to provide inventory forecasts and key performance indicators (KPIs) "
        "that help optimize your supply chain. Use the sidebar to navigate between sections."
    )

# Forecast Section (Existing functionality with robust checks)
def show_forecast_section():
    st.header("Forecast")
    st.write("Upload a CSV (with 'ds' and 'y' columns) or use synthetic data. Then select a model and click **Generate Forecast**.")

    # File Uploader
    uploaded_file_local = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

    # Model Dropdown
    model_choice = st.selectbox("Select a Forecasting Model", ["Prophet", "XGBoost", "LightGBM"])

    # Generate Forecast Button
    if st.button("Generate Forecast"):
        # Load data or fallback to synthetic
        if uploaded_file_local:
            df_user = load_user_data(uploaded_file_local)
            if df_user is None:
                st.error("Error reading file. Using synthetic data instead.")
                df = generate_synthetic_data()
            else:
                df = df_user
        else:
            df = generate_synthetic_data()

        # Validate required columns
        if model_choice == "Prophet":
            if not {'ds','y'}.issubset(df.columns):
                st.error("Your data must have 'ds' and 'y' columns for Prophet. Using synthetic fallback.")
                df = generate_synthetic_data()
        else:
            if 'ds' not in df.columns:
                st.error("Your data must have 'ds' column for XGBoost/LightGBM. Using synthetic fallback.")
                df = generate_synthetic_data()

        # Run selected model forecast
        if model_choice == "Prophet":
            from forecast_utils import prophet_forecast
            model, forecast_df = prophet_forecast(df, days_to_predict=90)
            pred_col = 'yhat'
        elif model_choice == "XGBoost":
            from forecast_utils import xgboost_forecast
            model, forecast_df = xgboost_forecast(df, days_to_predict=90)
            pred_col = 'xgb_pred'
        else:
            from forecast_utils import lightgbm_forecast
            model, forecast_df = lightgbm_forecast(df, days_to_predict=90)
            pred_col = 'lgb_pred'

        # Forecast KPIs
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

        # Historical KPIs
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
            avg_inv = df['actual_inventory'].mean()
            total_cogs = df['cost_of_goods_sold'].sum()
            if avg_inv > 0:
                inv_turnover = total_cogs / avg_inv
                days_inv = 365 / inv_turnover if inv_turnover != 0 else None
            else:
                inv_turnover = None
                days_inv = None

            col7, col8 = st.columns(2)
            col7.metric("Inventory Turnover Ratio", f"{inv_turnover:.2f}" if inv_turnover else "N/A")
            col8.metric("Days of Inventory on Hand", f"{days_inv:.0f}" if days_inv else "N/A")
        else:
            st.info("Inventory Efficiency KPIs require 'cost_of_goods_sold' and 'actual_inventory' columns.")

        # Additional Inventory Metrics (Reserved/Obsolete)
        st.subheader("Additional Inventory Metrics (Historical)")
        if 'reserved_inventory' in df.columns and 'obsolete_inventory' in df.columns:
            reserved_total = df['reserved_inventory'].sum()
            obsolete_total = df['obsolete_inventory'].sum()
            ratio = reserved_total / obsolete_total if obsolete_total != 0 else None
            colA, colB, colC = st.columns(3)
            colA.metric("Turns", f"{inv_turnover:.2f}" if inv_turnover else "N/A")
            colB.metric("Days of Supply", f"{days_inv:.0f}" if days_inv else "N/A")
            colC.metric("Reserved/Obsolete Ratio", f"{ratio:.2f}" if ratio is not None else "N/A")
        else:
            st.info("Additional Inventory Metrics require 'reserved_inventory' and 'obsolete_inventory' columns.")

        # Additional Fills KPIs
        st.subheader("Additional Fills KPIs (Historical)")
        if all(col in df.columns for col in ['90_day_fills','brand_fills','generic_fills','partial_fills']):
            total_90_fills = df['90_day_fills'].sum()
            total_brand_fills = df['brand_fills'].sum()
            total_generic_fills = df['generic_fills'].sum()
            total_partial_fills = df['partial_fills'].sum()
            colF1, colF2, colF3, colF4 = st.columns(4)
            colF1.metric("Total 90 Day Fills", f"{total_90_fills:,.0f}")
            colF2.metric("Total Brand Fills", f"{total_brand_fills:,.0f}")
            colF3.metric("Total Generic Fills", f"{total_generic_fills:,.0f}")
            colF4.metric("Partial Fills", f"{total_partial_fills:,.0f}")
        else:
            st.info("Fills KPIs require '90_day_fills', 'brand_fills', 'generic_fills', and 'partial_fills' columns.")

        # Business Impact & Value (Dynamic calculation based on forecast data)
        st.subheader("Business Impact & Value")
        if 'cost_of_goods_sold' in df.columns and 'y' in df.columns:
            avg_cost = df['cost_of_goods_sold'].sum() / df['y'].sum()
        else:
            avg_cost = 50  # default cost per unit
        if {'target_inventory','actual_inventory'}.issubset(df.columns):
            df['inventory_diff'] = df['actual_inventory'] - df['target_inventory']
            total_overstock = df[df['inventory_diff'] > 0]['inventory_diff'].sum()
        else:
            total_overstock = 0
        potential_savings = total_overstock * avg_cost * 0.30  # assume 30% reduction yields savings

        st.subheader("Key Business Impact Metrics")
        colBI1, colBI2 = st.columns(2)
        colBI1.metric("Estimated Savings from Reduced Overstock", f"${potential_savings:,.0f} per period")
        colBI2.metric("Business Value Summary", "Optimized inventory = higher profitability")
        st.markdown(
            """
            **Business Value:**
            - Reducing overstock frees up significant working capital.
            - Improved forecast accuracy leads to fewer stockouts and higher customer satisfaction.
            - Efficient inventory management directly translates to cost savings and increased revenue.
            """
        )

        # Forecast Accuracy Metrics (Prophet only)
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

        # Show Forecast Data & Plots
        st.markdown("**Forecast Data (Last 5 Rows):**")
        if not forecast_df.empty:
            st.write(forecast_df.tail())
        else:
            st.warning("Forecast DataFrame is empty—no forecast to display.")

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
            if pred_col in forecast_df.columns and 'ds' in forecast_df.columns:
                st.line_chart(data=forecast_df.set_index('ds')[pred_col], use_container_width=True)
            else:
                st.error("Cannot plot line chart: missing 'ds' or prediction column in forecast_df.")
    else:
        st.info("Select a model and click **Generate Forecast** to see results.")

# Business Impact Section (separate module)
def show_business_impact():
    from business_impact import show_business_impact as bip
    bip()

# Main Navigation Logic
if menu == "Overview":
    show_overview()
elif menu == "Forecast":
    show_forecast_section()
elif menu == "Business Impact":
    show_business_impact()
elif menu == "Automated Insights":
    from automated_insights import show_automated_insights
    # For automated insights, we can generate synthetic data and a Prophet forecast as an example
    df = generate_synthetic_data()
    from forecast_utils import prophet_forecast
    model, forecast_df = prophet_forecast(df, days_to_predict=90)
    pred_col = 'yhat'
    forecast_period = forecast_df.tail(90)
    forecast_kpis = {
        "total": forecast_period[pred_col].sum(),
        "average": forecast_period[pred_col].mean(),
        "peak": forecast_period[pred_col].max()
    }
    if {'target_inventory','actual_inventory'}.issubset(df.columns):
        df['inventory_diff'] = df['actual_inventory'] - df['target_inventory']
        historical_kpis = {"inventory_gap": df['inventory_diff'].sum()}
    else:
        historical_kpis = {"inventory_gap": 0}
    if 'cost_of_goods_sold' in df.columns and 'y' in df.columns:
        avg_cost = df['cost_of_goods_sold'].sum() / df['y'].sum()
    else:
        avg_cost = 50
    potential_savings = 0.3 * forecast_kpis["total"] * avg_cost
    business_metrics = {"savings": potential_savings}
    from automated_insights import show_automated_insights
    show_automated_insights(forecast_kpis, historical_kpis, business_metrics)
elif menu == "Model Comparison":
    show_model_comparison()
elif menu == "Parameter Tuning":
    show_parameter_tuning()
elif menu == "What-If Analysis":
    show_what_if_analysis()
