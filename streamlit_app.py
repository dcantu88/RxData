# streamlit_app.py

import streamlit as st
from forecast_utils import generate_synthetic_data, load_user_data  # from your updated forecast_utils.py

# (Optional) If you want to dynamically import the other modules, you can do so below.
# Otherwise, we’ll import them in the relevant menu sections.

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
    st.write("Upload a CSV (with 'ds' and 'y' columns) or skip to use synthetic data. Then click **Generate Forecast** to see results.")

    # 1. File Uploader
    uploaded_file_local = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

    # 2. Generate Forecast Button
    if st.button("Generate Forecast"):
        # 2a. Load data or fallback to synthetic
        if uploaded_file_local:
            df_user = load_user_data(uploaded_file_local)
            if df_user is None:
                st.error("Error reading file. Using synthetic data instead.")
                df = generate_synthetic_data()
            else:
                df = df_user
        else:
            df = generate_synthetic_data()

        # 2b. Run Prophet Forecast
        from forecast_utils import prophet_forecast
        model, forecast_df = prophet_forecast(df, days_to_predict=90)

        # 2c. Display Forecast KPIs
        st.subheader("Forecast KPIs (Next 90 Days)")
        forecast_period = forecast_df.tail(90)
        total_forecast_demand = forecast_period['yhat'].sum()
        average_forecast_demand = forecast_period['yhat'].mean()
        peak_forecast_demand = forecast_period['yhat'].max()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Forecast Demand", f"{total_forecast_demand:,.0f}")
        col2.metric("Average Forecast Demand", f"{average_forecast_demand:,.0f}")
        col3.metric("Peak Forecast Demand", f"{peak_forecast_demand:,.0f}")

        # 2d. Historical KPIs
        st.subheader("Historical KPIs")
        if 'target_inventory' in df.columns and 'actual_inventory' in df.columns:
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

        # 2e. Forecast Accuracy (Prophet)
        st.subheader("Forecast Accuracy Metrics (Prophet)")
        if 'y' in df.columns:
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
        else:
            st.info("Forecast Accuracy Metrics require a 'y' column with actual demand data.")

        # 2f. Show Forecast Data & Plots
        st.markdown("**Forecast Data (Last 5 Rows):**")
        st.write(forecast_df.tail())

        fig = model.plot(forecast_df)
        st.pyplot(fig)

        fig2 = model.plot_components(forecast_df)
        st.pyplot(fig2)
    else:
        st.info("Click **Generate Forecast** to see results.")

# Import new modules for other sections
from model_comparison import show_model_comparison
from parameter_tuning import show_parameter_tuning
from what_if_analysis import show_what_if_analysis

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
