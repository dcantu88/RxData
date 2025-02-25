# streamlit_app.py
import streamlit as st
from forecast_utils import generate_synthetic_data, load_user_data
# Import additional modules when needed via dynamic imports

# Set up page configuration and CSS
st.set_page_config(page_title="RxData Inventory Forecast Dashboard", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #1E1E1E !important; }
    body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span { color: #FFFFFF !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #FFFFFF !important; }
    [data-testid="metric-container"], [data-testid="stMetric"] {
        border: 1px solid #FFFFFF; padding: 1rem; border-radius: 0.5rem;
        background-color: rgba(255,255,255,0.05); margin-bottom: 1rem;
    }
    .stButton button { background-color: #E94F37 !important; color: #FFFFFF !important;
        border-radius: 10px !important; border: none !important; font-size: 1rem !important;
        padding: 0.6rem 1.2rem !important; cursor: pointer; }
    .stButton button:hover { background-color: #D8432F !important; }
    h1, h2, h3, h4 { color: #FAFAFA !important; font-family: "Arial Black", sans-serif; }
    .my-hero-section { background-color:#262730; padding:40px; border-radius:10px;
        text-align:center; margin-bottom:20px; margin-top: -1rem; }
    .my-hero-section h1 { color:#FAFAFA; font-size:2.5em; margin-bottom:0; }
    .my-hero-section p { color:#F0F0F0; font-size:1.2em; margin-top:10px; }
    </style>
    """, unsafe_allow_html=True
)

# Hero Section
st.markdown("""
<div class="my-hero-section">
    <h1>RxData Inventory Forecast & KPI Dashboard</h1>
    <p>Advanced AI/ML solutions to optimize your inventory and drive insights.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Overview", "Forecast", "Model Comparison", "Parameter Tuning", "What-If Analysis"])

def show_overview():
    st.header("Overview")
    st.write("Welcome to the RxData Inventory Forecast & KPI Dashboard. This tool leverages AI/ML to provide inventory forecasts and key performance indicators (KPIs) to optimize your supply chain. Use the sidebar to navigate between sections.")

if menu == "Overview":
    show_overview()
elif menu == "Forecast":
    st.header("Forecast")
    # In the Forecast section, use your existing workflow.
    uploaded_file_local = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])
    if uploaded_file_local:
        df = load_user_data(uploaded_file_local)
        if df is None:
            st.error("Error reading file. Using synthetic data instead.")
            df = generate_synthetic_data()
    else:
        df = generate_synthetic_data()
    from forecast_utils import prophet_forecast
    model, forecast_df = prophet_forecast(df, days_to_predict=90)
    
    st.subheader("Forecast KPIs (Next 90 Days)")
    fp = forecast_df.tail(90)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Forecast Demand", f"{fp['yhat'].sum():,.0f}")
    col2.metric("Average Forecast Demand", f"{fp['yhat'].mean():,.0f}")
    col3.metric("Peak Forecast Demand", f"{fp['yhat'].max():,.0f}")
    
    st.subheader("Historical KPIs")
    # (Insert your full historical KPI code block here—this ensures all historical metrics are shown above the plots.)
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
        st.info("Historical KPIs require 'target_inventory' and 'actual_inventory' columns.")
    
    st.subheader("Forecast Accuracy Metrics (Prophet)")
    if "y" in df.columns:
        forecast_merged = forecast_df.merge(df[['ds','y']], on='ds', how='left')
        historical_data = forecast_merged[forecast_merged['y'].notnull()]
        if not historical_data.empty:
            historical_data['error'] = historical_data['yhat'] - historical_data['y']
            rmse = np.sqrt((historical_data['error']**2).mean())
            historical_data_nonzero = historical_data[historical_data['y'] != 0]
            mape = (abs(historical_data_nonzero['yhat'] - historical_data_nonzero['y']) / historical_data_nonzero['y']).mean() * 100 if not historical_data_nonzero.empty else None
            col7, col8 = st.columns(2)
            col7.metric("RMSE", f"{rmse:.2f}")
            col8.metric("MAPE (%)", f"{mape:.2f}%" if mape is not None else "N/A")
        else:
            st.info("Not enough historical data for accuracy metrics.")
    else:
        st.info("Forecast Accuracy Metrics require a 'y' column.")
    
    st.markdown("**Forecast Data & Plots**")
    st.write(forecast_df.tail())
    fig = model.plot(forecast_df)
    st.pyplot(fig)
    fig2 = model.plot_components(forecast_df)
    st.pyplot(fig2)
elif menu == "Model Comparison":
    from model_comparison import show_model_comparison
    show_model_comparison()
elif menu == "Parameter Tuning":
    from parameter_tuning import show_parameter_tuning
    show_parameter_tuning()
elif menu == "What-If Analysis":
    from what_if_analysis import show_what_if_analysis
    show_what_if_analysis()

