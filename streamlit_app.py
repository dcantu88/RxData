import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb

#############################
# Basic Setup & CSS (unchanged)
#############################
st.set_page_config(page_title="RxData Inventory Forecast & KPI Dashboard", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #1E1E1E !important;
    }
    body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }
    [data-testid="metric-container"],
    [data-testid="stMetric"] {
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

#############################
# Hero Section (Simplified Title)
#############################
hero_html = """
<div class="my-hero-section">
    <h1>RxData Inventory Forecast & KPI Dashboard</h1>
    <p>Advanced AI/ML solutions to optimize your inventory and drive insights.</p>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

#############################
# Sidebar Menu for Navigation
#############################
menu = st.sidebar.radio("Navigation", ["Overview", "Forecast", "Model Comparison", "Parameter Tuning", "What-If Analysis"])

#############################
# Helper Functions (Data loading, synthetic data)
#############################
def generate_synthetic_data(days=365, start_date="2024-01-01"):
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    base_demand = 50
    trend = np.linspace(0, 10, days)
    noise = np.random.normal(0, 5, days)
    demand_values = base_demand + trend + noise
    demand_values = np.clip(demand_values, a_min=0, a_max=None)
    df = pd.DataFrame({"ds": dates, "y": demand_values})
    # Additional KPI columns
    df["target_inventory"] = df["y"] * 1.1 + np.random.normal(0, 2, days)
    df["actual_inventory"] = df["y"] + np.random.normal(0, 2, days)
    df["cost_of_goods_sold"] = df["y"] * 20 + np.random.normal(0, 10, days)
    df["90_day_fills"] = np.random.randint(0, 50, days)
    df["brand_fills"] = np.random.randint(0, 30, days)
    df["generic_fills"] = np.random.randint(0, 20, days)
    df["partial_fills"] = np.random.randint(0, 10, days)
    df["reserved_inventory"] = np.random.randint(0, 10, days)
    df["obsolete_inventory"] = np.random.randint(0, 5, days)
    return df

def load_user_data(uploaded_file):
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file, parse_dates=['ds'])
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(uploaded_file, parse_dates=['ds'])
            else:
                st.error("Unsupported file format!")
                return None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None

def get_data():
    # Use CSV upload if available; otherwise, synthetic
    if uploaded_file:
        df = load_user_data(uploaded_file)
        if df is None:
            st.error("Error reading file. Using synthetic data.")
            return generate_synthetic_data()
        return df
    else:
        return generate_synthetic_data()

#############################
# Forecast Functions (Prophet, XGBoost, LightGBM)
#############################
def prophet_forecast(df, days_to_predict=90):
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days_to_predict, freq='D')
    forecast_df = model.predict(future)
    return model, forecast_df

def build_features_for_ml(df):
    df = df.copy()
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["day_of_month"] = df["ds"].dt.day
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    X = df[["day_of_week", "day_of_month", "month", "year"]]
    y = df["y"] if "y" in df.columns else None
    return X, y

def xgboost_forecast(df, days_to_predict=90):
    X, y = build_features_for_ml(df)
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})
    X_future, _ = build_features_for_ml(future_df)
    future_df["xgb_pred"] = model_xgb.predict(X_future)
    return model_xgb, future_df

def lightgbm_forecast(df, days_to_predict=90):
    X, y = build_features_for_ml(df)
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})
    X_future, _ = build_features_for_ml(future_df)
    future_df["lgb_pred"] = model_lgb.predict(X_future)
    return model_lgb, future_df

#############################
# Section Functions (to organize code)
#############################

def show_overview():
    st.header("Dashboard Overview")
    st.write("This dashboard provides AI/ML-based inventory forecasting along with key performance indicators (KPIs). Use the sidebar menu to navigate through forecast comparisons, parameter tuning, and what-if analyses.")
    # You can add more descriptive text or visuals here as needed.

def show_forecast():
    st.header("Forecast")
    df = get_data()
    if model_option == "Prophet":
        st.subheader("Prophet Forecast")
        model, forecast_df = prophet_forecast(df, days_to_predict=90)
        forecast_period = forecast_df.tail(90)
        st.markdown("**Forecast KPIs (Next 90 Days)**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Forecast Demand", f"{forecast_period['yhat'].sum():,.0f}")
        col2.metric("Average Forecast Demand", f"{forecast_period['yhat'].mean():,.0f}")
        col3.metric("Peak Forecast Demand", f"{forecast_period['yhat'].max():,.0f}")
        st.markdown("**Historical KPIs**")
        # Historical KPI blocks (inventory, efficiency, fills, etc.) go here...
        # (You can copy your existing historical KPI code)
        st.markdown("**Forecast Accuracy Metrics (Prophet)**")
        forecast_merged = forecast_df.merge(df[['ds','y']], on='ds', how='left')
        historical_data = forecast_merged[forecast_merged['y'].notnull()]
        if not historical_data.empty:
            historical_data['error'] = historical_data['yhat'] - historical_data['y']
            rmse = np.sqrt((historical_data['error']**2).mean())
            historical_data_nonzero = historical_data[historical_data['y'] != 0]
            mape = (abs(historical_data_nonzero['yhat'] - historical_data_nonzero['y']) / historical_data_nonzero['y']).mean() * 100 if not historical_data_nonzero.empty else None
            col4, col5 = st.columns(2)
            col4.metric("RMSE", f"{rmse:.2f}")
            col5.metric("MAPE (%)", f"{mape:.2f}%" if mape is not None else "N/A")
        else:
            st.info("Not enough historical data for accuracy metrics.")
        st.markdown("**Forecast Table & Plots**")
        st.write(forecast_df.tail())
        fig = model.plot(forecast_df)
        st.pyplot(fig)
        fig2 = model.plot_components(forecast_df)
        st.pyplot(fig2)
    elif model_option == "XGBoost":
        st.subheader("XGBoost Forecast")
        model_xgb, future_df = xgboost_forecast(df, days_to_predict=90)
        st.markdown("**Forecast KPIs (Next 90 Days)**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Forecast Demand", f"{future_df['xgb_pred'].sum():,.0f}")
        col2.metric("Average Forecast Demand", f"{future_df['xgb_pred'].mean():,.0f}")
        col3.metric("Peak Forecast Demand", f"{future_df['xgb_pred'].max():,.0f}")
        # Historical KPIs code here...
        st.markdown("**Forecast Data & Chart**")
        st.write(future_df.tail())
        st.line_chart(data=future_df.set_index('ds')['xgb_pred'], use_container_width=True)
    else:  # LightGBM
        st.subheader("LightGBM Forecast")
        model_lgb, future_df = lightgbm_forecast(df, days_to_predict=90)
        st.markdown("**Forecast KPIs (Next 90 Days)**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Forecast Demand", f"{future_df['lgb_pred'].sum():,.0f}")
        col2.metric("Average Forecast Demand", f"{future_df['lgb_pred'].mean():,.0f}")
        col3.metric("Peak Forecast Demand", f"{future_df['lgb_pred'].max():,.0f}")
        # Historical KPIs code here...
        st.markdown("**Forecast Data & Chart**")
        st.write(future_df.tail())
        st.line_chart(data=future_df.set_index('ds')['lgb_pred'], use_container_width=True)

def show_model_comparison():
    st.header("Model Comparison")
    st.write("Compare forecasts from different models side-by-side. (This section can include charts, error metrics, and feature importance for each model.)")
    # You can call forecast functions for each model and display their KPIs and charts together.

def show_parameter_tuning():
    st.header("Parameter Tuning")
    st.write("Adjust model hyperparameters using the sliders below and see how your forecast changes in real time.")
    # Example: A slider for XGBoost learning rate, number of estimators, etc.
    # You can integrate these into your forecast functions or call separate tuning functions.

def show_what_if_analysis():
    st.header("What-If Analysis")
    st.write("Simulate different scenarios by modifying input parameters (e.g., sudden demand spikes) and see how the forecast changes.")
    # Provide input fields for users to change variables, then re-run forecasts and display the outcomes.

#############################
# Main Navigation
#############################
if menu == "Overview":
    show_overview()
elif menu == "Forecast":
    show_forecast()
elif menu == "Model Comparison":
    show_model_comparison()
elif menu == "Parameter Tuning":
    show_parameter_tuning()
elif menu == "What-If Analysis":
    show_what_if_analysis()
