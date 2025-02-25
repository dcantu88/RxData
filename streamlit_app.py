import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb

#############################
# 1. Basic Setup & CSS
#############################

st.set_page_config(page_title="RxData Inventory Forecast Dashboard", layout="wide")

st.markdown(
    """
    <style>
    /* Force a dark background for the entire app */
    [data-testid="stAppViewContainer"] {
        background-color: #1E1E1E !important;
    }

    /* Ensure all text is white (overriding any default or system theme) */
    body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #FFFFFF !important;
    }

    /* Force the metric text (both label and value) to be white */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }

    /* Card-like styling for metrics */
    [data-testid="metric-container"],
    [data-testid="stMetric"] {
        border: 1px solid #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(255,255,255,0.05);
        margin-bottom: 1rem;
    }

    /* Buttons */
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

    /* Headings */
    h1, h2, h3, h4 {
        color: #FAFAFA !important;
        font-family: "Arial Black", sans-serif;
    }

    /* Hero section */
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
# 2. Hero Section
#############################
hero_html = """
<div class="my-hero-section">
    <h1>Avoid overstock, save in working capital</h1>
    <p>
        30% reduction in inventory costs | 20% fewer stock-outs | Profitable Rx 90 Day Fills | 15% boost in patient retention
    </p>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

#############################
# 3. Title & File Uploader
#############################
st.title("RxData Inventory Forecast Dashboard")

uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

#############################
# 4. Synthetic Data Generator
#############################
def generate_synthetic_data(days=365, start_date="2024-01-01"):
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    base_demand = 50
    trend = np.linspace(0, 10, days)
    noise = np.random.normal(0, 5, days)
    demand_values = base_demand + trend + noise
    demand_values = np.clip(demand_values, a_min=0, a_max=None)
    
    df = pd.DataFrame({"ds": dates, "y": demand_values})
    # Additional columns for inventory/fills KPIs
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

#############################
# 5. Helper: Load User Data
#############################
def load_user_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, parse_dates=['ds'])
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file, parse_dates=['ds'])
            else:
                st.error("Unsupported file format!")
                return None
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None

#############################
# 6. Prophet Forecast
#############################
def prophet_forecast(df, days_to_predict=90):
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days_to_predict, freq='D')
    forecast_df = model.predict(future)
    return model, forecast_df

#############################
# 7. XGBoost Forecast
#############################
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

#############################
# 8. LightGBM Forecast
#############################
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
# 9. Model Selection UI
#############################
model_option = st.selectbox("Select Forecasting Model", ("Prophet", "XGBoost", "LightGBM"))

if st.button("Generate Forecast"):
    # Load or generate data
    if uploaded_file:
        df_user = load_user_data(uploaded_file)
        if df_user is None:
            st.error("Error reading file. Using synthetic data instead.")
            df = generate_synthetic_data()
        else:
            df = df_user
    else:
        df = generate_synthetic_data()

    #################################
    # A. Forecast with chosen model
    #################################
    if model_option == "Prophet":
        st.subheader("Prophet Forecast")
        model, forecast_df = prophet_forecast(df, days_to_predict=90)
        
        # Forecast KPIs
        forecast_period = forecast_df.tail(90)
        total_forecast_demand = forecast_period['yhat'].sum()
        average_forecast_demand = forecast_period['yhat'].mean()
        peak_forecast_demand = forecast_period['yhat'].max()

        st.subheader("Forecast KPIs (Next 90 Days)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Forecast Demand", f"{total_forecast_demand:,.0f}")
        col2.metric("Average Forecast Demand", f"{average_forecast_demand:,.0f}")
        col3.metric("Peak Forecast Demand", f"{peak_forecast_demand:,.0f}")

        # (Then show historical KPIs, accuracy, plots, etc. as needed...)

    elif model_option == "XGBoost":
        st.subheader("XGBoost Forecast")
        model_xgb, future_df = xgboost_forecast(df, days_to_predict=90)

        # Forecast KPIs
        total_forecast_demand = future_df['xgb_pred'].sum()
        average_forecast_demand = future_df['xgb_pred'].mean()
        peak_forecast_demand = future_df['xgb_pred'].max()

        st.subheader("Forecast KPIs (Next 90 Days)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Forecast Demand", f"{total_forecast_demand:,.0f}")
        col2.metric("Average Forecast Demand", f"{average_forecast_demand:,.0f}")
        col3.metric("Peak Forecast Demand", f"{peak_forecast_demand:,.0f}")

        # (Then show historical KPIs, etc.)

    else:  # LightGBM
        st.subheader("LightGBM Forecast")
        model_lgb, future_df = lightgbm_forecast(df, days_to_predict=90)

        # Forecast KPIs
        total_forecast_demand = future_df['lgb_pred'].sum()
        average_forecast_demand = future_df['lgb_pred'].mean()
        peak_forecast_demand = future_df['lgb_pred'].max()

        st.subheader("Forecast KPIs (Next 90 Days)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Forecast Demand", f"{total_forecast_demand:,.0f}")
        col2.metric("Average Forecast Demand", f"{average_forecast_demand:,.0f}")
        col3.metric("Peak Forecast Demand", f"{peak_forecast_demand:,.0f}")

        # (Then show historical KPIs, etc.)
