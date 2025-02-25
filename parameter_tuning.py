# parameter_tuning.py

import streamlit as st
import pandas as pd
import xgboost as xgb
from forecast_utils import generate_synthetic_data, build_features_for_ml

def show_parameter_tuning():
    st.header("Parameter Tuning")
    st.write("Adjust XGBoost hyperparameters to see how the forecast changes.")
    
    lr = st.slider("XGBoost Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    
    df = generate_synthetic_data()
    X, y = build_features_for_ml(df)
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})
    X_future, _ = build_features_for_ml(future_df)
    future_df["xgb_pred"] = model_xgb.predict(X_future)
    
    st.markdown("#### Tuned XGBoost Forecast KPIs (Next 90 Days)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Forecast Demand", f"{future_df['xgb_pred'].sum():,.0f}")
    col2.metric("Average Forecast Demand", f"{future_df['xgb_pred'].mean():,.0f}")
    col3.metric("Peak Forecast Demand", f"{future_df['xgb_pred'].max():,.0f}")
    
    st.line_chart(data=future_df.set_index('ds')['xgb_pred'], use_container_width=True)
    st.info(f"XGBoost learning rate set to {lr}")
