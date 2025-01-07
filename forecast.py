# forecast.py
import numpy as np
import pandas as pd
from prophet import Prophet

def generate_synthetic_data(days=365, start_date="2024-01-01"):
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    base_demand = 50
    trend = np.linspace(0, 10, days)
    noise = np.random.normal(0, 5, days)
    demand_values = base_demand + trend + noise
    demand_values = np.clip(demand_values, a_min=0, a_max=None)
    df = pd.DataFrame({"ds": dates, "y": demand_values})
    return df

def get_forecast(days_to_predict=90):
    df = generate_synthetic_data()
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days_to_predict, freq='D')
    forecast_df = model.predict(future)
    return df, forecast_df, model
