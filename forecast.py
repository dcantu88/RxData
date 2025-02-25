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
    
    # Create the main DataFrame with the required 'ds' and 'y' columns for Prophet
    df = pd.DataFrame({"ds": dates, "y": demand_values})
    
    # Add additional synthetic columns for KPIs:
    # Inventory KPIs
    df["target_inventory"] = df["y"] * 1.1 + np.random.normal(0, 2, days)  # slightly above demand
    df["actual_inventory"] = df["y"] + np.random.normal(0, 2, days)  # similar to demand
    # Cost of Goods Sold (COGS)
    df["cost_of_goods_sold"] = df["y"] * 20 + np.random.normal(0, 10, days)
    
    # Fills KPIs: Random integer values for demonstration
    df["90_day_fills"] = np.random.randint(0, 50, days)
    df["brand_fills"] = np.random.randint(0, 30, days)
    df["generic_fills"] = np.random.randint(0, 20, days)
    df["partial_fills"] = np.random.randint(0, 10, days)
    
    # Reserved/Obsolete Inventory: Random integers as dummy values
    df["reserved_inventory"] = np.random.randint(0, 10, days)
    df["obsolete_inventory"] = np.random.randint(0, 5, days)
    
    return df

def get_forecast(days_to_predict=90):
    df = generate_synthetic_data()
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days_to_predict, freq='D')
    forecast_df = model.predict(future)
    return df, forecast_df, model

