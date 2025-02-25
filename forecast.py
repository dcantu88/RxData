import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor
import lightgbm as lgb

def generate_synthetic_data(days=365, start_date="2024-01-01"):
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    base_demand = 50
    trend = np.linspace(0, 10, days)
    noise = np.random.normal(0, 5, days)
    demand_values = base_demand + trend + noise
    demand_values = np.clip(demand_values, a_min=0, a_max=None)
    
    # Create the basic DataFrame
    df = pd.DataFrame({"ds": dates, "y": demand_values})
    
    # Add additional columns for inventory KPIs (dummy values)
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

def build_features_xgboost(df):
    """
    Create simple features from the date for ML models.
    """
    df = df.copy()
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["day_of_month"] = df["ds"].dt.day
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    X = df[["day_of_week", "day_of_month", "month", "year"]]
    y = df["y"]
    return X, y

def train_xgboost(df, days_to_predict=90):
    """
    Train an XGBoost regressor on historical data and predict the next days_to_predict days.
    """
    X, y = build_features_xgboost(df)
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    
    # Create future dates
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})
    X_future, _ = build_features_xgboost(future_df)
    future_df["xgb_pred"] = model.predict(X_future)
    
    return model, future_df

def train_lightgbm(df, days_to_predict=90):
    """
    Train a LightGBM regressor on historical data and predict the next days_to_predict days.
    """
    X, y = build_features_xgboost(df)
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    
    # Create future dates
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})
    X_future, _ = build_features_xgboost(future_df)
    future_df["lgb_pred"] = model.predict(X_future)
    
    return model, future_df

def get_forecast(days_to_predict=90):
    """
    Train a Prophet model and generate a forecast.
    """
    df = generate_synthetic_data()
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days_to_predict, freq='D')
    forecast_df = model.predict(future)
    return df, forecast_df, model
