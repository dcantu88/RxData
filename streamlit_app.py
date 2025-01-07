# streamlit_app.py
import streamlit as st
from forecast import get_forecast

st.title("RxData Forecast Demo")

# Simple example of calling the forecast
if st.button("Generate Forecast"):
    df_actual, df_forecast, model = get_forecast(90)
    st.write("Forecast (last 5 rows):", df_forecast.tail())
    fig = model.plot(df_forecast)
    st.pyplot(fig)
