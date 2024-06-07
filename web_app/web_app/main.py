import os

import requests
import streamlit as st

from components import build_plot

API_KEY = os.getenv("API_KEY")
TITLE = os.getenv("TITLE")
API_URL = os.getenv("API_URL", "http://localhost:8001/api/v1")

st.set_page_config(page_title=TITLE)
st.title(TITLE)

headers = {"Authorization": f"Bearer {API_KEY}"}


# Create dropdown for symbol selection
symbols = (
    requests.get(
        f"{API_URL}/available_stock_symbols", headers=headers, verify=False, timeout=10
    )
    .json()
    .get("values", [])
)

symbol = st.selectbox(
    label="Select the symbol for which you want to see the predictions",
    options=symbols,
    index=None,
)

# Create dropdown for year and month selection
year = st.selectbox(
    label="Select the year for which you want to see the predictions",
    options=[2024],
    index=None,
)

month_names = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
month = st.selectbox(
    label="Select the month for which you want to see the predictions",
    options=month_names,
    index=None,
)

# Check if both area and consumer type have values listed, then create plot for data.
if symbol and year and month:
    # Convert month name to month number
    month = month_names.index(month) + 1

    st.plotly_chart(
        build_plot(
            url=f"{API_URL}/predictions/{symbol}/{year}/{month}",
            y_tile="Close Price Forecast",
            title=f"Market Price Forecasts for {symbol}",
            value_key="prediction",
        )
    )

    st.plotly_chart(
        build_plot(
            url=f"{API_URL}/metrics/{symbol}/{year}/{month}",
            y_tile="Error",
            title=f"Metrics for {symbol}",
            value_key="metric",
        )
    )
