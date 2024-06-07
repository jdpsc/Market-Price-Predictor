from typing import List
from datetime import datetime
import os

import requests
import pandas as pd
import plotly.graph_objects as go

API_KEY = os.getenv("API_KEY")


def build_plot(url: str, y_tile: str, title: str, value_key: str):
    """
    Build plotly graph for data.
    """

    # Get predictions from the API
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(url=url, headers=headers, verify=False, timeout=10)

    if response.status_code != 200:
        # If the response is invalid, build empty dataframes
        preds_df = pd.DataFrame(
            list(zip([], [])),
            columns=["datetime", "value"],
        )

        title = "NO DATA AVAILABLE"
    else:
        json_response = response.json()

        # Build DataFrames for plotting.
        datetime_index = json_response.get("datetime")
        values = json_response.get(value_key)

        preds_df = build_dataframe(datetime_index, values)

    # Create plot
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Arial", size=16),
        ),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Datetime")
    fig.update_yaxes(title_text=y_tile)
    fig.add_scatter(
        x=preds_df["datetime"],
        y=preds_df["value"],
        line=dict(color="#674ea7"),
        hovertemplate="<br>".join(["Datetime: %{x}", "Value: %{y}"]),
        name=y_tile,
    )

    return fig


def build_dataframe(datetime_index: List[datetime], values: List[float]):
    """
    Build DataFrame for plotting from timestamps and values.

    Args:
        datetime_index (List[datetime]): list of timestamp values
        values (List[float]): list of values
    """

    df = pd.DataFrame(
        list(zip(datetime_index, values)),
        columns=["datetime", "value"],
    )

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Resample to daily frequency to make the data continuous
    df = df.set_index("datetime")
    df = df.resample("D").asfreq()
    df = df.reset_index()

    return df
