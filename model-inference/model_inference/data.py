import tempfile
import os
from datetime import datetime

import boto3
import hopsworks
import pandas as pd


def load_data_from_feature_store(
    symbol: str,
    start_datetime: datetime,
    end_datetime: datetime,
    feature_view_version: int,
) -> pd.DataFrame:
    """Load stock data from the feature store for a given symbol and time range."""

    # Connect to feature store
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT_ID"],
    )

    # Connect to Hopsworks Feature Store
    fs = project.get_feature_store()

    feature_view = fs.get_feature_view(
        name="stock_data_feature_view", version=feature_view_version
    )
    data = feature_view.get_batch_data(start_time=start_datetime, end_time=end_datetime)

    data = data.sort_values("date")
    data = data[data["s"] == symbol]

    return data


def load_df_from_s3(path: str) -> pd.DataFrame:
    """Load a parquet file from s3, if it exists."""

    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = f"{tmp_dir}/temp_file.parquet"

        # Check if the file exists is s3
        try:
            s3.head_object(Bucket="stock-market-preds", Key=path)
        except:
            pass
        else:
            # If the file exists, download it
            s3.download_file("stock-market-preds", path, temp_path)
            # Append the new predictions to the existing file
            with open(temp_path, "rb") as f:
                df = pd.read_parquet(f)
                return df

    return None


def add_to_s3(df: pd.DataFrame, path: str):
    """Add a DataFrame to a parquet file in S3, appending if the file already exists"""

    old_df = load_df_from_s3(path)
    df = pd.concat([old_df, df]) if old_df is not None else df
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = f"{tmp_dir}/temp_file.parquet"

        with open(temp_path, "wb") as f:
            f.write(df.to_parquet())
        s3.upload_file(temp_path, "stock-market-preds", path)


def save_preds_to_s3(predictions, symbol: str, inference_datetime: datetime):

    # get month and year from the datetime
    month = inference_datetime.month
    year = inference_datetime.year

    predictions_path = f"predictions/{symbol}/{symbol}_{year}_{month}.parquet"

    predictions = pd.DataFrame(
        {"inference_datetime": [inference_datetime], "prediction": predictions[-1]}
    )
    predictions.set_index("inference_datetime", inplace=True)

    add_to_s3(predictions, predictions_path)


def save_metrics_to_s3(value: float, symbol: str, inference_datetime: datetime):

    # get month and year from the datetime
    month = inference_datetime.month
    year = inference_datetime.year

    metrics_path = f"metrics/{symbol}/{symbol}_{year}_{month}.parquet"

    metrics = pd.DataFrame(
        {"inference_datetime": [inference_datetime], "abs_error": value}
    )
    metrics.set_index("inference_datetime", inplace=True)

    add_to_s3(metrics, metrics_path)
