from datetime import datetime
import argparse

from dotenv import load_dotenv
import pandas as pd

from model_inference.utils import get_logger
from model_inference.data import (
    load_data_from_feature_store,
    save_metrics_to_s3,
    load_df_from_s3,
)

logger = get_logger(__name__)

load_dotenv()


def calculate_metrics(
    symbol: str, inference_datetime: datetime, feature_view_version: int = 1
):
    """Calculate metrics for a given symbol and inference datetime. Save the metrics to S3.
    Args:
        symbol: Stock symbol.
        inference_datetime: Inference datetime.
        feature_view_version: Feature view version.
    """

    # If needed, add the time to the inference_datetime as 00:00:00.
    inference_datetime = datetime(inference_datetime.year, inference_datetime.month, inference_datetime.day, 0, 0, 0)

    logger.info(
        f"Calculating metrics for symbol {symbol} for {inference_datetime} with feature_view_version={feature_view_version}"
    )

    logger.info("Loading data")
    # TODO: Can be optimized to load only the necessary data, review Hopsworks feature store API
    start_datetime = pd.to_datetime(inference_datetime) - pd.Timedelta(days=1)
    end_datetime = pd.to_datetime(inference_datetime) + pd.Timedelta(days=1)
    data = load_data_from_feature_store(
        symbol=symbol,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        feature_view_version=feature_view_version,
    )
    logger.info("Loaded data from feature store")

    data = data.set_index("date")
    # Convert the index to datetime, extract only the date without the time
    data.index = pd.to_datetime(pd.to_datetime(data.index).date)

    month = inference_datetime.month
    year = inference_datetime.year
    predictions_path = f"predictions/{symbol}/{symbol}_{year}_{month}.parquet"

    logger.info("Loading predictions from S3")
    df_preds = load_df_from_s3(predictions_path)
    if df_preds is None or len(df_preds) == 0:
        logger.info("Haven't found any predictions. Exiting...")
        return
    df_preds.index = pd.to_datetime(pd.to_datetime(df_preds.index).date)
    logger.info("Loaded predictions from S3")

    if not inference_datetime in df_preds.index:
        logger.info(
            f"Couldn't find {inference_datetime} in the predictions. Exiting..."
        )
        logger.info(f"Available dates: {df_preds.index}")
        return
    df_preds = df_preds.loc[[inference_datetime]]

    if not inference_datetime in data.index:
        logger.info(
            f"Couldn't find {inference_datetime} in the grouth truth data. Exiting..."
        )
        logger.info(f"Available dates: {data.index}")
        return
    data = data.loc[[inference_datetime]]

    logger.info("Calculating metrics")
    abs_error_value = abs(data["close"].values[0] - df_preds["prediction"].values[0])
    # if the value is NaN, return (happens in holidays or weekends when there is no data to compare with)
    if pd.isna(abs_error_value):
        logger.info("The absolute error value is NaN. Exiting...")
        return
    logger.info("Calculated metrics")

    logger.info("Saving metrics to S3")
    save_metrics_to_s3(abs_error_value, symbol, inference_datetime)
    logger.info("Successfully saved metrics")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Market Price Predictor Monitoring Pipeline")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument(
        "--inference_date",
        type=str,
        default="2024-03-01",
        help="Inference date in format YYYY-MM-DD",
    )
    parser.add_argument(
        "--feature_group_version", type=int, default=1, help="Feature group version"
    )

    args = parser.parse_args()

    calculate_metrics(
        args.symbol,
        datetime.strptime(args.inference_date, "%Y-%m-%d"),
        args.feature_group_version,
    )
