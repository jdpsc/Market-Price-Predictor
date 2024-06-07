from datetime import datetime
import argparse

from dotenv import load_dotenv
import pandas as pd

from model_inference.utils import get_logger
from model_inference.data import load_data_from_feature_store, save_preds_to_s3
from model_inference.model import make_predictions, load_model_from_registry

logger = get_logger(__name__)

load_dotenv()


def run_inference(
    symbol: str, inference_datetime: datetime, feature_view_version: int = 1
):
    """ Run the inference pipeline, saving the predictions to S3.
    args:
        symbol: Stock symbol.
        inference_datetime: Inference datetime.
        feature_view_version: Feature view version.
    """

    logger.info(f"Running inference pipeline for symbol {symbol}.")

    model_name = f"market_price_predictor_symbol_{symbol}"
    logger.info(f"Loading model from model registry: {model_name}")
    model = load_model_from_registry(model_name=model_name)
    logger.info("Successfully loaded model from model registry.")

    start_datetime = pd.to_datetime(inference_datetime) - pd.Timedelta(days=14)
    end_datetime = pd.to_datetime(inference_datetime)

    logger.info(f"Loading data from {start_datetime} to {end_datetime}")
    data = load_data_from_feature_store(
        symbol=symbol,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        feature_view_version=feature_view_version,
    )
    logger.info("Loaded data from feature store")

    logger.info("Making predictions")
    predictions = make_predictions(model=model, data=data)
    logger.info("Successfully made predictions")

    logger.info("Saving predictions")
    save_preds_to_s3(
        predictions=predictions, symbol=symbol, inference_datetime=inference_datetime
    )
    logger.info("Successfully saved predictions")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Market Price Prediction Pipeline")
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

    run_inference(
        args.symbol,
        datetime.strptime(args.inference_date, "%Y-%m-%d"),
        args.feature_group_version,
    )
