from datetime import datetime
import argparse

from dotenv import load_dotenv

from data_ingest.etl import extract, transform, load, validate
from data_ingest.utils import get_logger

logger = get_logger(__name__)

load_dotenv()


def run_pipeline(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d",
    feature_group_version: int = 1,
) -> dict:
    """
    Args:
        symbol: Stock symbol.
        start_date: Start date.
        end_date: End date.
        interval: Data interval.
        feature_group_version: Feature group version.

    Returns: metadata from the data extraction.
    """


    logger.info(
        f"Running pipeline from {start_date} to {end_date} for {symbol} at {interval} interval"
    )

    # Extract
    data, metadata = extract.stock_data_from_package(
        symbol, start_date, end_date, interval
    )

    logger.info(
        f"Extracted data from {metadata['df_start_date']} to {metadata['df_end_date']} with missing data rate of {metadata['missing data rate']}"
    )

    # Transform
    logger.info("Transforming data")
    data = transform.transform_stock_data(data, symbol)
    logger.info("Data transformed")

    # Validate Shape, names, nulls, ranges, etc
    logger.info("Building validation expectation suite")
    validation_expectation_suite = validate.build_expectation_suite()
    logger.info("Successfully built validation expectation suite")

    # Load
    logger.info("Loading data")
    load.stock_data_feature_store(
        data, feature_group_version, validation_expectation_suite
    )
    logger.info("Data loaded")

    metadata["feature_group_version"] = feature_group_version

    logger.info(f"Pipeline completed for {symbol}")

    return metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Market Price Predictor Data Ingestion Pipeline")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument(
        "--start_date",
        type=str,
        default="2021-01-01",
        help="Start date in format YYYY-MM-DD",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2024-06-01",
        help="End date in format YYYY-MM-DD",
    )
    parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    parser.add_argument(
        "--feature_group_version", type=int, default=1, help="Feature group version"
    )

    args = parser.parse_args()

    run_pipeline(
        args.symbol,
        datetime.strptime(args.start_date, "%Y-%m-%d"),
        datetime.strptime(args.end_date, "%Y-%m-%d"),
        args.interval,
        args.feature_group_version,
    )
