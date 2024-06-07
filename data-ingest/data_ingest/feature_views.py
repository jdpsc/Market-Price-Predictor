import argparse
from datetime import datetime
import os

import hopsworks
import hsfs
from dotenv import load_dotenv

from data_ingest.utils import get_logger

logger = get_logger(__name__)

load_dotenv()


def create_feature_view(
    start_date: str,
    end_date: str,
    feature_group_version: int = 1,
) -> dict:
    """Create a new feature view version and training dataset
    based on the given feature group version and start and end datetimes.
    """

    # Connect to feature store
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_ID"),
    )

    # Connect to Hopsworks Feature Store
    fs = project.get_feature_store()

    # Delete old feature views to remain in free tier
    try:
        feature_views = fs.get_feature_views(name="stock_data_feature_view")
    except hsfs.client.exceptions.RestAPIError:
        logger.info("No feature views found for stock_data_feature_view.")

        feature_views = []

    for feature_view in feature_views:
        try:
            feature_view.delete_all_training_datasets()
        except hsfs.client.exceptions.RestAPIError:
            logger.error(
                f"Failed to delete training datasets for feature view {feature_view.name} with version {feature_view.version}."
            )

        try:
            feature_view.delete()
        except hsfs.client.exceptions.RestAPIError:
            logger.error(
                f"Failed to delete feature view {feature_view.name} with version {feature_view.version}."
            )

    # Create feature view in the given feature group version
    stock_feature_group = fs.get_feature_group(
        "stock_data", version=feature_group_version
    )

    ds_query = stock_feature_group.select_all()
    feature_view = fs.create_feature_view(
        name="stock_data_feature_view",
        description="Feature view for stock data",
        query=ds_query,
        labels=[],
    )

    # Create training dataset.
    logger.info(f"Creating training dataset between {start_date} and {end_date}.")

    feature_view.create_training_data(
        description="Training dataset for stock data",
        data_format="parquet",
        start_time=start_date,
        end_time=end_date,
        write_options={"wait_for_job": True},
        coalesce=False,
    )

    return {}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature View Creation")
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
    parser.add_argument(
        "--feature_group_version", type=int, default=1, help="Feature group version"
    )

    args = parser.parse_args()

    create_feature_view(
        datetime.strptime(args.start_date, "%Y-%m-%d"),
        datetime.strptime(args.end_date, "%Y-%m-%d"),
        args.feature_group_version,
    )
