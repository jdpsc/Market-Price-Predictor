import os

import pandas as pd
import hopsworks
from hsfs.feature_group import FeatureGroup
from great_expectations.core import ExpectationSuite


def stock_data_feature_store(
    df: pd.DataFrame,
    feature_group_version: int,
    validation_expectation_suite: ExpectationSuite,
) -> FeatureGroup:
    """
    Load stock data into the feature store.
    Args:
        df: Stock data.
        feature_group_version: Feature group version.
        validation_expectation_suite: Validation expectation suite.
    Returns:
        Feature group.
    """

    # Connect to feature store
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_ID"),
    )

    # Connect to Hopsworks Feature Store
    fs = project.get_feature_store()

    stock_feature_group = fs.get_or_create_feature_group(
        name="stock_data",
        version=feature_group_version,
        description="Stock data feature group",
        primary_key="s",
        event_time="date",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    # Insert data into the feature store
    stock_feature_group.insert(
        features=df, overwrite=False, write_options={"wait_for_job": True}
    )

    feature_descriptions = {
        "open": "Opening price",
        "high": "Highest price",
        "low": "Lowest price",
        "close": "Closing price",
        "volume": "Volume",
        "s": "Stock symbol",
        "date": "Date",
    }

    for feature_name, feature_description in feature_descriptions.items():
        stock_feature_group.update_feature_description(
            feature_name, feature_description
        )

    # Obtain statistics for the feature group
    stock_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    stock_feature_group.update_statistics_config()
    stock_feature_group.compute_statistics()

    return stock_feature_group
