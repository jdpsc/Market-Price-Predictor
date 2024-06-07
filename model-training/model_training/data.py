import os

import hopsworks
import pandas as pd


def load_stock_data_from_feature_store(
    symbol: str, feature_view_version: int, train_dataset_version: int
) -> pd.DataFrame:
    """Load stock data from the feature store.
    args:
        symbol: Stock symbol.
        feature_view_version: Feature view version.
        train_dataset_version: Training dataset version.
    returns:
        Stock data.
    """

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

    # get the training dataset, where the s columns is symbol, and order by date
    data, _ = feature_view.get_training_data(
        training_dataset_version=train_dataset_version
    )

    data = data.sort_values("date")
    data = data[data["s"] == symbol]

    return data
