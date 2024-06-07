from typing import Callable, Union

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataFrameTransformer:
    """Class to aggregate all the transformations to be applied to the input data"""

    def __init__(self):
        self.input_transformers = []
        self.input_scaler = None
        self.target_scaler = None

    def add_transformer(self, transformer: Callable):
        """Add a transformer to the list of input transformers"""
        self.input_transformers.append(transformer)

    def set_scalers(
        self,
        input_scaler: Union[StandardScaler, MinMaxScaler, RobustScaler],
        target_scaler: Union[StandardScaler, MinMaxScaler, RobustScaler],
    ):
        """Set the input and target scalers"""
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

    def fit_transform(self, df, target_col: str):
        """Fit and transform the input and output data"""
        transformed_df = df.copy()
        for transformer in self.input_transformers:
            transformed_df = transformer(transformed_df)
        transformed_df = self.input_scaler.fit_transform(transformed_df)
        self.target_scaler.fit(df.loc[:, target_col].values.reshape(-1, 1))

        return transformed_df

    def transform_input(self, df):
        """Transform the input data"""
        transformed_df = df.copy()
        for transformer in self.input_transformers:
            transformed_df = transformer(transformed_df)
        transformed_df = self.input_scaler.transform(transformed_df)
        return transformed_df

    def inverse_target_transform(self, y_pred):
        """Inverse transform the output data"""
        return self.target_scaler.inverse_transform(y_pred)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:

    df["daily_return"] = df["close"].pct_change()

    return df


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(columns=["date", "s"])

    return df
