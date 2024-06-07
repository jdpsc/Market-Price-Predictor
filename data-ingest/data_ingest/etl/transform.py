import pandas as pd


def transform_stock_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:

    df = rename_columns(df)
    df = cast_columns(df)
    df = resample_data(df)
    df = create_columns(df, symbol)

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:

    # drop unneeded columns
    df = df.drop(columns=["Dividends", "Stock Splits"])

    # rename columns
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    return df


def create_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:

    # create new columns
    df["s"] = symbol
    df["date"] = df.index

    return df


def resample_data(df: pd.DataFrame) -> pd.DataFrame:

    # resample data
    df = df.resample("1d").mean()

    return df


def cast_columns(df: pd.DataFrame) -> pd.DataFrame:

    # cast columns
    df["volume"] = df["volume"].astype("float64")

    return df
