import tempfile

import pandas as pd
import boto3


def load_df_from_s3(path: str) -> pd.DataFrame:
    """ Load a DataFrame from S3, if it exists. """

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
