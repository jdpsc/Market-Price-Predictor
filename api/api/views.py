from typing import Any, List
import os

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import boto3

from api import schemas
from api.config import get_settings
from api.utils import load_df_from_s3

api_router = APIRouter()
security = HTTPBearer()

os.environ["AWS_ACCESS_KEY_ID"] = get_settings().AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = get_settings().AWS_SECRET_ACCESS_KEY


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != get_settings().KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired token",
        )
    return credentials.credentials


@api_router.get(
    "/health",
    response_model=schemas.Health,
    status_code=200,
    dependencies=[Depends(verify_token)],
)
def health() -> dict:
    """
    Health check endpoint.
    """

    health_data = schemas.Health(
        name=get_settings().PROJECT_NAME, api_version=get_settings().VERSION
    )

    return health_data.dict()


@api_router.get(
    "/available_stock_symbols",
    response_model=schemas.UniqueSymbols,
    status_code=200,
    dependencies=[Depends(verify_token)],
)
def available_stock_symbols() -> List:
    """
    Retrieve available stock symbols.
    """

    # list folders in the S3 bucket
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(
        Bucket=get_settings().S3_PREDS_BUCKET, Prefix="predictions/"
    )
    available_stock_symbols_list = []
    for obj in response.get("Contents", []):
        available_stock_symbols_list.append(obj.get("Key").split("/")[1])

    # remove duplicates
    available_stock_symbols_list = list(set(available_stock_symbols_list))

    return {"values": available_stock_symbols_list}


@api_router.get(
    "/predictions/{symbol}/{year}/{month}",
    response_model=schemas.PredictionResults,
    status_code=200,
    dependencies=[Depends(verify_token)],
)
async def get_predictions(symbol: str, year: int, month: int) -> Any:
    """
    Get forecasted predictions based on the given symbol, year, and month.
    """

    predictions_path = f"predictions/{symbol}/{symbol}_{year}_{month}.parquet"

    predictions = load_df_from_s3(predictions_path)

    if predictions is None or len(predictions) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for the given symbol, year, and month: {symbol}, {year}, {month}",
        )

    results = {
        "datetime": predictions.index.to_list(),
        "prediction": predictions["prediction"].to_list(),
    }

    return results


@api_router.get(
    "/metrics/{symbol}/{year}/{month}",
    response_model=schemas.Metrics,
    status_code=200,
    dependencies=[Depends(verify_token)],
)
async def get_metrics(symbol: str, year: int, month: int) -> Any:
    """
    Get metrics based on the given symbol, year, and month.
    """

    metrics_path = f"metrics/{symbol}/{symbol}_{year}_{month}.parquet"

    metrics = load_df_from_s3(metrics_path)

    if metrics is None or len(metrics) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for the given symbol, year, and month: {symbol}, {year}, {month}",
        )

    results = {
        "datetime": metrics.index.to_list(),
        "metric": metrics["abs_error"].to_list(),
    }

    return results
