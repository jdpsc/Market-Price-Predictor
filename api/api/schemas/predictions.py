from typing import List, Any
from datetime import datetime

from pydantic import BaseModel


class PredictionResults(BaseModel):
    datetime: List[datetime]
    prediction: List[float]