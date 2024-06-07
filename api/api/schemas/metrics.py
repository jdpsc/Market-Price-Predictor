from typing import List, Any
from datetime import datetime

from pydantic import BaseModel


class Metrics(BaseModel):
    datetime: List[datetime]
    metric: List[float]