from typing import List

from pydantic import BaseModel


class UniqueSymbols(BaseModel):
    values: List[str]