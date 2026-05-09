import os
from dataclasses import dataclass

@dataclass(slots=True)
class Config:
    model:str
