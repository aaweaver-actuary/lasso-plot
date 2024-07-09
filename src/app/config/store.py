from dash import dcc
from dataclasses import dataclass
from uuid import uuid4

@dataclass
class Store:
    cache_location: str = "app/.cache"

    def __post_init__(self):
        self.session_id = str(uuid4())
        self.id = f"{self.session_id}-store"

    def __call__(self):
        return dcc.Store(id=self.id, data=self.data)