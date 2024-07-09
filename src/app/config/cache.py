"""Initialize and define the cache for the app."""

from app._app_config import app
from flask_caching import Cache

# Initialize the cache
cache = Cache(
    app.server,
    config={"CACHE_TYPE": "redis", "CACHE_DIR": "app/.cache", "CACHE_THRESHOLD": 50},
)

__all__ = ["cache"]
