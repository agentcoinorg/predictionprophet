import os
from typing import TypeVar, Callable, Any, cast
from joblib import Memory
from functools import cache

MEMORY = Memory("./.cachedir", verbose=0)
DISABLE_CACHE = os.getenv("DISABLE_CACHE", "0") == "1"

T = TypeVar("T", bound=Callable[..., Any])


def persisted_inmemory_cache(func: T) -> T:
    """
    Wraps a function with both file cache (for persistent cache) and in-memory cache (for speed).
    """
    return cast(T, cache(MEMORY.cache(func)) if not DISABLE_CACHE else func)
