import os
from typing import TypeVar, Callable, Any, cast
from joblib import Memory
from functools import cache
from dotenv import load_dotenv

load_dotenv()

MEMORY = Memory("./.cachedir", verbose=0)
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "0") == "1"

T = TypeVar("T", bound=Callable[..., Any])


def persistent_inmemory_cache(func: T) -> T:
    """
    Wraps a function with both file cache (for persistent cache) and in-memory cache (for speed).
    """
    return cast(T, cache(MEMORY.cache(func)) if ENABLE_CACHE else func)
