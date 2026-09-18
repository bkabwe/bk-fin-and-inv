"""Shared TTL cache fallback for non-Streamlit (e.g. FastAPI/CLI) deployments.

``st.cache_data`` already hashes ``DataFrame``/``Series``/``ndarray`` arguments
by content when Streamlit is available. Outside of Streamlit, several modules
previously defined their own ad-hoc ``cache_data`` fallback that built the
cache key from ``hash((args, kwargs))`` directly; whenever an argument (most
notably a price-history ``DataFrame``) wasn't hashable, that key construction
raised ``TypeError`` and callers either bypassed the cache entirely (silently
recomputing every call) or crashed outright.

This module provides a single ``ttl_cache`` decorator that mirrors
``st.cache_data``'s behavior by deriving a stable, content-based cache key for
common unhashable argument types (DataFrame, Series, ndarray, dict, list,
set) before falling back to skipping the cache only if a truly exotic,
unfreezable argument is encountered.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

import numpy as np
import pandas as pd


def _freeze(value: Any) -> Any:
    """Return a hashable representation of ``value`` for cache-key purposes.

    DataFrames/Series are reduced to a content hash (via
    ``pandas.util.hash_pandas_object``) combined with their shape and index
    bounds, so two calls with equal data hit the same cache entry while two
    calls with different data (even same shape) miss. ndarrays are hashed via
    their raw bytes. Containers (dict/list/set/tuple) are frozen recursively.
    Anything already hashable is returned unchanged.
    """
    if isinstance(value, (pd.DataFrame, pd.Series)):
        try:
            row_hash = hashlib.sha256(pd.util.hash_pandas_object(value, index=True).values.tobytes()).hexdigest()
        except Exception:
            row_hash = repr(value.values.tobytes()) if hasattr(value, "values") else repr(value)
        index = value.index
        index_bounds = (repr(index[0]), repr(index[-1])) if len(index) else (None, None)
        columns = tuple(value.columns) if isinstance(value, pd.DataFrame) else None
        return ("__frame__", value.shape, columns, index_bounds, row_hash)
    if isinstance(value, np.ndarray):
        return ("__ndarray__", value.shape, str(value.dtype), hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict):
        return ("__dict__", tuple(sorted((_freeze(k), _freeze(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        marker = "__list__" if isinstance(value, list) else "__tuple__"
        return (marker, tuple(_freeze(v) for v in value))
    if isinstance(value, set):
        return ("__set__", tuple(sorted(_freeze(v) for v in value)))
    try:
        hash(value)
        return value
    except TypeError:
        # Exotic unhashable type we don't know how to freeze -- surface it
        # unchanged; the caller falls back to uncached execution.
        return value


def _make_key(args: tuple, kwargs: dict) -> Any | None:
    """Build a hashable cache key from ``args``/``kwargs``, or ``None`` if a
    stable hashable representation can't be produced."""
    try:
        frozen_args = tuple(_freeze(a) for a in args)
        frozen_kwargs = tuple(sorted((k, _freeze(v)) for k, v in kwargs.items()))
        key = (frozen_args, frozen_kwargs)
        hash(key)
        return key
    except TypeError:
        return None


def ttl_cache(ttl: int | None = None):
    """TTL-aware, stampede-safe cache decorator for non-Streamlit deployments.

    Content-hashes DataFrame/Series/ndarray/dict/list/set arguments so calls
    with equal data reuse a cached result, mirroring ``st.cache_data``'s
    behavior. Only falls back to uncached execution if a key still can't be
    derived (a genuinely unhashable/unfreezable argument type).
    """

    def decorator(func):
        _cache: dict = {}
        _inflight: dict = {}
        _lock = threading.Lock()

        def wrapper(*args, **kwargs):
            key = _make_key(args, kwargs)
            if key is None:
                return func(*args, **kwargs)

            while True:
                now = time.monotonic()
                with _lock:
                    entry = _cache.get(key)
                    if entry is not None:
                        value, ts = entry
                        if ttl is None or (now - ts) < ttl:
                            return value
                    event = _inflight.get(key)
                    if event is None:
                        ev = threading.Event()
                        _inflight[key] = ev
                        break
                event.wait(timeout=300)

            try:
                result = func(*args, **kwargs)
                with _lock:
                    _cache[key] = (result, time.monotonic())
                return result
            finally:
                with _lock:
                    ev = _inflight.pop(key, None)
                if ev is not None:
                    ev.set()

        return wrapper

    return decorator


# Backwards-compatible alias matching the name previously defined locally in
# each module (``cache_data``) so call sites only need to change the import.
cache_data = ttl_cache
