from __future__ import annotations

import os
from io import BytesIO
from typing import Any
from urllib.parse import quote

import joblib
import requests

from modules.logger import get_logger

logger = get_logger(__name__)

LIGHTGBM_BATCH_SCHEMA_VERSION = 1
LIVE_RETURN_HORIZONS = (30, 180)
DEFAULT_LIVE_RELEASE_TAG = "lightgbm-model-live"
DEFAULT_LIVE_MANIFEST_ASSET_NAME = "latest.json"
DEFAULT_BATCH_RELEASE_PREFIX = "lightgbm-batch"
DEFAULT_BATCH_ASSET_NAME = "lightgbm_return_model_batch.joblib"
DEFAULT_RELEASE_REPOSITORY = "bkabwe/bk-fin-and-inv"
LIGHTGBM_RELEASE_REPOSITORY_ENV = "LIGHTGBM_RELEASE_REPOSITORY"
LIGHTGBM_LIVE_MANIFEST_URL_ENV = "LIGHTGBM_LIVE_MANIFEST_URL"

# GitHub enforces a hard 2 GiB (2,147,483,648 byte) limit per release asset. A
# full-universe batch of trained models can exceed that, so the batch file is
# split into parts safely under this limit before upload and reassembled by
# concatenation on download.
GITHUB_RELEASE_ASSET_MAX_BYTES = 2 * 1024 * 1024 * 1024
BATCH_ASSET_SPLIT_CHUNK_BYTES = 1_900_000_000


def batch_asset_part_name(base_name: str, index: int, total: int) -> str:
    clean_name = str(base_name).strip()
    if int(total) <= 1:
        return clean_name
    return f"{clean_name}.part{int(index):03d}of{int(total):03d}"


def batch_asset_part_count(size_bytes: int, *, chunk_size: int = BATCH_ASSET_SPLIT_CHUNK_BYTES) -> int:
    size_bytes = int(size_bytes)
    chunk_size = int(chunk_size)
    if size_bytes <= 0 or chunk_size <= 0:
        return 1
    return max(1, -(-size_bytes // chunk_size))


def resolve_release_repository(repository: str | None = None) -> str:
    return str(
        repository
        or os.getenv(LIGHTGBM_RELEASE_REPOSITORY_ENV)
        or os.getenv("GITHUB_REPOSITORY")
        or DEFAULT_RELEASE_REPOSITORY
        or ""
    ).strip()


def build_release_asset_url(repository: str, tag: str, asset_name: str) -> str:
    repo = str(repository or "").strip().strip("/")
    if not repo:
        raise ValueError("repository is required to build a release asset URL")
    return f"https://github.com/{repo}/releases/download/{quote(str(tag).strip())}/{quote(str(asset_name).strip())}"


def default_live_manifest_url(repository: str | None = None) -> str | None:
    repo = resolve_release_repository(repository)
    if not repo:
        return None
    return build_release_asset_url(repo, DEFAULT_LIVE_RELEASE_TAG, DEFAULT_LIVE_MANIFEST_ASSET_NAME)


def fetch_live_manifest(
    *,
    manifest_url: str | None = None,
    repository: str | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    url = str(manifest_url or os.getenv(LIGHTGBM_LIVE_MANIFEST_URL_ENV) or default_live_manifest_url(repository) or "").strip()
    if not url:
        return {}
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def create_return_model_batch(
    *,
    batch_id: str,
    models: dict[str, dict[int, object]],
    training_metadata: dict[str, dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    normalized_models: dict[str, dict[int, object]] = {}
    for ticker, ticker_models in (models or {}).items():
        clean_ticker = str(ticker or "").strip().upper()
        if not clean_ticker or not isinstance(ticker_models, dict):
            continue
        normalized_models[clean_ticker] = {int(horizon): model for horizon, model in ticker_models.items()}
    normalized_meta = {str(ticker).strip().upper(): dict(meta or {}) for ticker, meta in (training_metadata or {}).items() if str(ticker).strip()}
    return {
        "schema_version": LIGHTGBM_BATCH_SCHEMA_VERSION,
        "batch_id": str(batch_id).strip(),
        "created_at": str(created_at).strip() if created_at else None,
        "models": normalized_models,
        "training_metadata": normalized_meta,
    }


def load_return_model_batch(source) -> dict[str, Any]:
    payload = joblib.load(source)
    return payload if isinstance(payload, dict) else {}


def load_return_model_batch_from_urls(urls: list[str], *, timeout: int = 30) -> dict[str, Any]:
    """Download one or more release assets and concatenate them before loading.

    Batches larger than GitHub's 2 GiB per-asset limit are uploaded as multiple
    ordered parts (see `batch_asset_part_name`); this reassembles them in order
    into a single in-memory buffer before deserializing.
    """
    buffer = BytesIO()
    for url in urls:
        response = requests.get(str(url).strip(), timeout=timeout, stream=True)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                buffer.write(chunk)
    buffer.seek(0)
    return load_return_model_batch(buffer)


def load_return_model_batch_from_url(url: str, *, timeout: int = 30) -> dict[str, Any]:
    return load_return_model_batch_from_urls([url], timeout=timeout)


def get_batch_models_for_ticker(
    batch: dict[str, Any] | None,
    ticker: str,
    *,
    horizons: tuple[int, ...] = LIVE_RETURN_HORIZONS,
) -> dict[int, object]:
    models_by_ticker = (batch or {}).get("models") or {}
    ticker_models = models_by_ticker.get(str(ticker or "").strip().upper()) or {}
    return {int(horizon): ticker_models[int(horizon)] for horizon in horizons if int(horizon) in ticker_models}


def get_batch_training_metadata(batch: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    metadata = (batch or {}).get("training_metadata") or {}
    return {
        str(ticker).strip().upper(): dict(value or {})
        for ticker, value in metadata.items()
        if str(ticker).strip()
    }


def merge_current_and_previous_batches(
    current_batch: dict[str, Any] | None,
    *,
    previous_manifest: dict[str, Any] | None = None,
    previous_batch: dict[str, Any] | None = None,
    max_stale_cycles: int = 4,
) -> tuple[dict[str, dict[int, object]], dict[str, dict[str, Any]], list[str], list[str]]:
    # `live_models` below always rebuilds fresh per-ticker dicts from `current_models`
    # rather than mutating it in place, so a defensive copy.deepcopy() here is not
    # required for correctness. The trained model objects are immutable after
    # training, and deep-copying them was needlessly duplicating the entire
    # (multi-GB, for a full production batch) model set in memory during
    # reduce-promote, which contributed to out-of-memory failures.
    current_models = (current_batch or {}).get("models") or {}
    current_meta = get_batch_training_metadata(current_batch)
    batch_created_at = (current_batch or {}).get("created_at")
    live_models: dict[str, dict[int, object]] = {
        str(ticker).strip().upper(): {int(horizon): model for horizon, model in (ticker_models or {}).items()}
        for ticker, ticker_models in current_models.items()
        if str(ticker).strip()
    }
    live_meta: dict[str, dict[str, Any]] = {}
    for ticker, meta in current_meta.items():
        merged = dict(meta or {})
        merged["last_trained"] = str(merged.get("last_trained") or batch_created_at or "").strip() or None
        merged["cycles_since_training"] = 0
        live_meta[ticker] = merged

    previous_manifest_tickers = ((previous_manifest or {}).get("tickers") or {}) if isinstance(previous_manifest, dict) else {}
    previous_models = ((previous_batch or {}).get("models") or {}) if isinstance(previous_batch, dict) else {}
    previous_meta = get_batch_training_metadata(previous_batch)
    carried_forward: list[str] = []
    dropped_stale: list[str] = []

    for ticker, prior_state in previous_manifest_tickers.items():
        clean_ticker = str(ticker or "").strip().upper()
        if not clean_ticker or clean_ticker in live_models:
            continue
        ticker_models = previous_models.get(clean_ticker)
        if not isinstance(ticker_models, dict) or not ticker_models:
            dropped_stale.append(clean_ticker)
            continue
        next_cycles = int((prior_state or {}).get("cycles_since_training") or 0) + 1
        if next_cycles > int(max_stale_cycles):
            dropped_stale.append(clean_ticker)
            continue
        live_models[clean_ticker] = {int(horizon): model for horizon, model in ticker_models.items()}
        merged = dict(previous_meta.get(clean_ticker) or {})
        merged["exchange"] = (prior_state or {}).get("exchange") or merged.get("exchange")
        merged["last_trained"] = (prior_state or {}).get("last_trained") or merged.get("last_trained")
        merged["cycles_since_training"] = next_cycles
        live_meta[clean_ticker] = merged
        carried_forward.append(clean_ticker)

    return live_models, live_meta, sorted(carried_forward), sorted(set(dropped_stale))


def build_live_manifest(
    *,
    repository: str,
    batch_id: str,
    batch_release_tag: str,
    batch_asset_name: str,
    created_at: str,
    training_metadata: dict[str, dict[str, Any]],
    asset_part_count: int = 1,
) -> dict[str, Any]:
    part_count = max(1, int(asset_part_count))
    batch_asset_urls = [
        build_release_asset_url(repository, batch_release_tag, batch_asset_part_name(batch_asset_name, index, part_count))
        for index in range(1, part_count + 1)
    ]
    tickers: dict[str, dict[str, Any]] = {}
    for ticker, meta in sorted(training_metadata.items()):
        row_counts = {str(horizon): int(value) for horizon, value in dict(meta.get("row_counts") or {}).items()}
        trained_horizons = sorted(int(value) for value in (meta.get("trained_horizons") or LIVE_RETURN_HORIZONS))
        tickers[str(ticker).strip().upper()] = {
            "exchange": meta.get("exchange"),
            "last_trained": meta.get("last_trained"),
            "cycles_since_training": int(meta.get("cycles_since_training") or 0),
            "trained_horizons": trained_horizons,
            "row_counts": row_counts,
            "batch_id": str(batch_id).strip(),
        }
    return {
        "schema_version": LIGHTGBM_BATCH_SCHEMA_VERSION,
        "generated_at": str(created_at).strip(),
        "latest_batch": {
            "batch_id": str(batch_id).strip(),
            "release_tag": str(batch_release_tag).strip(),
            "asset_name": str(batch_asset_name).strip(),
            "asset_url": batch_asset_urls[0],
            "asset_count": part_count,
            "asset_urls": batch_asset_urls,
        },
        "tickers": tickers,
    }


def batch_asset_urls_from_manifest(manifest: dict[str, Any] | None) -> list[str]:
    """Return the ordered list of release asset URLs for the promoted batch.

    Prefers the multi-part `asset_urls` list (present once a batch has been
    split across more than one release asset) and falls back to the single
    `asset_url` for older/simple manifests.
    """
    latest_batch = ((manifest or {}).get("latest_batch") or {}) if isinstance(manifest, dict) else {}
    urls = latest_batch.get("asset_urls")
    if isinstance(urls, list) and urls:
        cleaned = [str(url).strip() for url in urls if str(url).strip()]
        if cleaned:
            return cleaned
    single = str(latest_batch.get("asset_url") or "").strip()
    return [single] if single else []
