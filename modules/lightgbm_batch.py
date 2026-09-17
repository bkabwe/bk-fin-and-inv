from __future__ import annotations

import os
import tempfile
from pathlib import Path
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


def batch_shard_asset_name(base_name: str, shard_index: int, shard_count: int) -> str:
    """Filename for one ticker-partitioned storage shard of the combined batch.

    Ticker-partitioned shards let a scan/scoring consumer download only the
    slice of tickers it needs instead of the full combined batch asset (see
    `assign_ticker_shards`/`partition_batch_by_shard`). This is a distinct,
    orthogonal split from `batch_asset_part_name`'s byte-range parts, which
    exist solely to stay under GitHub's 2 GiB per-asset limit -- a shard can
    itself still be split into byte-range parts if it happens to exceed that
    limit. Returns `base_name` unchanged when there is only one shard.
    """
    clean_name = str(base_name).strip()
    if int(shard_count) <= 1:
        return clean_name
    path = Path(clean_name)
    return f"{path.stem}.shard{int(shard_index):03d}of{int(shard_count):03d}{path.suffix}"


def assign_ticker_shards(tickers: list[str] | set[str], shard_count: int) -> dict[str, int]:
    """Deterministically assign each ticker to a storage shard index.

    Tickers are sorted alphabetically before distributing round-robin
    (`idx % shard_count`) so the assignment depends only on the final live
    ticker set -- not discovery/training order -- and stays evenly balanced
    as tickers are added or dropped between promotions.
    """
    shard_count = max(1, int(shard_count))
    ordered = sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})
    return {ticker: idx % shard_count for idx, ticker in enumerate(ordered)}


def partition_batch_by_shard(
    models: dict[str, dict[int, object]],
    metadata: dict[str, dict[str, Any]],
    ticker_shards: dict[str, int],
    shard_count: int,
) -> dict[int, tuple[dict[str, dict[int, object]], dict[str, dict[str, Any]]]]:
    """Split a combined model/metadata set into per-shard subsets.

    Each shard's dicts reference the same model/metadata objects as the
    combined set (no copying), so partitioning a multi-GB batch does not
    meaningfully increase peak memory.
    """
    shard_count = max(1, int(shard_count))
    shards: dict[int, tuple[dict[str, dict[int, object]], dict[str, dict[str, Any]]]] = {
        index: ({}, {}) for index in range(shard_count)
    }
    for ticker, ticker_models in models.items():
        shard_index = int(ticker_shards.get(ticker, 0)) % shard_count
        shard_models, shard_meta = shards[shard_index]
        shard_models[ticker] = ticker_models
        if ticker in metadata:
            shard_meta[ticker] = metadata[ticker]
    return shards


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
    into a temp file on disk (rather than an in-memory buffer) before
    deserializing. Writing to disk means the raw downloaded bytes and the
    deserialized python objects are never both resident in memory at once --
    for a multi-GB batch, buffering the full download in memory before
    `joblib.load()` deserialized a second full copy on top of it was a direct
    contributor to reduce-promote out-of-memory failures.
    """
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_fh:
        tmp_path = Path(tmp_fh.name)
        for url in urls:
            response = requests.get(str(url).strip(), timeout=timeout, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    tmp_fh.write(chunk)
    try:
        return load_return_model_batch(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


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


def extract_carry_forward_state(
    *,
    current_tickers: set[str],
    previous_manifest: dict[str, Any] | None = None,
    previous_batch: dict[str, Any] | None = None,
    max_stale_cycles: int = 4,
) -> tuple[dict[str, dict[int, object]], dict[str, dict[str, Any]], list[str], list[str]]:
    """Extract only the previous-release models that would be carried forward.

    This depends only on `current_tickers` (a cheap set of ticker strings) --
    not the current batch's full model set -- so callers can free the
    (multi-GB) current model set from memory before loading the (also
    multi-GB) previous batch, and only need to hold the small carried-forward
    subset returned here alongside the current models afterward. Holding both
    full batches in memory at once was a direct contributor to reduce-promote
    out-of-memory failures once a previous live release exists to merge
    against.
    """
    previous_manifest_tickers = ((previous_manifest or {}).get("tickers") or {}) if isinstance(previous_manifest, dict) else {}
    previous_models = ((previous_batch or {}).get("models") or {}) if isinstance(previous_batch, dict) else {}
    previous_meta = get_batch_training_metadata(previous_batch)
    carried_models: dict[str, dict[int, object]] = {}
    carried_meta: dict[str, dict[str, Any]] = {}
    carried_forward: list[str] = []
    dropped_stale: list[str] = []

    for ticker, prior_state in previous_manifest_tickers.items():
        clean_ticker = str(ticker or "").strip().upper()
        if not clean_ticker or clean_ticker in current_tickers:
            continue
        ticker_models = previous_models.get(clean_ticker)
        if not isinstance(ticker_models, dict) or not ticker_models:
            dropped_stale.append(clean_ticker)
            continue
        next_cycles = int((prior_state or {}).get("cycles_since_training") or 0) + 1
        if next_cycles > int(max_stale_cycles):
            dropped_stale.append(clean_ticker)
            continue
        carried_models[clean_ticker] = {int(horizon): model for horizon, model in ticker_models.items()}
        merged = dict(previous_meta.get(clean_ticker) or {})
        merged["exchange"] = (prior_state or {}).get("exchange") or merged.get("exchange")
        merged["last_trained"] = (prior_state or {}).get("last_trained") or merged.get("last_trained")
        merged["cycles_since_training"] = next_cycles
        carried_meta[clean_ticker] = merged
        carried_forward.append(clean_ticker)

    return carried_models, carried_meta, sorted(carried_forward), sorted(set(dropped_stale))


def combine_current_and_carried_forward(
    current_batch: dict[str, Any] | None,
    *,
    carried_models: dict[str, dict[int, object]] | None = None,
    carried_meta: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[int, object]], dict[str, dict[str, Any]]]:
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

    live_models.update(carried_models or {})
    live_meta.update(carried_meta or {})
    return live_models, live_meta


def merge_current_and_previous_batches(
    current_batch: dict[str, Any] | None,
    *,
    previous_manifest: dict[str, Any] | None = None,
    previous_batch: dict[str, Any] | None = None,
    max_stale_cycles: int = 4,
) -> tuple[dict[str, dict[int, object]], dict[str, dict[str, Any]], list[str], list[str]]:
    current_tickers = {
        str(ticker).strip().upper() for ticker in ((current_batch or {}).get("models") or {}) if str(ticker).strip()
    }
    carried_models, carried_meta, carried_forward, dropped_stale = extract_carry_forward_state(
        current_tickers=current_tickers,
        previous_manifest=previous_manifest,
        previous_batch=previous_batch,
        max_stale_cycles=max_stale_cycles,
    )
    live_models, live_meta = combine_current_and_carried_forward(
        current_batch, carried_models=carried_models, carried_meta=carried_meta
    )
    return live_models, live_meta, carried_forward, dropped_stale


def build_live_manifest(
    *,
    repository: str,
    batch_id: str,
    batch_release_tag: str,
    batch_asset_name: str,
    created_at: str,
    training_metadata: dict[str, dict[str, Any]],
    asset_part_count: int = 1,
    scan_shard_count: int = 1,
    ticker_shards: dict[str, int] | None = None,
    shard_asset_part_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    part_count = max(1, int(asset_part_count))
    batch_asset_urls = [
        build_release_asset_url(repository, batch_release_tag, batch_asset_part_name(batch_asset_name, index, part_count))
        for index in range(1, part_count + 1)
    ]

    # Ticker-partitioned storage shards (distinct from the byte-range parts
    # above) let a scan/scoring consumer download only the slice of tickers
    # it needs instead of the full combined batch asset. `shards` is only
    # populated when the caller actually promoted more than one shard;
    # otherwise the manifest looks exactly as it did before this feature.
    shard_count = max(1, int(scan_shard_count))
    ticker_shards = ticker_shards or {}
    shard_asset_part_counts = shard_asset_part_counts or {}
    shards: list[dict[str, Any]] = []
    if shard_count > 1:
        for shard_index in range(shard_count):
            shard_asset_name = batch_shard_asset_name(batch_asset_name, shard_index, shard_count)
            shard_part_count = max(1, int(shard_asset_part_counts.get(shard_index, 1)))
            shard_urls = [
                build_release_asset_url(
                    repository, batch_release_tag, batch_asset_part_name(shard_asset_name, index, shard_part_count)
                )
                for index in range(1, shard_part_count + 1)
            ]
            shards.append(
                {
                    "shard_index": shard_index,
                    "asset_name": shard_asset_name,
                    "asset_count": shard_part_count,
                    "asset_urls": shard_urls,
                }
            )

    tickers: dict[str, dict[str, Any]] = {}
    for ticker, meta in sorted(training_metadata.items()):
        row_counts = {str(horizon): int(value) for horizon, value in dict(meta.get("row_counts") or {}).items()}
        trained_horizons = sorted(int(value) for value in (meta.get("trained_horizons") or LIVE_RETURN_HORIZONS))
        clean_ticker = str(ticker).strip().upper()
        entry: dict[str, Any] = {
            "exchange": meta.get("exchange"),
            "last_trained": meta.get("last_trained"),
            "cycles_since_training": int(meta.get("cycles_since_training") or 0),
            "trained_horizons": trained_horizons,
            "row_counts": row_counts,
            "batch_id": str(batch_id).strip(),
        }
        if shards and clean_ticker in ticker_shards:
            entry["shard_index"] = int(ticker_shards[clean_ticker])
        tickers[clean_ticker] = entry

    latest_batch: dict[str, Any] = {
        "batch_id": str(batch_id).strip(),
        "release_tag": str(batch_release_tag).strip(),
        "asset_name": str(batch_asset_name).strip(),
        "asset_url": batch_asset_urls[0],
        "asset_count": part_count,
        "asset_urls": batch_asset_urls,
    }
    if shards:
        latest_batch["shard_count"] = shard_count
        latest_batch["shards"] = shards

    return {
        "schema_version": LIGHTGBM_BATCH_SCHEMA_VERSION,
        "generated_at": str(created_at).strip(),
        "latest_batch": latest_batch,
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


def shard_asset_urls_from_manifest(manifest: dict[str, Any] | None, ticker: str) -> list[str]:
    """Return the release asset URL(s) for the ticker-partitioned shard holding `ticker`.

    Returns an empty list when the manifest predates shard partitioning, was
    promoted with a single shard, or does not list the requested ticker --
    callers should fall back to `batch_asset_urls_from_manifest()` (the full
    combined batch) in that case.
    """
    if not isinstance(manifest, dict):
        return []
    clean_ticker = str(ticker or "").strip().upper()
    if not clean_ticker:
        return []
    ticker_entry = (manifest.get("tickers") or {}).get(clean_ticker) or {}
    shard_index = ticker_entry.get("shard_index")
    if shard_index is None:
        return []
    shards = ((manifest.get("latest_batch") or {}).get("shards")) or []
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        if int(shard.get("shard_index", -1)) != int(shard_index):
            continue
        urls = shard.get("asset_urls")
        if isinstance(urls, list) and urls:
            cleaned = [str(url).strip() for url in urls if str(url).strip()]
            if cleaned:
                return cleaned
    return []
