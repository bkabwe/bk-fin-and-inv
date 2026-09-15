from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.data_fetcher import get_all_active_ticker_details, get_stock_data
from modules.feature_engineering import build_feature_table
from modules.fred_client import get_macro_feature_table
from modules.lightgbm_batch import (
    DEFAULT_BATCH_ASSET_NAME,
    DEFAULT_BATCH_RELEASE_PREFIX,
    DEFAULT_LIVE_MANIFEST_ASSET_NAME,
    DEFAULT_LIVE_RELEASE_TAG,
    build_live_manifest,
    create_return_model_batch,
    fetch_live_manifest,
    get_batch_training_metadata,
    load_return_model_batch,
    load_return_model_batch_from_url,
    merge_current_and_previous_batches,
    resolve_release_repository,
)
from modules.lightgbm_model import (
    _label_sanity_bounds,
    build_return_training_examples_for_ticker,
    latest_lightgbm_feature_row,
    predict_forward_return,
    train_return_models,
    training_row_counts,
)
from modules.logger import get_logger
from modules.scoring_engine import fast_screen_score

logger = get_logger(__name__)
LIVE_HORIZONS = (30, 180)
DEFAULT_FAST_SCREEN_MIN_SCORE = 50
DEFAULT_FAST_SCREEN_MARGIN = 15
DEFAULT_SANITY_TICKERS = ("AAPL", "MSFT")
DEFAULT_BATCH_RETENTION = 4
COMMON_STOCK_TYPES = {"CS", "COMMON STOCK", "COMMON_STOCK"}


class GitHubReleaseClient:
    def __init__(self, repository: str, token: str):
        self.repository = str(repository).strip()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "Authorization": "Bearer " + token,
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        self.api_base = f"https://api.github.com/repos/{self.repository}"

    def _request(self, method: str, url: str, **kwargs):
        response = self.session.request(method, url, timeout=60, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub API {method} {url} failed: {response.status_code} {response.text}")
        if not response.content:
            return {}
        payload = response.json()
        return payload if isinstance(payload, dict) else payload

    def get_release_by_tag(self, tag: str) -> dict[str, Any] | None:
        response = self.session.get(f"{self.api_base}/releases/tags/{tag}", timeout=60)
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub API GET release tag {tag} failed: {response.status_code} {response.text}")
        payload = response.json()
        return payload if isinstance(payload, dict) else None

    def ensure_release(self, *, tag: str, name: str, body: str, target_commitish: str | None = None) -> dict[str, Any]:
        existing = self.get_release_by_tag(tag)
        payload = {"tag_name": tag, "name": name, "body": body, "draft": False, "prerelease": False}
        if target_commitish:
            payload["target_commitish"] = target_commitish
        if existing:
            return self._request("PATCH", f"{self.api_base}/releases/{existing['id']}", json=payload)
        return self._request("POST", f"{self.api_base}/releases", json=payload)

    def delete_asset_if_exists(self, release: dict[str, Any], asset_name: str) -> None:
        for asset in release.get("assets", []) or []:
            if str(asset.get("name") or "").strip() == str(asset_name).strip():
                self._request("DELETE", f"{self.api_base}/releases/assets/{asset['id']}")
                return

    def upload_asset(self, release: dict[str, Any], asset_path: Path, asset_name: str) -> dict[str, Any]:
        self.delete_asset_if_exists(release, asset_name)
        upload_url = str(release.get("upload_url") or "").split("{", 1)[0]
        with asset_path.open("rb") as fh:
            response = self.session.post(
                upload_url,
                params={"name": asset_name},
                headers={"Content-Type": "application/octet-stream"},
                data=fh.read(),
                timeout=120,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub asset upload failed: {response.status_code} {response.text}")
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def list_releases(self, *, per_page: int = 100) -> list[dict[str, Any]]:
        response = self.session.get(f"{self.api_base}/releases", params={"per_page": per_page}, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub API list releases failed: {response.status_code} {response.text}")
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def delete_release(self, release_id: int) -> None:
        self._request("DELETE", f"{self.api_base}/releases/{int(release_id)}")

    def delete_tag_ref(self, tag: str) -> None:
        ref = f"tags/{tag}"
        response = self.session.delete(f"{self.api_base}/git/refs/{ref}", timeout=60)
        if response.status_code not in (204, 404):
            raise RuntimeError(f"GitHub tag delete failed for {tag}: {response.status_code} {response.text}")


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _common_stock(row: dict[str, Any]) -> bool:
    ticker_type = " ".join(str(row.get("type") or "").strip().upper().replace("-", " ").replace("_", " ").split())
    return ticker_type in COMMON_STOCK_TYPES


def _chunk_for_shard(items: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    return [item for idx, item in enumerate(items) if idx % shard_count == shard_index]


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def discover(args: argparse.Namespace) -> int:
    discovered_at = _utc_now().isoformat().replace("+00:00", "Z")
    tickers = get_all_active_ticker_details()
    filtered: list[dict[str, Any]] = []
    skipped: dict[str, dict[str, Any]] = {}
    fast_screen_threshold = int(args.fast_screen_min_score) - int(args.fast_screen_margin)

    for row in tickers:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        if not _common_stock(row):
            skipped[ticker] = {"reason": "non_common_stock", "type": row.get("type")}
            continue
        price_data = get_stock_data(ticker, period=args.fast_screen_period, interval=args.fast_screen_interval)
        close = (
            price_data["Close"].dropna()
            if price_data is not None and not price_data.empty and "Close" in price_data
            else pd.Series(dtype="float64")
        )
        current_price = float(close.iloc[-1]) if not close.empty else None
        if current_price is None or float(current_price) < float(args.price_floor):
            skipped[ticker] = {"reason": "price_below_floor", "current_price": current_price}
            continue
        fast_score, err = fast_screen_score(
            ticker,
            period=args.fast_screen_period,
            interval=args.fast_screen_interval,
            data=price_data,
        )
        if err:
            skipped[ticker] = {"reason": "fast_screen_error", "detail": err}
            continue
        if int(fast_score) < fast_screen_threshold:
            skipped[ticker] = {"reason": "fast_screen_below_threshold", "fast_score": int(fast_score)}
            continue
        filtered.append(
            {
                "ticker": ticker,
                "exchange": row.get("primary_exchange"),
                "type": row.get("type"),
                "current_price": float(current_price),
                "fast_score": int(fast_score),
            }
        )

    end_date = _utc_now().date()
    start_date = end_date - timedelta(days=int(args.macro_lookback_days))
    macro_table = get_macro_feature_table(start_date, end_date)
    max_tickers = int(getattr(args, "max_tickers", 0) or 0)
    if max_tickers > 0 and len(filtered) > max_tickers:
        ranked = sorted(filtered, key=lambda item: (-int(item.get("fast_score", 0)), str(item.get("ticker") or "")))
        retained = ranked[:max_tickers]
        for item in ranked[max_tickers:]:
            ticker = str(item.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            skipped[ticker] = {"reason": "max_tickers_cap", "fast_score": int(item.get("fast_score") or 0)}
        filtered = retained

    output_payload = {
        "discovered_at": discovered_at,
        "price_floor": float(args.price_floor),
        "fast_screen_min_score": int(args.fast_screen_min_score),
        "fast_screen_margin": int(args.fast_screen_margin),
        "fast_screen_threshold": int(fast_screen_threshold),
        "max_tickers": max_tickers if max_tickers > 0 else None,
        "tickers": filtered,
        "skipped": skipped,
        "matrix_jobs": int(args.matrix_jobs),
    }
    _json_dump(Path(args.output), output_payload)
    Path(args.macro_output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(macro_table, Path(args.macro_output))
    print(f"Discovered {len(filtered)} trainable pre-row-count tickers from {len(tickers)} active tickers.")
    print(f"Shared macro feature table rows: {len(macro_table)}")
    return 0


def train_shard(args: argparse.Namespace) -> int:
    discovery = json.loads(Path(args.discovery_file).read_text(encoding="utf-8"))
    macro_table = joblib.load(Path(args.macro_file))
    tickers = _chunk_for_shard(list(discovery.get("tickers") or []), int(args.shard_index), int(args.shard_count))
    trained_at = _utc_now().isoformat().replace("+00:00", "Z")
    models_by_ticker: dict[str, dict[int, object]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    expected_tickers: list[str] = []
    skipped: dict[str, dict[str, Any]] = {}

    for row in tickers:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        try:
            examples = build_return_training_examples_for_ticker(
                ticker=ticker,
                period=args.training_period,
                interval=args.training_interval,
                lookback_days=int(args.lookback_days),
                horizons=LIVE_HORIZONS,
                shared_macro_table=macro_table,
            )
        except Exception as exc:
            logger.exception("Training example build failed for %s", ticker)
            skipped[ticker] = {"reason": "training_examples_error", "detail": str(exc)}
            continue

        row_counts = training_row_counts(examples)
        if any(int(row_counts.get(horizon, 0)) < int(args.min_rows_per_horizon) for horizon in LIVE_HORIZONS):
            skipped[ticker] = {"reason": "insufficient_rows", "row_counts": row_counts}
            continue

        expected_tickers.append(ticker)
        models = train_return_models(examples, min_rows_per_horizon=int(args.min_rows_per_horizon))
        if not models or any(int(horizon) not in models for horizon in LIVE_HORIZONS):
            skipped[ticker] = {"reason": "missing_horizon_model", "trained_horizons": sorted(list((models or {}).keys())), "row_counts": row_counts}
            continue
        models_by_ticker[ticker] = {int(horizon): models[int(horizon)] for horizon in LIVE_HORIZONS}
        metadata[ticker] = {
            "exchange": row.get("exchange"),
            "row_counts": {str(horizon): int(row_counts.get(horizon, 0)) for horizon in LIVE_HORIZONS},
            "trained_horizons": list(LIVE_HORIZONS),
            "last_trained": trained_at,
            "fast_score": int(row.get("fast_score") or 0),
            "current_price": float(row.get("current_price") or 0.0),
        }

    batch = create_return_model_batch(
        batch_id=str(args.batch_id),
        models=models_by_ticker,
        training_metadata=metadata,
        created_at=trained_at,
    )
    payload = {
        **batch,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "expected_tickers": sorted(expected_tickers),
        "assigned_tickers": [str(row.get("ticker") or "").strip().upper() for row in tickers if str(row.get("ticker") or "").strip()],
        "skipped": skipped,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    print(
        f"Shard {args.shard_index}/{args.shard_count}: assigned={len(tickers)} expected={len(expected_tickers)} trained={len(models_by_ticker)}"
    )
    return 0


def _merge_partial_batches(partial_files: list[Path]) -> tuple[dict[str, dict[int, object]], dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    models: dict[str, dict[int, object]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    expected: list[str] = []
    extras: dict[str, Any] = {"skipped": {}, "assigned": []}
    for file_path in partial_files:
        payload = load_return_model_batch(file_path)
        extras["assigned"].extend(payload.get("assigned_tickers") or [])
        extras["skipped"].update(payload.get("skipped") or {})
        expected.extend(payload.get("expected_tickers") or [])
        for ticker, ticker_models in (payload.get("models") or {}).items():
            if ticker in models:
                raise RuntimeError(f"Duplicate ticker across partial batches: {ticker}")
            models[ticker] = {int(horizon): model for horizon, model in ticker_models.items()}
        metadata.update(get_batch_training_metadata(payload))
    return models, metadata, sorted(set(expected)), extras


def _validate_expected_horizons(models: dict[str, dict[int, object]], expected_tickers: list[str]) -> None:
    missing = [ticker for ticker in expected_tickers if any(horizon not in (models.get(ticker) or {}) for horizon in LIVE_HORIZONS)]
    if missing:
        raise RuntimeError(f"Expected trained tickers missing one or more horizons: {', '.join(sorted(missing)[:20])}")


def _validate_row_counts(metadata: dict[str, dict[str, Any]], min_rows_per_horizon: int) -> None:
    bad: list[str] = []
    for ticker, meta in metadata.items():
        row_counts = {int(k): int(v) for k, v in dict(meta.get("row_counts") or {}).items()}
        if any(int(row_counts.get(horizon, 0)) < int(min_rows_per_horizon) for horizon in LIVE_HORIZONS):
            bad.append(ticker)
    if bad:
        raise RuntimeError(f"Merged batch includes tickers below row-count threshold: {', '.join(sorted(bad)[:20])}")


def _smoke_test_predictions(models: dict[str, dict[int, object]], macro_table: pd.DataFrame | None, sanity_tickers: tuple[str, ...]) -> None:
    tested: list[str] = []
    for ticker in sanity_tickers:
        if ticker not in models:
            continue
        data = get_stock_data(ticker, period="2y", interval="1d")
        if data is None or data.empty:
            raise RuntimeError(f"Smoke test price history unavailable for {ticker}")
        feature_table = build_feature_table(ticker, data, lookback_days=len(data), shared_macro_table=macro_table)
        latest_row = latest_lightgbm_feature_row(feature_table)
        if latest_row is None:
            raise RuntimeError(f"Smoke test feature row unavailable for {ticker}")
        for horizon in LIVE_HORIZONS:
            prediction = predict_forward_return(models[ticker].get(horizon), latest_row)
            if prediction is None:
                raise RuntimeError(f"Smoke test prediction missing for {ticker} horizon {horizon}")
            min_label, max_label = _label_sanity_bounds(horizon)
            if not (float(min_label) <= float(prediction) <= float(max_label)):
                raise RuntimeError(
                    f"Smoke test prediction out of bounds for {ticker} horizon {horizon}: {prediction} not in [{min_label}, {max_label}]"
                )
        tested.append(ticker)
    if not tested:
        raise RuntimeError("No sanity tickers were present in the merged batch for smoke testing")


def _load_previous_live_state(repository: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        manifest = fetch_live_manifest(repository=repository)
    except Exception as exc:
        logger.warning("Previous live manifest unavailable: %s", exc)
        return {}, {}
    asset_url = str((((manifest or {}).get("latest_batch") or {}).get("asset_url") or "")).strip()
    if not asset_url:
        return manifest, {}
    try:
        batch = load_return_model_batch_from_url(asset_url)
    except Exception as exc:
        logger.warning("Previous live batch unavailable: %s", exc)
        return manifest, {}
    return manifest, batch


def _promote_outputs(args: argparse.Namespace, batch_path: Path, manifest_path: Path, release_tag: str) -> None:
    repository = resolve_release_repository(args.repository)
    token = str(os.getenv("GITHUB_TOKEN") or "").strip()
    if not repository or not token:
        raise RuntimeError("Promotion requires repository and GITHUB_TOKEN")
    client = GitHubReleaseClient(repository, token)
    batch_release = client.ensure_release(
        tag=release_tag,
        name=release_tag,
        body=f"Automated LightGBM batch promotion for {release_tag}",
        target_commitish=os.getenv("GITHUB_SHA"),
    )
    client.upload_asset(batch_release, batch_path, DEFAULT_BATCH_ASSET_NAME)
    live_release = client.ensure_release(
        tag=DEFAULT_LIVE_RELEASE_TAG,
        name=DEFAULT_LIVE_RELEASE_TAG,
        body="Stable pointer to the latest promoted LightGBM batch manifest.",
        target_commitish=os.getenv("GITHUB_SHA"),
    )
    client.upload_asset(live_release, manifest_path, DEFAULT_LIVE_MANIFEST_ASSET_NAME)


def reduce_and_validate(args: argparse.Namespace) -> int:
    partial_files = sorted(Path(args.partial_dir).glob("*.joblib"))
    if not partial_files:
        raise RuntimeError(f"No partial batch files found under {args.partial_dir}")
    macro_table = joblib.load(Path(args.macro_file)) if args.macro_file else None
    repository = resolve_release_repository(args.repository)
    created_at = _utc_now().isoformat().replace("+00:00", "Z")
    batch_id = str(args.batch_id).strip()
    current_models, current_meta, expected_tickers, extras = _merge_partial_batches(partial_files)
    _validate_expected_horizons(current_models, expected_tickers)
    _validate_row_counts(current_meta, int(args.min_rows_per_horizon))

    current_batch = create_return_model_batch(
        batch_id=batch_id,
        models=current_models,
        training_metadata=current_meta,
        created_at=created_at,
    )
    merged_batch_path = Path(args.output_batch)
    merged_batch_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(current_batch, merged_batch_path)
    reloaded = load_return_model_batch(merged_batch_path)
    if not reloaded.get("models"):
        raise RuntimeError("Merged batch failed clean joblib reload")
    _smoke_test_predictions(current_models, macro_table, tuple(args.sanity_tickers))

    previous_manifest, previous_batch = _load_previous_live_state(repository)
    live_models, live_meta, carried_forward, dropped_stale = merge_current_and_previous_batches(
        current_batch,
        previous_manifest=previous_manifest,
        previous_batch=previous_batch,
        max_stale_cycles=int(args.max_stale_cycles),
    )
    release_tag = f"{DEFAULT_BATCH_RELEASE_PREFIX}-{batch_id}"
    live_batch = create_return_model_batch(
        batch_id=batch_id,
        models=live_models,
        training_metadata=live_meta,
        created_at=created_at,
    )
    joblib.dump(live_batch, merged_batch_path)
    manifest = build_live_manifest(
        repository=repository,
        batch_id=batch_id,
        batch_release_tag=release_tag,
        batch_asset_name=DEFAULT_BATCH_ASSET_NAME,
        created_at=created_at,
        training_metadata=live_meta,
    )
    manifest["current_run"] = {
        "expected_tickers": expected_tickers,
        "trained_tickers": sorted(current_models.keys()),
        "carried_forward_tickers": carried_forward,
        "dropped_stale_tickers": dropped_stale,
        "skipped": extras.get("skipped") or {},
    }
    manifest_path = Path(args.output_manifest)
    _json_dump(manifest_path, manifest)
    print(
        f"Merged current={len(current_models)} live={len(live_models)} carried_forward={len(carried_forward)} dropped_stale={len(dropped_stale)}"
    )
    if args.promote:
        _promote_outputs(args, merged_batch_path, manifest_path, release_tag)
        print(f"Promoted release tag {release_tag}")
    return 0


def cleanup(args: argparse.Namespace) -> int:
    repository = resolve_release_repository(args.repository)
    token = str(os.getenv("GITHUB_TOKEN") or "").strip()
    if not repository or not token:
        raise RuntimeError("Cleanup requires repository and GITHUB_TOKEN")
    client = GitHubReleaseClient(repository, token)
    releases = [
        release
        for release in client.list_releases()
        if str(release.get("tag_name") or "").startswith(DEFAULT_BATCH_RELEASE_PREFIX + "-")
    ]
    releases.sort(key=lambda item: str(item.get("created_at") or item.get("published_at") or ""), reverse=True)
    to_delete = releases[int(args.keep_promoted_batches) :]
    for release in to_delete:
        tag = str(release.get("tag_name") or "").strip()
        client.delete_release(int(release["id"]))
        client.delete_tag_ref(tag)
        print(f"Deleted old promoted batch release {tag}")
    print(f"Kept {min(len(releases), int(args.keep_promoted_batches))} promoted batches")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch LightGBM training pipeline helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--output", required=True)
    discover_parser.add_argument("--macro-output", required=True)
    discover_parser.add_argument("--price-floor", type=float, default=0.10)
    discover_parser.add_argument("--fast-screen-min-score", type=int, default=DEFAULT_FAST_SCREEN_MIN_SCORE)
    discover_parser.add_argument("--fast-screen-margin", type=int, default=DEFAULT_FAST_SCREEN_MARGIN)
    discover_parser.add_argument("--fast-screen-period", default="1y")
    discover_parser.add_argument("--fast-screen-interval", default="1d")
    discover_parser.add_argument("--matrix-jobs", type=int, default=4)
    discover_parser.add_argument("--macro-lookback-days", type=int, default=365 * 5)
    discover_parser.add_argument("--max-tickers", type=int, default=0)
    discover_parser.set_defaults(func=discover)

    train_parser = subparsers.add_parser("train-shard")
    train_parser.add_argument("--discovery-file", required=True)
    train_parser.add_argument("--macro-file", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--batch-id", required=True)
    train_parser.add_argument("--shard-index", type=int, required=True)
    train_parser.add_argument("--shard-count", type=int, required=True)
    train_parser.add_argument("--training-period", default="5y")
    train_parser.add_argument("--training-interval", default="1d")
    train_parser.add_argument("--lookback-days", type=int, default=1260)
    train_parser.add_argument("--min-rows-per-horizon", type=int, default=50)
    train_parser.set_defaults(func=train_shard)

    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--partial-dir", required=True)
    reduce_parser.add_argument("--macro-file")
    reduce_parser.add_argument("--output-batch", required=True)
    reduce_parser.add_argument("--output-manifest", required=True)
    reduce_parser.add_argument("--batch-id", required=True)
    reduce_parser.add_argument("--repository", default="")
    reduce_parser.add_argument("--min-rows-per-horizon", type=int, default=50)
    reduce_parser.add_argument("--max-stale-cycles", type=int, default=4)
    reduce_parser.add_argument("--sanity-tickers", nargs="+", default=list(DEFAULT_SANITY_TICKERS))
    reduce_parser.add_argument("--promote", action="store_true")
    reduce_parser.set_defaults(func=reduce_and_validate)

    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--repository", default="")
    cleanup_parser.add_argument("--keep-promoted-batches", type=int, default=DEFAULT_BATCH_RETENTION)
    cleanup_parser.set_defaults(func=cleanup)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
