from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.email_reports import csv_attachment, render_html_table, render_metric_tiles, render_report_html, send_brevo_email
from modules.fred_client import get_macro_feature_table
from modules.lightgbm_batch import (
    batch_asset_urls_from_manifest,
    fetch_live_manifest,
    load_return_model_batch_from_urls,
    resolve_release_repository,
    shard_asset_urls_by_index,
)
from modules.prediction_tracker import SUBSCORE_ROW_FIELDS, record_predictions_from_scan
from modules.profit_opportunities import (
    DEFAULT_FAST_SCREEN_MARGIN,
    HORIZON_SETTINGS as _ALL_HORIZON_SETTINGS,
    filter_by_upside,
    scan_profit_opportunities,
)
from modules.screener import run_screener

# Scheduled scans only cover short/medium term horizons; keep the same
# restricted subset here (rather than the full three-horizon superset in
# modules.profit_opportunities, which also serves the long-term Streamlit page).
HORIZON_SETTINGS = {k: v for k, v in _ALL_HORIZON_SETTINGS.items() if k in ("short_term", "medium_term")}
DEFAULT_MIN_SCORE = 50
DEFAULT_MIN_UPSIDE_PCT = 15.0
DEFAULT_FAST_SCREEN_PROXY_THRESHOLD = 15
TOP_RESULTS_LIMIT = 75
PREVIEW_LIMIT = 10


@contextmanager
def live_scoring_context(manifest: dict[str, Any], batch: dict[str, Any], shared_macro_table: pd.DataFrame):
    import modules.scoring_engine as scoring_engine

    original_build_feature_table = scoring_engine.build_feature_table

    def _build_feature_table_with_shared_macro(*args, **kwargs):
        kwargs.setdefault("shared_macro_table", shared_macro_table)
        return original_build_feature_table(*args, **kwargs)

    with (
        patch.object(scoring_engine, "_load_live_lightgbm_manifest", return_value=manifest),
        patch.object(scoring_engine, "_load_live_lightgbm_batch", return_value=batch),
        # The scan scripts preload the full combined batch above (`batch`)
        # once for every ticker in the manifest, so a per-ticker shard
        # download here would be pure redundant network I/O. Force shard
        # lookups to report "not found" so `_load_live_lightgbm_models`
        # always falls through to the already-loaded combined batch instead.
        patch.object(scoring_engine, "_load_live_lightgbm_shard_batch", return_value={}),
        patch.object(scoring_engine, "build_feature_table", side_effect=_build_feature_table_with_shared_macro),
    ):
        yield


def manifest_tickers(manifest: dict[str, Any] | None) -> list[str]:
    return sorted(
        str(ticker).strip().upper()
        for ticker in ((manifest or {}).get("tickers") or {})
        if str(ticker).strip()
    )


def manifest_shard_count(manifest: dict[str, Any] | None) -> int:
    return max(1, int(((manifest or {}).get("latest_batch") or {}).get("shard_count") or 1))


def tickers_for_shard(manifest: dict[str, Any] | None, shard_index: int, shard_count: int) -> list[str]:
    """Return the sorted tickers assigned to `shard_index` of `shard_count`.

    When `shard_count <= 1` (old manifests without shard partitioning, or a
    single-shard run) every manifest ticker belongs to the one shard. Any
    ticker missing a `shard_index` (shouldn't happen once shards exist, see
    `build_live_manifest`) defaults to shard 0 rather than being dropped.
    """
    tickers_meta = (manifest or {}).get("tickers") or {}
    if shard_count <= 1:
        return manifest_tickers(manifest)
    selected = []
    for ticker, meta in tickers_meta.items():
        clean_ticker = str(ticker).strip().upper()
        if not clean_ticker:
            continue
        ticker_shard = int((meta or {}).get("shard_index") or 0)
        if ticker_shard == int(shard_index):
            selected.append(clean_ticker)
    return sorted(selected)


def _shard_batch_asset_urls(manifest: dict[str, Any], shard_index: int, shard_count: int) -> list[str]:
    if shard_count > 1:
        urls = shard_asset_urls_by_index(manifest, shard_index)
        if urls:
            return urls
    return batch_asset_urls_from_manifest(manifest)


def discover_scan_inputs(repository: str | None, output_manifest: Path, output_macro: Path) -> dict[str, Any]:
    manifest = fetch_live_manifest(repository=resolve_release_repository(repository))
    tickers = manifest_tickers(manifest)
    if not tickers:
        raise RuntimeError("Live manifest is empty or unreachable; refusing to run an empty scan")

    shared_macro_table = fetch_shared_macro_table()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    output_macro.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(shared_macro_table, output_macro)

    return {"tickers": tickers, "shard_count": manifest_shard_count(manifest)}


def fetch_shared_macro_table(lookback_days: int = 365 * 5) -> pd.DataFrame:
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=int(lookback_days))
    macro_table = get_macro_feature_table(start_date, end_date, realtime_end=end_date)
    # stderr, not stdout: the `discover` subcommand's stdout is redirected
    # straight into `$GITHUB_OUTPUT` by the workflow, which requires every
    # stdout line to be a strict `key=value` pair.
    print(f"Loaded shared FRED macro table rows: {len(macro_table)}", file=sys.stderr)
    return macro_table


def scan_and_filter_profit_opportunities(
    tickers: list[str],
    horizon: str,
    *,
    min_upside_pct: float = DEFAULT_MIN_UPSIDE_PCT,
    max_results: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Scan `tickers` and filter by upside without stripping internal scoring
    fields or recording predictions.

    Rows keep their internal `_rsi`/`SUBSCORE_ROW_FIELDS` columns intact, and
    `max_results` defaults to unbounded, so a sharded caller can defer both
    final top-N truncation and prediction-recording until after all shard
    partials are merged (see `finalize_profit_results`).
    """
    scanned = scan_profit_opportunities(
        tickers,
        horizon,
        use_fast_screen=True,
        fast_screen_base=DEFAULT_FAST_SCREEN_PROXY_THRESHOLD + DEFAULT_FAST_SCREEN_MARGIN,
        fast_screen_margin=DEFAULT_FAST_SCREEN_MARGIN,
        parallel=False,
        prefetch_price_period="1y",
    )
    filtered = filter_by_upside(scanned, horizon, min_upside_pct=min_upside_pct, max_results=max_results)
    stats_base = {
        "scanned_count": int(scanned.attrs.get("scanned_count", len(tickers))),
        "fast_filtered_count": int(scanned.attrs.get("fast_filtered_count", 0)),
        "passed_fast_screen_count": int(scanned.attrs.get("fully_analyzed_count", 0)),
        "failed_count": int(scanned.attrs.get("failed_count", 0)),
    }
    return filtered, stats_base


def finalize_profit_results(
    results: pd.DataFrame,
    horizon: str,
    *,
    record: bool = True,
    stats_base: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Strip internal scoring fields from a filtered/sorted/truncated profit
    results frame, optionally recording predictions from the pre-strip rows.

    Recording must happen exactly once, on the final globally-truncated set,
    so callers merging multiple scan shards should only pass `record=True`
    here at the reduce stage (not per-shard).
    """
    recorded_rows = results.to_dict("records") if not results.empty else []
    display_results = results.drop(columns=["_rsi", *SUBSCORE_ROW_FIELDS], errors="ignore")

    if record and recorded_rows:
        recorded_count = record_predictions_from_scan(recorded_rows, horizon=horizon, source="profit_opportunities")
    else:
        recorded_count = 0

    stats = dict(stats_base or {})
    stats["qualified_count"] = len(display_results)
    stats["recorded_count"] = recorded_count
    return display_results, stats


def run_profit_opportunities_scan(
    tickers: list[str],
    horizon: str,
    *,
    min_upside_pct: float = DEFAULT_MIN_UPSIDE_PCT,
    max_results: int | None = TOP_RESULTS_LIMIT,
    record: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Single-shot scan + filter + finalize, used by non-sharded callers."""
    filtered, stats_base = scan_and_filter_profit_opportunities(
        tickers, horizon, min_upside_pct=min_upside_pct, max_results=max_results
    )
    return finalize_profit_results(filtered, horizon, record=record, stats_base=stats_base)


def _format_value(value: Any, *, currency: bool = False, pct: bool = False) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if currency:
        return f"${number:,.2f}"
    if pct:
        return f"{number:,.2f}%"
    return f"{number:,.2f}"


def _preview_rows(df: pd.DataFrame, columns: list[str], *, limit: int = PREVIEW_LIMIT) -> list[dict[str, object]]:
    preview = df.head(limit).copy() if not df.empty else pd.DataFrame(columns=columns)
    records = preview.to_dict("records") if not preview.empty else []
    formatted = []
    for row in records:
        formatted.append(
            {
                **row,
                "Score": _format_value(row.get("Score")),
                "Current Price": _format_value(row.get("Current Price"), currency=True),
                "Target Price": _format_value(row.get("Target Price"), currency=True),
                "Projected Upside %": _format_value(row.get("Projected Upside %"), pct=True),
            }
        )
    return formatted


def build_scan_report(
    manifest: dict[str, Any],
    tickers: list[str],
    screener_results: pd.DataFrame,
    profit_results: pd.DataFrame,
    profit_stats: dict[str, int],
    horizon: str,
    run_date: str,
) -> str:
    current_run = (manifest or {}).get("current_run") or {}
    screener_top = screener_results.iloc[0] if not screener_results.empty else None
    profit_top = profit_results.iloc[0] if not profit_results.empty else None
    screener_avg_score = float(screener_results["Score"].mean()) if not screener_results.empty else None
    screener_median_score = float(screener_results["Score"].median()) if not screener_results.empty else None
    profit_avg_upside = float(profit_results["Projected Upside %"].mean()) if not profit_results.empty else None
    profit_median_upside = float(profit_results["Projected Upside %"].median()) if not profit_results.empty else None

    sections = [
        (
            f"<h2 style=\"margin:0 0 10px 0;color:#f4fff8;\">Screener top 75</h2>"
            f"<p style=\"margin:0 0 16px 0;color:#a9bdd7;\">Score-based scan for {HORIZON_SETTINGS[horizon]['label']} workflows. Top 10 preview below; full top 75 is attached as CSV.</p>"
            f"{render_metric_tiles([{'label': 'Passed fast-screen', 'value': str(int(screener_results.attrs.get('fully_analyzed_count') or 0))}, {'label': 'Fast-filtered', 'value': str(int(screener_results.attrs.get('fast_filtered_count') or 0))}, {'label': 'Top ticker', 'value': str(screener_top['Ticker']) if screener_top is not None else '—'}, {'label': 'Avg / median score', 'value': ('—' if screener_avg_score is None else f'{screener_avg_score:.1f} / {screener_median_score:.1f}')}, {'label': 'High-confidence picks', 'value': str(int((screener_results['Score'] >= 80).sum())) if not screener_results.empty else '0'}])}"
            f"<div style=\"margin-top:18px;\">{render_html_table(['Ticker', 'Company', 'Score', 'Recommendation', 'Current Price', 'Target Price'], _preview_rows(screener_results, ['Ticker', 'Company', 'Score', 'Recommendation', 'Current Price', 'Target Price']))}</div>"
        ),
        (
            f"<h2 style=\"margin:0 0 10px 0;color:#f4fff8;\">Profit opportunities top 75</h2>"
            f"<p style=\"margin:0 0 16px 0;color:#a9bdd7;\">Projected-upside scan for {HORIZON_SETTINGS[horizon]['label']}. Top 10 preview below; full top 75 is attached as CSV.</p>"
            f"{render_metric_tiles([{'label': 'Passed fast-screen', 'value': str(int(profit_stats.get('passed_fast_screen_count') or 0))}, {'label': 'Recorded predictions', 'value': str(int(profit_stats.get('recorded_count') or 0))}, {'label': 'Top ticker', 'value': str(profit_top['Ticker']) if profit_top is not None else '—'}, {'label': 'Avg / median upside', 'value': ('—' if profit_avg_upside is None else f'{profit_avg_upside:.2f}% / {profit_median_upside:.2f}%')}, {'label': 'High-confidence picks', 'value': str(int((profit_results['Confidence'] == 'High').sum())) if not profit_results.empty else '0'}])}"
            f"<div style=\"margin-top:18px;\">{render_html_table(['Ticker', 'Company', 'Score', 'Current Price', 'Target Price', 'Projected Upside %', 'Confidence'], _preview_rows(profit_results, ['Ticker', 'Company', 'Score', 'Current Price', 'Target Price', 'Projected Upside %', 'Confidence']))}</div>"
        ),
    ]

    return render_report_html(
        title=f"{HORIZON_SETTINGS[horizon]['label']} scan report",
        subtitle=f"Run date {run_date} · ticker pool {len(tickers)} · manifest-trained universe from the promoted LightGBM release",
        metrics=[
            {"label": "Tickers scanned", "value": str(len(tickers))},
            {"label": "Newly trained", "value": str(len(current_run.get('trained_tickers') or []))},
            {"label": "Carried forward", "value": str(len(current_run.get('carried_forward_tickers') or []))},
            {"label": "Dropped stale", "value": str(len(current_run.get('dropped_stale_tickers') or []))},
        ],
        sections=sections,
    )


def run_scan_shard(
    horizon: str,
    manifest: dict[str, Any],
    macro_table: pd.DataFrame,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    """Run the screener + profit-opportunities scans for one ticker shard.

    Returns unrecorded, untruncated partial results (internal scoring fields
    intact) plus raw per-shard stats, ready to be merged with other shards'
    partials and finalized exactly once at the reduce stage.
    """
    empty_screener_attrs = {"fast_filtered_count": 0, "fully_analyzed_count": 0, "failed_count": 0, "source_ticker_count": 0}
    empty_profit_stats = {"scanned_count": 0, "fast_filtered_count": 0, "passed_fast_screen_count": 0, "failed_count": 0}

    shard_tickers = tickers_for_shard(manifest, shard_index, shard_count)
    if not shard_tickers:
        print(f"Shard {shard_index}/{shard_count} has no assigned tickers; writing empty partial output")
        return {
            "shard_index": shard_index,
            "screener": pd.DataFrame(),
            "screener_attrs": empty_screener_attrs,
            "profit": pd.DataFrame(),
            "profit_stats_base": empty_profit_stats,
        }

    asset_urls = _shard_batch_asset_urls(manifest, shard_index, shard_count)
    if not asset_urls:
        raise RuntimeError(f"No batch asset URLs resolved for shard {shard_index}/{shard_count}")
    batch = load_return_model_batch_from_urls(asset_urls)
    if not ((batch or {}).get("models") or {}):
        raise RuntimeError(f"Shard {shard_index}/{shard_count} batch artifact is empty or unreachable")

    with live_scoring_context(manifest, batch, macro_table):
        screener_partial = run_screener(
            universe="custom",
            custom_tickers=shard_tickers,
            min_score=DEFAULT_MIN_SCORE,
            max_results=max(len(shard_tickers), 1),
            max_workers=1,
            use_fast_screen=True,
        )
        profit_partial, profit_stats_base = scan_and_filter_profit_opportunities(shard_tickers, horizon, max_results=None)

    print(
        f"Shard {shard_index}/{shard_count}: tickers={len(shard_tickers)} "
        f"screener_qualified={len(screener_partial)} profit_qualified={len(profit_partial)}"
    )
    return {
        "shard_index": shard_index,
        "screener": screener_partial,
        "screener_attrs": {
            "fast_filtered_count": int(screener_partial.attrs.get("fast_filtered_count") or 0),
            "fully_analyzed_count": int(screener_partial.attrs.get("fully_analyzed_count") or 0),
            "failed_count": int(screener_partial.attrs.get("failed_count") or 0),
            "source_ticker_count": int(screener_partial.attrs.get("source_ticker_count") or len(shard_tickers)),
        },
        "profit": profit_partial,
        "profit_stats_base": profit_stats_base,
    }


def merge_shard_outputs(partial_dir: Path) -> dict[str, Any]:
    partial_files = sorted(Path(partial_dir).glob("*.joblib"))
    if not partial_files:
        raise RuntimeError(f"No scan-shard partial files found under {partial_dir}")

    screener_frames: list[pd.DataFrame] = []
    profit_frames: list[pd.DataFrame] = []
    screener_attrs_totals = {"fast_filtered_count": 0, "fully_analyzed_count": 0, "failed_count": 0, "source_ticker_count": 0}
    profit_stats_totals = {"scanned_count": 0, "fast_filtered_count": 0, "passed_fast_screen_count": 0, "failed_count": 0}

    for path in partial_files:
        partial = joblib.load(path)
        screener_df = partial.get("screener")
        if isinstance(screener_df, pd.DataFrame) and not screener_df.empty:
            screener_frames.append(screener_df)
        for key in screener_attrs_totals:
            screener_attrs_totals[key] += int((partial.get("screener_attrs") or {}).get(key, 0))

        profit_df = partial.get("profit")
        if isinstance(profit_df, pd.DataFrame) and not profit_df.empty:
            profit_frames.append(profit_df)
        for key in profit_stats_totals:
            profit_stats_totals[key] += int((partial.get("profit_stats_base") or {}).get(key, 0))

    return {
        "screener_frames": screener_frames,
        "screener_attrs_totals": screener_attrs_totals,
        "profit_frames": profit_frames,
        "profit_stats_totals": profit_stats_totals,
    }


def reduce_scan_shards(horizon: str, manifest: dict[str, Any], partial_dir: Path, run_date: str) -> int:
    tickers = manifest_tickers(manifest)
    merged = merge_shard_outputs(partial_dir)

    if merged["screener_frames"]:
        screener_results = pd.concat(merged["screener_frames"], ignore_index=True)
        screener_results = screener_results.sort_values("Score", ascending=False).head(TOP_RESULTS_LIMIT).reset_index(drop=True)
    else:
        screener_results = pd.DataFrame()
    screener_results.attrs.update(merged["screener_attrs_totals"])

    if merged["profit_frames"]:
        profit_raw = pd.concat(merged["profit_frames"], ignore_index=True)
        profit_raw = profit_raw.sort_values("Projected Upside %", ascending=False).head(TOP_RESULTS_LIMIT).reset_index(drop=True)
    else:
        profit_raw = pd.DataFrame()
    profit_results, profit_stats = finalize_profit_results(
        profit_raw, horizon, record=True, stats_base=merged["profit_stats_totals"]
    )

    html_content = build_scan_report(manifest, tickers, screener_results, profit_results, profit_stats, horizon, run_date)
    attachments = [
        csv_attachment(f"screener_top75_{run_date}.csv", screener_results.to_csv(index=False)),
        csv_attachment(f"profit_opportunities_top75_{run_date}.csv", profit_results.to_csv(index=False)),
    ]
    subject = f"BK Self {HORIZON_SETTINGS[horizon]['label']} scan report — {run_date}"
    sent_count = send_brevo_email(subject=subject, html_content=html_content, attachments=attachments)
    print(f"Sent scan report to {sent_count} recipient(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and send scheduled scan reports for the live LightGBM ticker "
            "universe, split across discover/scan-shard/reduce stages so large "
            "universes can be scanned in parallel without loading the full batch "
            "artifact in a single job."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser(
        "discover", help="Fetch the live manifest and shared macro table once for all scan shards."
    )
    discover_parser.add_argument("--repository", default=None)
    discover_parser.add_argument("--output-manifest", required=True)
    discover_parser.add_argument("--output-macro", required=True)

    scan_shard_parser = subparsers.add_parser(
        "scan-shard", help="Run the screener/profit-opportunities scans for one ticker shard."
    )
    scan_shard_parser.add_argument("--horizon", choices=sorted(HORIZON_SETTINGS.keys()), required=True)
    scan_shard_parser.add_argument("--manifest-file", required=True)
    scan_shard_parser.add_argument("--macro-file", required=True)
    scan_shard_parser.add_argument("--shard-index", type=int, required=True)
    scan_shard_parser.add_argument("--shard-count", type=int, required=True)
    scan_shard_parser.add_argument("--output", required=True)

    reduce_parser = subparsers.add_parser(
        "reduce",
        help="Merge scan-shard partial outputs, truncate to the final top-N, "
        "record predictions once, and send the email report.",
    )
    reduce_parser.add_argument("--horizon", choices=sorted(HORIZON_SETTINGS.keys()), required=True)
    reduce_parser.add_argument("--manifest-file", required=True)
    reduce_parser.add_argument("--partial-dir", required=True)
    reduce_parser.add_argument("--run-date", default=None)

    return parser


def _run_discover(args: argparse.Namespace) -> int:
    run_date = datetime.now(UTC).date().isoformat()
    info = discover_scan_inputs(args.repository, Path(args.output_manifest), Path(args.output_macro))
    # This subcommand's stdout is redirected straight into `$GITHUB_OUTPUT`
    # by the workflow (`>> "$GITHUB_OUTPUT"`), which requires every stdout
    # line to be a strict `key=value` pair, so human-readable logging goes
    # to stderr instead.
    print(f"Discovered {len(info['tickers'])} tickers across {info['shard_count']} scan shard(s)", file=sys.stderr)
    print(f"run_date={run_date}")
    print(f"shard_count={info['shard_count']}")
    print(f"shard_matrix={json.dumps(list(range(info['shard_count'])))}")
    return 0


def _run_scan_shard(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest_file).read_text(encoding="utf-8"))
    macro_table = joblib.load(Path(args.macro_file))
    partial = run_scan_shard(args.horizon, manifest, macro_table, int(args.shard_index), int(args.shard_count))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(partial, output_path)
    return 0


def _run_reduce(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest_file).read_text(encoding="utf-8"))
    run_date = args.run_date or datetime.now(UTC).date().isoformat()
    return reduce_scan_shards(args.horizon, manifest, Path(args.partial_dir), run_date)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "discover":
        return _run_discover(args)
    if args.command == "scan-shard":
        return _run_scan_shard(args)
    if args.command == "reduce":
        return _run_reduce(args)
    raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
