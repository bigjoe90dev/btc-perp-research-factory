from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import yaml

from .common import dump_json, ensure_dir, list_files, sha256_file, to_iso_z
from .validate_dataset import validate_file

API_BASE = "https://www.bitmex.com/api/v1"
DEFAULT_CACHE_ROOT = Path("research/data_cache/btc_bitmex_perp")
DEFAULT_DATASET_ID = "BTC_BITMEX_PERP_1M"


def _parse_day_start_utc(day_ymd: str) -> pd.Timestamp:
    ts = pd.to_datetime(f"{day_ymd}T00:00:00Z", utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid day format (expected YYYY-MM-DD): {day_ymd}")
    return ts


def _end_day_exclusive_utc(day_ymd: str) -> pd.Timestamp:
    return _parse_day_start_utc(day_ymd) + pd.Timedelta(days=1)


def _day_bounds(start_day: str, end_day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = _parse_day_start_utc(start_day)
    end_ts = _end_day_exclusive_utc(end_day)
    if end_ts <= start_ts:
        raise ValueError(f"end date must be >= start date (start={start_day}, end={end_day})")
    return start_ts, end_ts


def _iter_chunks(
    start_ts: pd.Timestamp,
    end_ts_exclusive: pd.Timestamp,
    chunk: pd.Timedelta,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    cur = start_ts
    while cur < end_ts_exclusive:
        nxt = min(cur + chunk, end_ts_exclusive)
        yield cur, nxt
        cur = nxt


def _iso(ts: pd.Timestamp) -> str:
    return pd.to_datetime(ts, utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _chunk_id(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    s = int(start_ts.value // 1_000_000)
    e = int(end_ts.value // 1_000_000)
    return f"{s}-{e}"


def _chunk_file(dir_path: Path, prefix: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Path:
    s = int(start_ts.value // 1_000_000)
    e = int(end_ts.value // 1_000_000)
    return dir_path / f"{prefix}_{s}_{e}.json"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_ts_ns(value: Any) -> int | None:
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return int(ts.value)


def _load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "candles": {}, "funding": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("version", 1)
            data.setdefault("candles", {})
            data.setdefault("funding", {})
            return data
    except Exception:
        pass
    return {"version": 1, "candles": {}, "funding": {}}


def _save_progress(path: Path, progress: dict[str, Any]) -> None:
    dump_json(path, progress)


def _stream_progress(progress: dict[str, Any], stream: str, key: str) -> dict[str, Any]:
    container = progress.setdefault(stream, {})
    entry = container.setdefault(key, {})
    entry.setdefault("completed_chunks", {})
    return entry


def _read_chunk_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("rows", [])
        if isinstance(rows, list):
            return [x for x in rows if isinstance(x, dict)]
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _sorted_chunk_paths(raw_dir: Path, prefix: str) -> list[Path]:
    out: list[tuple[int, int, Path]] = []
    for path in raw_dir.glob(f"{prefix}_*_*.json"):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            start_ms = int(parts[-2])
            end_ms = int(parts[-1])
        except (TypeError, ValueError):
            continue
        out.append((start_ms, end_ms, path))
    out.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in out]


def _chunk_paths_from_progress(completed_chunks: dict[str, Any]) -> list[Path]:
    out: list[tuple[int, int, Path]] = []
    for cid, meta in completed_chunks.items():
        if not isinstance(cid, str) or not isinstance(meta, dict):
            continue
        try:
            start_ms, end_ms = map(int, cid.split("-", 1))
        except (TypeError, ValueError):
            continue
        path_str = meta.get("path")
        if not path_str:
            continue
        path = Path(str(path_str))
        if not path.exists():
            continue
        out.append((start_ms, end_ms, path))
    out.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in out]


def _candle_df_from_chunk(path: Path) -> pd.DataFrame:
    rows = _read_chunk_rows(path)
    if not rows:
        return pd.DataFrame(
            columns=["ts_utc_ns", "open", "high", "low", "close", "volume", "source", "coin", "market"]
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame(
            columns=["ts_utc_ns", "open", "high", "low", "close", "volume", "source", "coin", "market"]
        )
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    mask = ts.notna()
    if not mask.any():
        return pd.DataFrame(
            columns=["ts_utc_ns", "open", "high", "low", "close", "volume", "source", "coin", "market"]
        )
    filtered = frame.loc[mask]
    out = pd.DataFrame(
        {
            "ts_utc_ns": ts.loc[mask].astype("int64"),
            "open": pd.to_numeric(filtered.get("open"), errors="coerce"),
            "high": pd.to_numeric(filtered.get("high"), errors="coerce"),
            "low": pd.to_numeric(filtered.get("low"), errors="coerce"),
            "close": pd.to_numeric(filtered.get("close"), errors="coerce"),
            "volume": pd.to_numeric(filtered.get("volume"), errors="coerce"),
            "source": "bitmex:api.trade_bucketed",
            "coin": "BTC",
            "market": "perp",
        }
    )
    out = out.dropna(subset=["ts_utc_ns", "open", "high", "low", "close", "volume"])
    return out


def _funding_df_from_chunk(path: Path) -> pd.DataFrame:
    rows = _read_chunk_rows(path)
    if not rows:
        return pd.DataFrame(columns=["ts_utc_ns", "funding_rate_raw", "source", "coin", "market"])
    frame = pd.DataFrame.from_records(rows)
    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame(columns=["ts_utc_ns", "funding_rate_raw", "source", "coin", "market"])
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    mask = ts.notna()
    if not mask.any():
        return pd.DataFrame(columns=["ts_utc_ns", "funding_rate_raw", "source", "coin", "market"])
    filtered = frame.loc[mask]
    out = pd.DataFrame(
        {
            "ts_utc_ns": ts.loc[mask].astype("int64"),
            "funding_rate_raw": pd.to_numeric(filtered.get("fundingRate"), errors="coerce"),
            "source": "bitmex:api.funding",
            "coin": "BTC",
            "market": "perp",
        }
    )
    out = out.dropna(subset=["ts_utc_ns", "funding_rate_raw"])
    return out


def _merge_by_ts_ns(path: Path, new_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        old_df = pd.read_parquet(path)
        for col in columns:
            if col not in old_df.columns:
                old_df[col] = pd.NA
        old_df = old_df[columns]
        merged = pd.concat([old_df, new_df[columns]], ignore_index=True)
    else:
        merged = new_df[columns].copy()

    merged = merged.dropna(subset=["ts_utc_ns"])
    merged["ts_utc_ns"] = pd.to_numeric(merged["ts_utc_ns"], errors="coerce")
    merged = merged.dropna(subset=["ts_utc_ns"])
    merged["ts_utc_ns"] = merged["ts_utc_ns"].astype("int64")
    merged = merged.drop_duplicates(subset=["ts_utc_ns"], keep="last")
    merged = merged.sort_values("ts_utc_ns")
    merged = merged.reset_index(drop=True)
    return merged[columns]


def _write_chunk(
    path: Path,
    stream: str,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "stream": stream,
        "symbol": symbol,
        "start_utc": _iso(start_ts),
        "end_utc_exclusive": _iso(end_ts),
        "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row_count": len(rows),
        "rows": rows,
    }
    dump_json(path, payload)


class BitmexClient:
    def __init__(
        self,
        timeout_sec: int = 30,
        max_retries: int = 8,
        base_backoff_sec: float = 0.5,
        max_backoff_sec: float = 30.0,
        rate_limit_sleep_sec: float = 0.1,
    ) -> None:
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.rate_limit_sleep_sec = rate_limit_sleep_sec
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                wait = max(float(retry_after), self.rate_limit_sleep_sec)
            except ValueError:
                wait = self.base_backoff_sec
        else:
            wait = min(self.max_backoff_sec, self.base_backoff_sec * (2**attempt))
        wait += random.uniform(0.0, 0.25)
        time.sleep(wait)

    def get_json(self, path: str, params: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_sec)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise RuntimeError(f"BitMEX returned invalid JSON at {url}") from exc
                if self.rate_limit_sleep_sec > 0:
                    time.sleep(self.rate_limit_sleep_sec)
                return data

            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                self._sleep_backoff(attempt, retry_after=retry_after)
                continue

            body = response.text.strip().replace("\n", " ")
            raise RuntimeError(
                f"BitMEX request failed status={response.status_code} path={path} body={body[:320]}"
            )

        raise RuntimeError(f"BitMEX request failed after retries: {last_error}")


def discover_listing_day(symbol: str, client: BitmexClient) -> str:
    data = client.get_json("instrument", {"symbol": symbol.upper(), "count": 1, "reverse": "false"})
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Unable to discover listing day for {symbol}")
    row = data[0]
    if not isinstance(row, dict) or "listing" not in row:
        raise RuntimeError(f"BitMEX instrument response missing listing field for {symbol}")
    listing = pd.to_datetime(row["listing"], utc=True, errors="coerce")
    if pd.isna(listing):
        raise RuntimeError(f"Invalid listing timestamp for {symbol}: {row.get('listing')}")
    return listing.strftime("%Y-%m-%d")


def _fetch_candle_chunk(
    client: BitmexClient,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    max_rows: int,
) -> list[dict[str, Any]]:
    params = {
        "binSize": "1m",
        "partial": "false",
        "symbol": symbol.upper(),
        "count": int(max_rows),
        "reverse": "false",
        "startTime": _iso(start_ts),
        "endTime": _iso(end_ts - pd.Timedelta(milliseconds=1)),
    }
    data = client.get_json("trade/bucketed", params=params)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected candle response type: {type(data).__name__}")
    return [x for x in data if isinstance(x, dict)]


def _fetch_funding_chunk(
    client: BitmexClient,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    max_rows: int,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    cursor = start_ts
    pages = 0
    while cursor < end_ts:
        params = {
            "symbol": symbol.upper(),
            "count": int(max_rows),
            "reverse": "false",
            "startTime": _iso(cursor),
            "endTime": _iso(end_ts - pd.Timedelta(milliseconds=1)),
        }
        data = client.get_json("funding", params=params)
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected funding response type: {type(data).__name__}")
        rows = [x for x in data if isinstance(x, dict)]
        if not rows:
            break
        all_rows.extend(rows)

        last_ts_ns = _to_ts_ns(rows[-1].get("timestamp"))
        if last_ts_ns is None:
            break
        next_cursor = pd.to_datetime(last_ts_ns + 1, utc=True, unit="ns")
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        pages += 1
        if len(rows) < max_rows:
            break
        if pages > 10_000:
            raise RuntimeError(f"Funding pagination guard hit for {symbol} {start_ts}..{end_ts}")

    dedup: dict[int, dict[str, Any]] = {}
    for row in all_rows:
        ts_ns = _to_ts_ns(row.get("timestamp"))
        if ts_ns is None:
            continue
        dedup[ts_ns] = row
    return [dedup[k] for k in sorted(dedup)] if dedup else []


def download_candles(
    symbol: str,
    start_day: str | None,
    end_day: str,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    chunk_minutes: int = 900,
    max_rows: int = 1000,
    backfill_earliest: bool = False,
    timeout_sec: int = 30,
    max_retries: int = 8,
    rate_limit_sleep_sec: float = 0.1,
    summary_out: str | Path | None = None,
) -> dict[str, Any]:
    if chunk_minutes <= 0 or chunk_minutes > max_rows:
        raise ValueError(f"chunk_minutes must be in 1..{max_rows}")
    if not backfill_earliest and not start_day:
        raise ValueError("--start is required unless --backfill-earliest is set")

    raw_dir = cache_root / "raw" / "candles"
    normalized_path = cache_root / "normalized" / "candles_1m.parquet"
    progress_path = cache_root / "progress.json"
    ensure_dir(raw_dir)
    ensure_dir(normalized_path.parent)

    client = BitmexClient(
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        rate_limit_sleep_sec=rate_limit_sleep_sec,
    )
    try:
        if backfill_earliest and not start_day:
            start_day = discover_listing_day(symbol=symbol, client=client)
        assert start_day is not None
        start_ts, end_ts_exclusive = _day_bounds(start_day, end_day)

        progress = _load_progress(progress_path)
        stream = _stream_progress(progress, "candles", f"{symbol.upper()}:1m")
        completed = stream["completed_chunks"]

        chunk = pd.Timedelta(minutes=chunk_minutes)
        total_chunks = 0
        fetched_chunks = 0
        reused_chunks = 0

        for chunk_start, chunk_end in _iter_chunks(start_ts, end_ts_exclusive, chunk):
            total_chunks += 1
            cid = _chunk_id(chunk_start, chunk_end)
            path = _chunk_file(raw_dir, f"{symbol.lower()}_1m", chunk_start, chunk_end)
            if path.exists():
                if cid not in completed:
                    existing_rows = _read_chunk_rows(path)
                    completed[cid] = {
                        "path": str(path),
                        "row_count": len(existing_rows),
                        "first_ts_ns": _to_ts_ns(existing_rows[0].get("timestamp")) if existing_rows else None,
                        "last_ts_ns": _to_ts_ns(existing_rows[-1].get("timestamp")) if existing_rows else None,
                        "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                    _save_progress(progress_path, progress)
                reused_chunks += 1
                continue

            rows = _fetch_candle_chunk(
                client=client,
                symbol=symbol,
                start_ts=chunk_start,
                end_ts=chunk_end,
                max_rows=max_rows,
            )
            _write_chunk(path, "candles", symbol.upper(), chunk_start, chunk_end, rows)
            first_ts = _to_ts_ns(rows[0].get("timestamp")) if rows else None
            last_ts = _to_ts_ns(rows[-1].get("timestamp")) if rows else None
            completed[cid] = {
                "path": str(path),
                "row_count": len(rows),
                "first_ts_ns": first_ts,
                "last_ts_ns": last_ts,
                "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            fetched_chunks += 1
            _save_progress(progress_path, progress)

        stream["updated_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _save_progress(progress_path, progress)
    finally:
        client.close()

    chunk_paths = _chunk_paths_from_progress(completed)
    if not chunk_paths:
        chunk_paths = _sorted_chunk_paths(raw_dir, f"{symbol.lower()}_1m")
    if not chunk_paths:
        raise RuntimeError("No candle chunk files exist after download")

    cols = ["ts_utc_ns", "open", "high", "low", "close", "volume", "source", "coin", "market"]
    frames: list[pd.DataFrame] = []
    total_paths = len(chunk_paths)
    for i, path in enumerate(chunk_paths, start=1):
        chunk_df = _candle_df_from_chunk(path)
        if not chunk_df.empty:
            frames.append(chunk_df)
        if i % 500 == 0:
            print(f"[candles] normalized {i}/{total_paths} chunk files")
    if not frames:
        raise RuntimeError("No candle rows downloaded in requested range")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError("No candle rows downloaded in requested range")
    # BitMEX bucket semantics can produce open outside [low, high]. Canonicalize to strict OHLC envelope.
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    merged = _merge_by_ts_ns(normalized_path, df, columns=cols)
    merged.to_parquet(normalized_path, index=False)

    validation = validate_file(normalized_path, kind="ohlc", expected_seconds=60)
    if not validation["ok"]:
        raise RuntimeError(f"BitMEX candle validation failed: {validation['errors']}")

    summary = {
        "dataset": DEFAULT_DATASET_ID,
        "stream": "candles",
        "symbol": symbol.upper(),
        "requested_start_day": start_day,
        "requested_end_day": end_day,
        "raw_dir": str(raw_dir),
        "normalized_path": str(normalized_path),
        "progress_path": str(progress_path),
        "chunk_minutes": int(chunk_minutes),
        "chunks_total": int(total_chunks),
        "chunks_fetched": int(fetched_chunks),
        "chunks_reused": int(reused_chunks),
        "row_count": int(validation["row_count"]),
        "start_ts_utc": validation["start_ts_utc"],
        "end_ts_utc": validation["end_ts_utc"],
        "gaps_found": validation["gaps_found"],
        "duplicates_found": validation["duplicates_found"],
        "validation": validation,
    }
    if summary_out:
        dump_json(summary_out, summary)
    return summary


def download_funding(
    symbol: str,
    start_day: str | None,
    end_day: str,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    chunk_days: int = 120,
    max_rows: int = 500,
    backfill_earliest: bool = False,
    timeout_sec: int = 30,
    max_retries: int = 8,
    rate_limit_sleep_sec: float = 0.1,
    summary_out: str | Path | None = None,
) -> dict[str, Any]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be > 0")
    if not backfill_earliest and not start_day:
        raise ValueError("--start is required unless --backfill-earliest is set")

    raw_dir = cache_root / "raw" / "funding"
    normalized_path = cache_root / "normalized" / "funding.parquet"
    progress_path = cache_root / "progress.json"
    ensure_dir(raw_dir)
    ensure_dir(normalized_path.parent)

    client = BitmexClient(
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        rate_limit_sleep_sec=rate_limit_sleep_sec,
    )
    try:
        if backfill_earliest and not start_day:
            start_day = discover_listing_day(symbol=symbol, client=client)
        assert start_day is not None
        start_ts, end_ts_exclusive = _day_bounds(start_day, end_day)

        progress = _load_progress(progress_path)
        stream = _stream_progress(progress, "funding", symbol.upper())
        completed = stream["completed_chunks"]

        chunk = pd.Timedelta(days=chunk_days)
        total_chunks = 0
        fetched_chunks = 0
        reused_chunks = 0

        for chunk_start, chunk_end in _iter_chunks(start_ts, end_ts_exclusive, chunk):
            total_chunks += 1
            cid = _chunk_id(chunk_start, chunk_end)
            path = _chunk_file(raw_dir, symbol.lower(), chunk_start, chunk_end)
            if path.exists():
                if cid not in completed:
                    existing_rows = _read_chunk_rows(path)
                    completed[cid] = {
                        "path": str(path),
                        "row_count": len(existing_rows),
                        "first_ts_ns": _to_ts_ns(existing_rows[0].get("timestamp")) if existing_rows else None,
                        "last_ts_ns": _to_ts_ns(existing_rows[-1].get("timestamp")) if existing_rows else None,
                        "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                    _save_progress(progress_path, progress)
                reused_chunks += 1
                continue

            rows = _fetch_funding_chunk(
                client=client,
                symbol=symbol,
                start_ts=chunk_start,
                end_ts=chunk_end,
                max_rows=max_rows,
            )
            _write_chunk(path, "funding", symbol.upper(), chunk_start, chunk_end, rows)
            first_ts = _to_ts_ns(rows[0].get("timestamp")) if rows else None
            last_ts = _to_ts_ns(rows[-1].get("timestamp")) if rows else None
            completed[cid] = {
                "path": str(path),
                "row_count": len(rows),
                "first_ts_ns": first_ts,
                "last_ts_ns": last_ts,
                "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            fetched_chunks += 1
            _save_progress(progress_path, progress)

        stream["updated_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _save_progress(progress_path, progress)
    finally:
        client.close()

    chunk_paths = _chunk_paths_from_progress(completed)
    if not chunk_paths:
        chunk_paths = _sorted_chunk_paths(raw_dir, symbol.lower())
    if not chunk_paths:
        raise RuntimeError("No funding chunk files exist after download")

    cols = ["ts_utc_ns", "funding_rate_raw", "source", "coin", "market"]
    frames: list[pd.DataFrame] = []
    total_paths = len(chunk_paths)
    for i, path in enumerate(chunk_paths, start=1):
        chunk_df = _funding_df_from_chunk(path)
        if not chunk_df.empty:
            frames.append(chunk_df)
        if i % 100 == 0:
            print(f"[funding] normalized {i}/{total_paths} chunk files")
    if not frames:
        raise RuntimeError("No funding rows downloaded in requested range")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError("No funding rows downloaded in requested range")

    merged = _merge_by_ts_ns(normalized_path, df, columns=cols)
    merged.to_parquet(normalized_path, index=False)

    validation = validate_file(normalized_path, kind="series")
    if not validation["ok"]:
        raise RuntimeError(f"BitMEX funding validation failed: {validation['errors']}")

    summary = {
        "dataset": DEFAULT_DATASET_ID,
        "stream": "funding",
        "symbol": symbol.upper(),
        "requested_start_day": start_day,
        "requested_end_day": end_day,
        "raw_dir": str(raw_dir),
        "normalized_path": str(normalized_path),
        "progress_path": str(progress_path),
        "chunk_days": int(chunk_days),
        "chunks_total": int(total_chunks),
        "chunks_fetched": int(fetched_chunks),
        "chunks_reused": int(reused_chunks),
        "row_count": int(validation["row_count"]),
        "start_ts_utc": validation["start_ts_utc"],
        "end_ts_utc": validation["end_ts_utc"],
        "gaps_found": validation["gaps_found"],
        "duplicates_found": validation["duplicates_found"],
        "validation": validation,
    }
    if summary_out:
        dump_json(summary_out, summary)
    return summary


def register_manifest_entry(
    manifest_path: Path,
    dataset_id: str,
    cache_root: Path,
    candle_summary: dict[str, Any],
    funding_summary: dict[str, Any],
) -> dict[str, Any]:
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            manifest = {}
    else:
        manifest = {}

    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        datasets = []
    manifest["datasets"] = datasets

    canonical_file = str(cache_root / "normalized" / "candles_1m.parquet")
    funding_file = str(cache_root / "normalized" / "funding.parquet")
    files = list_files(cache_root)

    entry = {
        "dataset_id": dataset_id,
        "source": "bitmex",
        "symbol": "BTC",
        "market": "perp",
        "timeframe": "1m",
        "start_ts_utc": candle_summary.get("start_ts_utc"),
        "end_ts_utc": candle_summary.get("end_ts_utc"),
        "row_count": candle_summary.get("row_count"),
        "gaps_found": candle_summary.get("gaps_found"),
        "duplicates_found": candle_summary.get("duplicates_found"),
        "longest_gap_seconds": candle_summary.get("validation", {}).get("longest_gap_seconds"),
        "checksum_sha256": sha256_file(canonical_file),
        "canonical_file": canonical_file,
        "funding_file": funding_file,
        "funding_start_ts_utc": funding_summary.get("start_ts_utc"),
        "funding_end_ts_utc": funding_summary.get("end_ts_utc"),
        "funding_row_count": funding_summary.get("row_count"),
        "funding_duplicates_found": funding_summary.get("duplicates_found"),
        "files_on_disk": files,
        "required_start_ts_utc": None,
        "required_end_ts_utc": None,
    }

    replaced = False
    for i, row in enumerate(datasets):
        if isinstance(row, dict) and str(row.get("dataset_id")) == dataset_id:
            datasets[i] = entry
            replaced = True
            break
    if not replaced:
        datasets.append(entry)

    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return entry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and normalize BitMEX BTC perp historical data")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT), help="Cache root directory")
    parser.add_argument("--timeout-sec", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--max-retries", type=int, default=8, help="Max retries per request")
    parser.add_argument("--rate-limit-sleep-sec", type=float, default=0.1, help="Sleep after each successful call")

    sub = parser.add_subparsers(dest="command", required=True)

    p_c = sub.add_parser("candles", help="Download BitMEX 1m candles")
    p_c.add_argument("--symbol", default="XBTUSD", help="BitMEX symbol (default XBTUSD)")
    p_c.add_argument("--start", help="Start day YYYY-MM-DD")
    p_c.add_argument("--end", required=True, help="End day YYYY-MM-DD (inclusive)")
    p_c.add_argument("--chunk-minutes", type=int, default=900, help="Chunk size in minutes (<= --max-rows)")
    p_c.add_argument("--max-rows", type=int, default=1000, help="Max rows requested per API call")
    p_c.add_argument("--backfill-earliest", action="store_true", help="Auto-start from instrument listing date")
    p_c.add_argument(
        "--summary-out",
        default="research/artefacts/logs/btc_bitmex_candles_summary.json",
        help="Summary JSON output path",
    )

    p_f = sub.add_parser("funding", help="Download BitMEX funding history")
    p_f.add_argument("--symbol", default="XBTUSD", help="BitMEX symbol (default XBTUSD)")
    p_f.add_argument("--start", help="Start day YYYY-MM-DD")
    p_f.add_argument("--end", required=True, help="End day YYYY-MM-DD (inclusive)")
    p_f.add_argument("--chunk-days", type=int, default=120, help="Chunk size in days")
    p_f.add_argument("--max-rows", type=int, default=500, help="Max rows requested per API call")
    p_f.add_argument("--backfill-earliest", action="store_true", help="Auto-start from instrument listing date")
    p_f.add_argument(
        "--summary-out",
        default="research/artefacts/logs/btc_bitmex_funding_summary.json",
        help="Summary JSON output path",
    )

    p_full = sub.add_parser("full", help="Download candles+funding and update manifest")
    p_full.add_argument("--symbol", default="XBTUSD", help="BitMEX symbol (default XBTUSD)")
    p_full.add_argument("--start", help="Start day YYYY-MM-DD")
    p_full.add_argument("--end", required=True, help="End day YYYY-MM-DD (inclusive)")
    p_full.add_argument("--chunk-minutes", type=int, default=900, help="Candle chunk size in minutes")
    p_full.add_argument("--chunk-days", type=int, default=120, help="Funding chunk size in days")
    p_full.add_argument("--max-candle-rows", type=int, default=1000, help="Max candle rows per API call")
    p_full.add_argument("--max-funding-rows", type=int, default=500, help="Max funding rows per API call")
    p_full.add_argument("--backfill-earliest", action="store_true", help="Auto-start from instrument listing date")
    p_full.add_argument("--manifest-path", default="research/data_manifest.yml", help="Manifest file to update")
    p_full.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Manifest dataset_id to write")
    p_full.add_argument(
        "--summary-out",
        default="research/artefacts/logs/btc_bitmex_full_summary.json",
        help="Summary JSON output path",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cache_root = Path(args.cache_root)

    if args.command == "candles":
        summary = download_candles(
            symbol=args.symbol,
            start_day=args.start,
            end_day=args.end,
            cache_root=cache_root,
            chunk_minutes=args.chunk_minutes,
            max_rows=args.max_rows,
            backfill_earliest=args.backfill_earliest,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            rate_limit_sleep_sec=args.rate_limit_sleep_sec,
            summary_out=args.summary_out,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "funding":
        summary = download_funding(
            symbol=args.symbol,
            start_day=args.start,
            end_day=args.end,
            cache_root=cache_root,
            chunk_days=args.chunk_days,
            max_rows=args.max_rows,
            backfill_earliest=args.backfill_earliest,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            rate_limit_sleep_sec=args.rate_limit_sleep_sec,
            summary_out=args.summary_out,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "full":
        candle_summary = download_candles(
            symbol=args.symbol,
            start_day=args.start,
            end_day=args.end,
            cache_root=cache_root,
            chunk_minutes=args.chunk_minutes,
            max_rows=args.max_candle_rows,
            backfill_earliest=args.backfill_earliest,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            rate_limit_sleep_sec=args.rate_limit_sleep_sec,
            summary_out=None,
        )
        funding_summary = download_funding(
            symbol=args.symbol,
            start_day=args.start,
            end_day=args.end,
            cache_root=cache_root,
            chunk_days=args.chunk_days,
            max_rows=args.max_funding_rows,
            backfill_earliest=args.backfill_earliest,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            rate_limit_sleep_sec=args.rate_limit_sleep_sec,
            summary_out=None,
        )
        manifest_entry = register_manifest_entry(
            manifest_path=Path(args.manifest_path),
            dataset_id=args.dataset_id,
            cache_root=cache_root,
            candle_summary=candle_summary,
            funding_summary=funding_summary,
        )
        summary = {
            "dataset_id": args.dataset_id,
            "cache_root": str(cache_root),
            "candles": candle_summary,
            "funding": funding_summary,
            "manifest_path": args.manifest_path,
            "manifest_entry": manifest_entry,
        }
        if args.summary_out:
            dump_json(args.summary_out, summary)
        print(json.dumps(summary, indent=2))
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
