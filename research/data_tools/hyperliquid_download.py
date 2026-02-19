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

from .common import dump_json, ensure_dir, parse_utc_ts, to_iso_z
from .validate_dataset import validate_file

API_URL = "https://api.hyperliquid.xyz/info"

DEFAULT_CACHE_ROOT = Path("research/data_cache/btc_hl_perp")


class HyperliquidClient:
    def __init__(
        self,
        timeout_sec: int = 30,
        max_retries: int = 6,
        base_backoff_sec: float = 1.0,
        max_backoff_sec: float = 30.0,
        rate_limit_sleep_sec: float = 0.25,
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

    def post_info(self, payload: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(API_URL, json=payload, timeout=self.timeout_sec)
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
                    raise RuntimeError(f"Hyperliquid returned invalid JSON for payload={payload}") from exc
                if self.rate_limit_sleep_sec > 0:
                    time.sleep(self.rate_limit_sleep_sec)
                return data

            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                self._sleep_backoff(attempt, response.headers.get("Retry-After"))
                continue

            body = response.text.strip().replace("\n", " ")
            raise RuntimeError(
                f"Hyperliquid request failed status={response.status_code} "
                f"type={payload.get('type')} body={body[:320]}"
            )

        raise RuntimeError(f"Hyperliquid request failed after retries: {last_error}")


def _parse_day_start_ms(day_ymd: str) -> int:
    ts = parse_utc_ts(f"{day_ymd}T00:00:00Z")
    return int(ts.value // 1_000_000)


def _end_day_exclusive_ms(end_day: str) -> int:
    end_exclusive = parse_utc_ts(f"{end_day}T00:00:00Z") + pd.Timedelta(days=1)
    return int(end_exclusive.value // 1_000_000)


def _day_bounds_ms(start_day: str, end_day: str) -> tuple[int, int]:
    start_ms = _parse_day_start_ms(start_day)
    end_ms_exclusive = _end_day_exclusive_ms(end_day)
    if end_ms_exclusive <= start_ms:
        raise ValueError(f"end date must be >= start date (start={start_day}, end={end_day})")
    return start_ms, end_ms_exclusive


def _iter_chunks(start_ms: int, end_ms_exclusive: int, chunk_ms: int) -> Iterable[tuple[int, int]]:
    cur = start_ms
    while cur < end_ms_exclusive:
        nxt = min(cur + chunk_ms, end_ms_exclusive)
        yield cur, nxt
        cur = nxt


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _stream_progress(progress: dict[str, Any], stream: str, stream_key: str) -> dict[str, Any]:
    container = progress.setdefault(stream, {})
    entry = container.setdefault(stream_key, {})
    entry.setdefault("completed_chunks", {})
    return entry


def _chunk_id(start_ms: int, end_ms_exclusive: int) -> str:
    return f"{start_ms}-{end_ms_exclusive}"


def _chunk_file(dir_path: Path, prefix: str, start_ms: int, end_ms_exclusive: int) -> Path:
    return dir_path / f"{prefix}_{start_ms}_{end_ms_exclusive}.json"


def _read_chunk_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("rows", [])
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
        return []
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def _write_chunk(
    path: Path,
    stream: str,
    coin: str,
    start_ms: int,
    end_ms_exclusive: int,
    rows: list[dict[str, Any]],
    extra_meta: dict[str, Any] | None = None,
) -> None:
    payload = {
        "stream": stream,
        "coin": coin,
        "start_ms": start_ms,
        "end_ms_exclusive": end_ms_exclusive,
        "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row_count": len(rows),
        "rows": rows,
    }
    if extra_meta:
        payload.update(extra_meta)
    dump_json(path, payload)


def _sorted_chunk_paths(raw_dir: Path, prefix: str) -> list[Path]:
    out: list[tuple[int, int, Path]] = []
    for path in raw_dir.glob(f"{prefix}_*_*.json"):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        start_ms = _to_int(parts[-2])
        end_ms = _to_int(parts[-1])
        if start_ms is None or end_ms is None:
            continue
        out.append((start_ms, end_ms, path))
    out.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in out]


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
    merged = merged.sort_values("ts_utc_ns").drop_duplicates(subset=["ts_utc_ns"], keep="last")
    merged = merged.reset_index(drop=True)
    return merged[columns]


def _ms_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    ts = pd.to_datetime(ms, unit="ms", utc=True, errors="coerce")
    return to_iso_z(ts)


def _validate_candles(parquet_path: Path) -> dict[str, Any]:
    validation = validate_file(parquet_path, kind="ohlc", expected_seconds=60)
    if not validation["ok"]:
        raise RuntimeError(f"Hyperliquid candle validation failed: {validation['errors']}")
    return validation


def _validate_funding(parquet_path: Path) -> dict[str, Any]:
    validation = validate_file(parquet_path, kind="series")
    if not validation["ok"]:
        raise RuntimeError(f"Hyperliquid funding validation failed: {validation['errors']}")
    return validation


def _fetch_candle_chunk(
    client: HyperliquidClient,
    coin: str,
    interval: str,
    start_ms: int,
    end_ms_exclusive: int,
) -> list[dict[str, Any]]:
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms_exclusive - 1,
        },
    }
    data = client.post_info(payload)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected candle response type: {type(data).__name__}")
    return [x for x in data if isinstance(x, dict)]


def _fetch_funding_chunk(
    client: HyperliquidClient,
    coin: str,
    start_ms: int,
    end_ms_exclusive: int,
) -> list[dict[str, Any]]:
    req_payload = {
        "type": "fundingHistory",
        "req": {
            "coin": coin,
            "startTime": start_ms,
            "endTime": end_ms_exclusive - 1,
        },
    }
    try:
        data = client.post_info(req_payload)
    except RuntimeError as exc:
        # Hyperliquid fundingHistory currently expects flattened args.
        if "status=422" not in str(exc):
            raise
        flat_payload = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": start_ms,
            "endTime": end_ms_exclusive - 1,
        }
        data = client.post_info(flat_payload)

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected funding response type: {type(data).__name__}")
    return [x for x in data if isinstance(x, dict)]


def download_candles(
    coin: str,
    interval: str,
    start_day: str | None,
    end_day: str,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    chunk_minutes: int = 4000,
    backfill_earliest: bool = False,
    max_empty_chunks: int = 2,
    timeout_sec: int = 30,
    max_retries: int = 6,
    rate_limit_sleep_sec: float = 0.25,
    summary_out: str | Path | None = None,
) -> dict[str, Any]:
    if interval != "1m":
        raise ValueError("Only interval=1m is currently supported for normalization")
    if chunk_minutes <= 0 or chunk_minutes > 4000:
        raise ValueError("chunk_minutes must be in 1..4000")
    if not backfill_earliest and not start_day:
        raise ValueError("--start is required unless --backfill-earliest is used")

    raw_dir = cache_root / "raw" / "candles"
    normalized_path = cache_root / "normalized" / "candles_1m.parquet"
    progress_path = cache_root / "progress.json"

    ensure_dir(raw_dir)
    ensure_dir(normalized_path.parent)

    end_ms_exclusive = _end_day_exclusive_ms(end_day)
    start_ms = None
    if not backfill_earliest:
        assert start_day is not None
        start_ms, end_ms_exclusive = _day_bounds_ms(start_day, end_day)

    chunk_ms = chunk_minutes * 60_000

    progress = _load_progress(progress_path)
    stream_key = f"{coin.upper()}:{interval}"
    stream = _stream_progress(progress, "candles", stream_key)
    completed = stream["completed_chunks"]

    total_chunks = 0
    fetched_chunks = 0
    reused_chunks = 0

    client = HyperliquidClient(
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        rate_limit_sleep_sec=rate_limit_sleep_sec,
    )

    try:
        if backfill_earliest:
            cursor_end = end_ms_exclusive
            saw_data = False
            empty_after_data = 0

            while cursor_end > 0:
                chunk_start = max(0, cursor_end - chunk_ms)
                chunk_end = cursor_end
                total_chunks += 1

                cid = _chunk_id(chunk_start, chunk_end)
                path = _chunk_file(raw_dir, f"{coin.lower()}_{interval}", chunk_start, chunk_end)
                done = completed.get(cid)

                if done and path.exists():
                    rows = _read_chunk_rows(path)
                    reused_chunks += 1
                else:
                    rows = _fetch_candle_chunk(client, coin.upper(), interval, chunk_start, chunk_end)
                    _write_chunk(
                        path,
                        stream="candles",
                        coin=coin.upper(),
                        start_ms=chunk_start,
                        end_ms_exclusive=chunk_end,
                        rows=rows,
                        extra_meta={"interval": interval},
                    )
                    completed[cid] = {
                        "path": str(path),
                        "row_count": len(rows),
                        "first_ts_ms": _to_int(rows[0].get("t")) if rows else None,
                        "last_ts_ms": _to_int(rows[-1].get("t")) if rows else None,
                        "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                    fetched_chunks += 1

                times = sorted(_to_int(r.get("t")) for r in rows if _to_int(r.get("t")) is not None)
                if times:
                    saw_data = True
                    empty_after_data = 0
                    next_end = max(times[0], chunk_start)
                else:
                    if saw_data:
                        empty_after_data += 1
                        if empty_after_data >= max_empty_chunks:
                            break
                    next_end = chunk_start

                if next_end >= cursor_end:
                    break
                cursor_end = next_end
                if chunk_start == 0 and not times:
                    break
        else:
            assert start_ms is not None
            for chunk_start, chunk_end in _iter_chunks(start_ms, end_ms_exclusive, chunk_ms):
                total_chunks += 1
                cid = _chunk_id(chunk_start, chunk_end)
                path = _chunk_file(raw_dir, f"{coin.lower()}_{interval}", chunk_start, chunk_end)
                done = completed.get(cid)
                if done and path.exists():
                    reused_chunks += 1
                    continue

                rows = _fetch_candle_chunk(client, coin.upper(), interval, chunk_start, chunk_end)
                _write_chunk(
                    path,
                    stream="candles",
                    coin=coin.upper(),
                    start_ms=chunk_start,
                    end_ms_exclusive=chunk_end,
                    rows=rows,
                    extra_meta={"interval": interval},
                )
                completed[cid] = {
                    "path": str(path),
                    "row_count": len(rows),
                    "first_ts_ms": _to_int(rows[0].get("t")) if rows else None,
                    "last_ts_ms": _to_int(rows[-1].get("t")) if rows else None,
                    "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                fetched_chunks += 1

        stream["updated_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _save_progress(progress_path, progress)
    finally:
        client.close()

    chunk_paths = _sorted_chunk_paths(raw_dir, f"{coin.lower()}_{interval}")
    if not chunk_paths:
        raise RuntimeError("No candle chunk files exist after download")

    records: list[dict[str, Any]] = []
    for path in chunk_paths:
        for row in _read_chunk_rows(path):
            ts_ms = _to_int(row.get("t"))
            if ts_ms is None:
                continue
            records.append(
                {
                    "ts_utc_ns": ts_ms * 1_000_000,
                    "open": _to_float(row.get("o")),
                    "high": _to_float(row.get("h")),
                    "low": _to_float(row.get("l")),
                    "close": _to_float(row.get("c")),
                    "volume": _to_float(row.get("v")),
                    "source": "hyperliquid:api.info:candleSnapshot",
                    "coin": coin.upper(),
                    "market": "perp",
                }
            )

    columns = ["ts_utc_ns", "open", "high", "low", "close", "volume", "source", "coin", "market"]
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame = frame.dropna(subset=["ts_utc_ns", "open", "high", "low", "close", "volume"])
    if frame.empty:
        raise RuntimeError("No candle rows were downloaded for the requested range")

    frame["ts_utc_ns"] = pd.to_numeric(frame["ts_utc_ns"], errors="coerce").astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])

    merged = _merge_by_ts_ns(normalized_path, frame, columns=columns)
    merged.to_parquet(normalized_path, index=False)

    validation = _validate_candles(normalized_path)
    summary = {
        "dataset": "BTC_HYPERLIQUID_PERP_1M",
        "stream": "candles",
        "coin": coin.upper(),
        "interval": interval,
        "backfill_earliest": backfill_earliest,
        "requested_start_day": start_day,
        "requested_end_day": end_day,
        "requested_start_ts_utc": _ms_to_iso(start_ms),
        "requested_end_ts_utc_exclusive": _ms_to_iso(end_ms_exclusive),
        "raw_dir": str(raw_dir),
        "normalized_path": str(normalized_path),
        "progress_path": str(progress_path),
        "chunk_minutes": chunk_minutes,
        "chunks_total": total_chunks,
        "chunks_fetched": fetched_chunks,
        "chunks_reused": reused_chunks,
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
    coin: str,
    start_day: str | None,
    end_day: str,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    chunk_days: int = 14,
    backfill_earliest: bool = False,
    max_empty_chunks: int = 2,
    timeout_sec: int = 30,
    max_retries: int = 6,
    rate_limit_sleep_sec: float = 0.25,
    summary_out: str | Path | None = None,
) -> dict[str, Any]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be > 0")
    if not backfill_earliest and not start_day:
        raise ValueError("--start is required unless --backfill-earliest is used")

    raw_dir = cache_root / "raw" / "funding"
    normalized_path = cache_root / "normalized" / "funding.parquet"
    progress_path = cache_root / "progress.json"

    ensure_dir(raw_dir)
    ensure_dir(normalized_path.parent)

    end_ms_exclusive = _end_day_exclusive_ms(end_day)
    start_ms = None
    if not backfill_earliest:
        assert start_day is not None
        start_ms, end_ms_exclusive = _day_bounds_ms(start_day, end_day)

    chunk_ms = chunk_days * 24 * 60 * 60 * 1000

    progress = _load_progress(progress_path)
    stream_key = coin.upper()
    stream = _stream_progress(progress, "funding", stream_key)
    completed = stream["completed_chunks"]

    total_chunks = 0
    fetched_chunks = 0
    reused_chunks = 0

    client = HyperliquidClient(
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        rate_limit_sleep_sec=rate_limit_sleep_sec,
    )

    try:
        if backfill_earliest:
            cursor_end = end_ms_exclusive
            saw_data = False
            empty_after_data = 0

            while cursor_end > 0:
                chunk_start = max(0, cursor_end - chunk_ms)
                chunk_end = cursor_end
                total_chunks += 1

                cid = _chunk_id(chunk_start, chunk_end)
                path = _chunk_file(raw_dir, coin.lower(), chunk_start, chunk_end)
                done = completed.get(cid)

                if done and path.exists():
                    rows = _read_chunk_rows(path)
                    reused_chunks += 1
                else:
                    rows = _fetch_funding_chunk(client, coin.upper(), chunk_start, chunk_end)
                    _write_chunk(
                        path,
                        stream="funding",
                        coin=coin.upper(),
                        start_ms=chunk_start,
                        end_ms_exclusive=chunk_end,
                        rows=rows,
                    )
                    completed[cid] = {
                        "path": str(path),
                        "row_count": len(rows),
                        "first_ts_ms": _to_int(rows[0].get("time")) if rows else None,
                        "last_ts_ms": _to_int(rows[-1].get("time")) if rows else None,
                        "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                    fetched_chunks += 1

                times = sorted(_to_int(r.get("time")) for r in rows if _to_int(r.get("time")) is not None)
                if times:
                    saw_data = True
                    empty_after_data = 0
                    next_end = max(times[0], chunk_start)
                else:
                    if saw_data:
                        empty_after_data += 1
                        if empty_after_data >= max_empty_chunks:
                            break
                    next_end = chunk_start

                if next_end >= cursor_end:
                    break
                cursor_end = next_end
                if chunk_start == 0 and not times:
                    break
        else:
            assert start_ms is not None
            for chunk_start, chunk_end in _iter_chunks(start_ms, end_ms_exclusive, chunk_ms):
                total_chunks += 1
                cid = _chunk_id(chunk_start, chunk_end)
                path = _chunk_file(raw_dir, coin.lower(), chunk_start, chunk_end)
                done = completed.get(cid)
                if done and path.exists():
                    reused_chunks += 1
                    continue

                rows = _fetch_funding_chunk(client, coin.upper(), chunk_start, chunk_end)
                _write_chunk(
                    path,
                    stream="funding",
                    coin=coin.upper(),
                    start_ms=chunk_start,
                    end_ms_exclusive=chunk_end,
                    rows=rows,
                )
                completed[cid] = {
                    "path": str(path),
                    "row_count": len(rows),
                    "first_ts_ms": _to_int(rows[0].get("time")) if rows else None,
                    "last_ts_ms": _to_int(rows[-1].get("time")) if rows else None,
                    "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                fetched_chunks += 1

        stream["updated_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _save_progress(progress_path, progress)
    finally:
        client.close()

    chunk_paths = _sorted_chunk_paths(raw_dir, coin.lower())
    if not chunk_paths:
        raise RuntimeError("No funding chunk files exist after download")

    records: list[dict[str, Any]] = []
    for path in chunk_paths:
        for row in _read_chunk_rows(path):
            ts_ms = _to_int(row.get("time"))
            if ts_ms is None:
                continue
            records.append(
                {
                    "ts_utc_ns": ts_ms * 1_000_000,
                    "funding_rate_raw": _to_float(row.get("fundingRate")),
                    "source": "hyperliquid:api.info:fundingHistory",
                    "coin": str(row.get("coin") or coin).upper(),
                    "market": "perp",
                }
            )

    columns = ["ts_utc_ns", "funding_rate_raw", "source", "coin", "market"]
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame = frame.dropna(subset=["ts_utc_ns", "funding_rate_raw"])
    if frame.empty:
        raise RuntimeError("No funding rows were downloaded for the requested range")

    frame["ts_utc_ns"] = pd.to_numeric(frame["ts_utc_ns"], errors="coerce").astype("int64")
    frame["funding_rate_raw"] = pd.to_numeric(frame["funding_rate_raw"], errors="coerce")
    frame = frame.dropna(subset=["funding_rate_raw"])

    merged = _merge_by_ts_ns(normalized_path, frame, columns=columns)
    merged.to_parquet(normalized_path, index=False)

    validation = _validate_funding(normalized_path)
    summary = {
        "dataset": "BTC_HYPERLIQUID_PERP_1M",
        "stream": "funding",
        "coin": coin.upper(),
        "backfill_earliest": backfill_earliest,
        "requested_start_day": start_day,
        "requested_end_day": end_day,
        "requested_start_ts_utc": _ms_to_iso(start_ms),
        "requested_end_ts_utc_exclusive": _ms_to_iso(end_ms_exclusive),
        "raw_dir": str(raw_dir),
        "normalized_path": str(normalized_path),
        "progress_path": str(progress_path),
        "chunk_days": chunk_days,
        "chunks_total": total_chunks,
        "chunks_fetched": fetched_chunks,
        "chunks_reused": reused_chunks,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and normalize Hyperliquid perpetual market data")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT), help="Cache root directory")
    parser.add_argument("--timeout-sec", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--max-retries", type=int, default=6, help="Max retries per request")
    parser.add_argument("--rate-limit-sleep-sec", type=float, default=0.25, help="Sleep after each successful call")

    sub = parser.add_subparsers(dest="command", required=True)

    p_c = sub.add_parser("candles", help="Download Hyperliquid candles")
    p_c.add_argument("--coin", default="BTC", help="Coin symbol, e.g. BTC")
    p_c.add_argument("--interval", default="1m", help="Candle interval (currently supports 1m)")
    p_c.add_argument("--start", help="Start date YYYY-MM-DD (UTC)")
    p_c.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC, inclusive)")
    p_c.add_argument("--chunk-minutes", type=int, default=4000, help="Per-request window in minutes (<= 4000)")
    p_c.add_argument("--backfill-earliest", action="store_true", help="Step backward until API returns no older candles")
    p_c.add_argument("--max-empty-chunks", type=int, default=2, help="Stop backfill after this many empty chunks after data")
    p_c.add_argument(
        "--summary-out",
        default="research/artefacts/logs/btc_hl_candles_summary.json",
        help="Summary JSON output path",
    )

    p_f = sub.add_parser("funding", help="Download Hyperliquid funding history")
    p_f.add_argument("--coin", default="BTC", help="Coin symbol, e.g. BTC")
    p_f.add_argument("--start", help="Start date YYYY-MM-DD (UTC)")
    p_f.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC, inclusive)")
    p_f.add_argument("--chunk-days", type=int, default=14, help="Per-request window in days")
    p_f.add_argument("--backfill-earliest", action="store_true", help="Step backward until API returns no older funding")
    p_f.add_argument("--max-empty-chunks", type=int, default=2, help="Stop backfill after this many empty chunks after data")
    p_f.add_argument(
        "--summary-out",
        default="research/artefacts/logs/btc_hl_funding_summary.json",
        help="Summary JSON output path",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cache_root = Path(args.cache_root)

    if args.command == "candles":
        summary = download_candles(
            coin=args.coin,
            interval=args.interval,
            start_day=args.start,
            end_day=args.end,
            cache_root=cache_root,
            chunk_minutes=args.chunk_minutes,
            backfill_earliest=args.backfill_earliest,
            max_empty_chunks=args.max_empty_chunks,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            rate_limit_sleep_sec=args.rate_limit_sleep_sec,
            summary_out=args.summary_out,
        )
    elif args.command == "funding":
        summary = download_funding(
            coin=args.coin,
            start_day=args.start,
            end_day=args.end,
            cache_root=cache_root,
            chunk_days=args.chunk_days,
            backfill_earliest=args.backfill_earliest,
            max_empty_chunks=args.max_empty_chunks,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            rate_limit_sleep_sec=args.rate_limit_sleep_sec,
            summary_out=args.summary_out,
        )
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")

    print(json.dumps(summary, indent=2))
    print(
        "Summary: "
        f"start={summary.get('start_ts_utc')} end={summary.get('end_ts_utc')} "
        f"rows={summary.get('row_count')} gaps={summary.get('gaps_found')} duplicates={summary.get('duplicates_found')}"
    )


if __name__ == "__main__":
    main()
