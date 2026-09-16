#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "MGPT_IFE1_PROGRAM_LOCK_20260916.json"
LOCK = json.loads(LOCK_PATH.read_text())
DATA = ROOT / "data" / "development"
CTX = ROOT / "data" / "context"
OUT = ROOT / "results"
for p in [DATA, CTX, OUT]:
    p.mkdir(parents=True, exist_ok=True)

MARKET_START = LOCK["windows"]["warmupStart"][:10]
MARKET_END = LOCK["windows"]["developmentScreen"]["endExclusive"][:10]
CONTEXT_START = LOCK["windows"]["contextWarmupStart"][:10]
CONTEXT_END = MARKET_END


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        cols = []
        for col in x.columns:
            bits = [str(v) for v in col]
            cols.append(next((b for b in bits if b in fields), bits[0]))
        x.columns = cols
    return x


def fetch(symbol: str, *, start: str, end: str, interval: str) -> pd.DataFrame:
    last = None
    for attempt in range(5):
        try:
            d = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                actions=False,
                prepost=False,
                progress=False,
                threads=False,
                repair=False,
                timeout=40,
            )
            d = flatten_columns(d)
            if not d.empty:
                break
            last = "empty dataframe"
        except Exception as e:
            last = repr(e)
        time.sleep(min(20, 2 ** attempt))
    else:
        raise RuntimeError(f"{symbol}: download failed after retries: {last}")

    req = ["Open", "High", "Low", "Close"]
    if any(c not in d.columns for c in req):
        raise RuntimeError(f"{symbol}: missing OHLC; got {list(d.columns)}")

    x = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(d.index, utc=True),
            "open": pd.to_numeric(d["Open"], errors="coerce"),
            "high": pd.to_numeric(d["High"], errors="coerce"),
            "low": pd.to_numeric(d["Low"], errors="coerce"),
            "close": pd.to_numeric(d["Close"], errors="coerce"),
            "volume": pd.to_numeric(d["Volume"], errors="coerce") if "Volume" in d.columns else 0.0,
        }
    )
    x = x.dropna(subset=["timestamp", "open", "high", "low", "close"])
    x = x[(x.open > 0) & (x.high > 0) & (x.low > 0) & (x.close > 0)]
    x = x.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return x


def file_record(path: Path, symbol: str, family: str | None, rows: int, first, last) -> dict:
    return {
        "symbol": symbol,
        "marketFamily": family,
        "file": path.name,
        "rows": int(rows),
        "firstUtc": pd.Timestamp(first).isoformat(),
        "lastUtc": pd.Timestamp(last).isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    gate = LOCK["dataFreezeGate"]
    first_limit = pd.Timestamp(gate["firstMarketBarNoLaterThan"])
    last_limit = pd.Timestamp(gate["lastMarketBarNoEarlierThan"])
    manifest = []
    context_manifest = []
    errors = []

    for n, item in enumerate(LOCK["universe"], start=1):
        symbol = item["symbol"]
        fam = item["marketFamily"]
        try:
            x = fetch(symbol, start=MARKET_START, end=MARKET_END, interval="1h")
            min_rows = int(gate["minimumRowsByFamily"][fam])
            if len(x) < min_rows:
                raise RuntimeError(f"rows {len(x)} < required {min_rows}")
            first = x.timestamp.iloc[0]
            last = x.timestamp.iloc[-1]
            if first > first_limit:
                raise RuntimeError(f"first bar {first.isoformat()} later than {first_limit.isoformat()}")
            if last < last_limit:
                raise RuntimeError(f"last bar {last.isoformat()} earlier than {last_limit.isoformat()}")
            p = DATA / f"{item['fileKey']}_1h.csv"
            x.to_csv(p, index=False, float_format="%.12g")
            manifest.append(file_record(p, symbol, fam, len(x), first, last))
        except Exception as e:
            errors.append({"kind": "market", "symbol": symbol, "marketFamily": fam, "error": str(e)})
        if n % 8 == 0:
            time.sleep(1.0)

    for item in LOCK["contextSeries"]:
        symbol = item["symbol"]
        try:
            x = fetch(symbol, start=CONTEXT_START, end=CONTEXT_END, interval="1d")
            min_rows = int(gate["minimumContextDailyRows"])
            if len(x) < min_rows:
                raise RuntimeError(f"context rows {len(x)} < required {min_rows}")
            p = CTX / f"{item['fileKey']}_1d.csv"
            x.to_csv(p, index=False, float_format="%.12g")
            context_manifest.append(file_record(p, symbol, None, len(x), x.timestamp.iloc[0], x.timestamp.iloc[-1]))
        except Exception as e:
            errors.append({"kind": "context", "symbol": symbol, "error": str(e)})

    status = {
        "schema": "mgpt_ife1_data_freeze_v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DATA_FREEZE" if not errors else "BLOCKED_DATA_FREEZE",
        "performanceOutcomesComputed": False,
        "freshValidationOpened": False,
        "sealedHoldoutOpened": False,
        "lockSha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "provider": LOCK["provider"]["developmentTransport"],
        "marketWindow": {"start": MARKET_START, "endExclusive": MARKET_END},
        "contextWindow": {"start": CONTEXT_START, "endExclusive": CONTEXT_END},
        "marketFiles": manifest,
        "contextFiles": context_manifest,
        "errors": errors,
    }

    sp = OUT / "MGPT_IFE1_DATA_FREEZE_STATUS_20260916.json"
    sp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    sums = []
    for row in manifest:
        sums.append(f"{row['sha256']}  data/development/{row['file']}")
    for row in context_manifest:
        sums.append(f"{row['sha256']}  data/context/{row['file']}")
    sums.append(f"{hashlib.sha256(sp.read_bytes()).hexdigest()}  results/{sp.name}")
    (OUT / "MGPT_IFE1_DATA_SHA256SUMS_20260916.txt").write_text("\n".join(sums) + "\n")

    print(json.dumps({
        "status": status["status"],
        "marketFiles": len(manifest),
        "contextFiles": len(context_manifest),
        "errors": errors,
    }, indent=2))

    if errors:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
