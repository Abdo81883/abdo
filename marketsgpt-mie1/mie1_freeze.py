#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "MGPT_MIE1_MICROSTRUCTURE_LOCK_20260916.json"
LOCK = json.loads(LOCK_PATH.read_text())
BASE = "https://data.binance.vision/data/futures/um/monthly"
OUT = ROOT / "results"
CONTRACT_DIR = ROOT / "data" / "contract"
PREMIUM_DIR = ROOT / "data" / "premium"
FUNDING_DIR = ROOT / "data" / "funding"
for p in [OUT, CONTRACT_DIR, PREMIUM_DIR, FUNDING_DIR]:
    p.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp(LOCK["windows"]["archiveWarmupStart"])
END = pd.Timestamp(LOCK["windows"]["developmentScreen"]["endExclusive"])
MONTHS = list(LOCK["archiveMonths"])

KLINE_COLS = [
    "open_time","open","high","low","close","volume","close_time",
    "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
]


def get_bytes(url: str, retries: int = 5) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=40)
            if r.status_code == 200 and r.content:
                return r.content
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(10, 2 ** attempt))
    raise RuntimeError(f"download failed: {url}: {last}")


def verify_zip(url: str) -> tuple[bytes, str]:
    z = get_bytes(url)
    checksum = get_bytes(url + ".CHECKSUM").decode("utf-8", errors="replace").strip()
    m = re.search(r"\b([0-9a-fA-F]{64})\b", checksum)
    if not m:
        raise RuntimeError(f"invalid checksum file: {url}.CHECKSUM")
    expected = m.group(1).lower()
    actual = hashlib.sha256(z).hexdigest()
    if actual != expected:
        raise RuntimeError(f"checksum mismatch {url}: {actual} != {expected}")
    return z, actual


def unzip_csv(zbytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"expected one CSV in ZIP, found {names}")
        return zf.read(names[0])


def detect_timestamp_unit(s: pd.Series) -> str:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return "ms"
    med = float(v.median())
    return "us" if med > 1e14 else "ms"


def read_kline_csv(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8", errors="replace")
    first = text.splitlines()[0] if text else ""
    has_header = any(x in first.lower() for x in ["open_time","open time","open"])
    if has_header:
        x = pd.read_csv(io.StringIO(text))
        norm = {str(c).strip().lower().replace(" ","_"): c for c in x.columns}
        rename = {}
        synonyms = {
            "open_time":"open_time","open":"open","high":"high","low":"low","close":"close",
            "volume":"volume","close_time":"close_time","quote_volume":"quote_volume",
            "quote_asset_volume":"quote_volume","number_of_trades":"trades","trades":"trades","count":"trades",
            "taker_buy_base_asset_volume":"taker_buy_base","taker_buy_base_volume":"taker_buy_base","taker_buy_volume":"taker_buy_base",
            "taker_buy_quote_asset_volume":"taker_buy_quote","taker_buy_quote_volume":"taker_buy_quote",
        }
        for k,v in synonyms.items():
            if k in norm:
                rename[norm[k]] = v
        x = x.rename(columns=rename)
        if "open_time" not in x.columns:
            x.columns = KLINE_COLS[:len(x.columns)]
    else:
        x = pd.read_csv(io.StringIO(text), header=None)
        x.columns = KLINE_COLS[:len(x.columns)]
    req = ["open_time","open","high","low","close"]
    if any(c not in x.columns for c in req):
        raise RuntimeError(f"kline missing columns; got {list(x.columns)}")
    for c in [c for c in KLINE_COLS if c in x.columns and c not in {"open_time","close_time"}]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    unit = detect_timestamp_unit(x["open_time"])
    x["timestamp"] = pd.to_datetime(pd.to_numeric(x["open_time"], errors="coerce"), unit=unit, utc=True)
    keep = ["timestamp","open","high","low","close"]
    for c in ["volume","quote_volume","trades","taker_buy_base","taker_buy_quote"]:
        if c in x.columns:
            keep.append(c)
    x = x[keep].dropna(subset=["timestamp","open","high","low","close"])
    return x


def read_funding_csv(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8", errors="replace")
    first = text.splitlines()[0].lower() if text else ""
    has_header = any(k in first for k in ["calc_time","funding","time"])
    x = pd.read_csv(io.StringIO(text), header=0 if has_header else None)
    if not has_header:
        # Current archive format is normally calc_time, funding_interval_hours, last_funding_rate.
        if x.shape[1] >= 3:
            x.columns = ["calc_time","funding_interval_hours","last_funding_rate"] + [f"extra_{i}" for i in range(x.shape[1]-3)]
        elif x.shape[1] == 2:
            x.columns = ["calc_time","last_funding_rate"]
        else:
            raise RuntimeError("funding archive has too few columns")
    norm = {str(c).strip().lower().replace(" ","_"): c for c in x.columns}
    time_col = next((norm[k] for k in ["calc_time","fundingtime","funding_time","time"] if k in norm), None)
    rate_col = next((norm[k] for k in ["last_funding_rate","fundingrate","funding_rate"] if k in norm), None)
    if time_col is None or rate_col is None:
        raise RuntimeError(f"funding columns unrecognized: {list(x.columns)}")
    unit = detect_timestamp_unit(x[time_col])
    y = pd.DataFrame({
        "timestamp": pd.to_datetime(pd.to_numeric(x[time_col], errors="coerce"), unit=unit, utc=True),
        "funding_rate": pd.to_numeric(x[rate_col], errors="coerce"),
    })
    if "funding_interval_hours" in norm:
        y["funding_interval_hours"] = pd.to_numeric(x[norm["funding_interval_hours"]], errors="coerce")
    return y.dropna(subset=["timestamp","funding_rate"])


def kline_url(dtype: str, symbol: str, month: str) -> str:
    return f"{BASE}/{dtype}/{symbol}/1h/{symbol}-1h-{month}.zip"


def funding_url(symbol: str, month: str) -> str:
    return f"{BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


def coverage_hourly(x: pd.DataFrame) -> float:
    expected = int((END - START) / pd.Timedelta(hours=1))
    z = x[(x.timestamp >= START) & (x.timestamp < END)]
    return float(z.timestamp.nunique() / max(1, expected))


def funding_window_counts(x: pd.DataFrame) -> list[int]:
    windows = [
        (pd.Timestamp("2025-07-01T00:00:00Z"), pd.Timestamp("2025-12-28T00:00:00Z")),
        (pd.Timestamp("2025-11-01T00:00:00Z"), pd.Timestamp("2026-04-30T00:00:00Z")),
    ]
    return [int(x[(x.timestamp >= a) & (x.timestamp < b)].timestamp.nunique()) for a,b in windows]


def write_csv(x: pd.DataFrame, path: Path) -> dict:
    x = x.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    x.to_csv(path, index=False, float_format="%.12g")
    return {
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(x)),
        "firstUtc": x.timestamp.iloc[0].isoformat() if len(x) else None,
        "lastUtc": x.timestamp.iloc[-1].isoformat() if len(x) else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    errors = []
    manifest = []
    archive_evidence = []

    for symbol in LOCK["universe"]:
        contract_parts = []
        premium_parts = []
        funding_parts = []

        for month in MONTHS:
            for dtype, target in [("klines", contract_parts), ("premiumIndexKlines", premium_parts)]:
                url = kline_url(dtype, symbol, month)
                try:
                    z, sha = verify_zip(url)
                    target.append(read_kline_csv(unzip_csv(z)))
                    archive_evidence.append({"symbol":symbol,"month":month,"dataset":dtype,"url":url,"zipSha256":sha})
                except Exception as e:
                    errors.append({"symbol":symbol,"month":month,"dataset":dtype,"error":str(e)})
            furl = funding_url(symbol, month)
            try:
                z, sha = verify_zip(furl)
                funding_parts.append(read_funding_csv(unzip_csv(z)))
                archive_evidence.append({"symbol":symbol,"month":month,"dataset":"fundingRate","url":furl,"zipSha256":sha})
            except Exception as e:
                errors.append({"symbol":symbol,"month":month,"dataset":"fundingRate","error":str(e)})

        if errors:
            continue

        contract = pd.concat(contract_parts, ignore_index=True).sort_values("timestamp")
        premium = pd.concat(premium_parts, ignore_index=True).sort_values("timestamp")
        funding = pd.concat(funding_parts, ignore_index=True).sort_values("timestamp")

        dup_c = int(contract.timestamp.duplicated().sum())
        dup_p = int(premium.timestamp.duplicated().sum())
        dup_f = int(funding.timestamp.duplicated().sum())
        cov_c = coverage_hourly(contract)
        cov_p = coverage_hourly(premium)
        fcounts = funding_window_counts(funding)

        gate = LOCK["dataFreezeGate"]
        if dup_c or dup_p or dup_f:
            errors.append({"symbol":symbol,"dataset":"combined","error":f"duplicate timestamps contract={dup_c} premium={dup_p} funding={dup_f}"})
            continue
        if cov_c < float(gate["minimumContractHourlyCoverage"]):
            errors.append({"symbol":symbol,"dataset":"klines","error":f"coverage {cov_c:.6f} below gate"})
            continue
        if cov_p < float(gate["minimumPremiumHourlyCoverage"]):
            errors.append({"symbol":symbol,"dataset":"premiumIndexKlines","error":f"coverage {cov_p:.6f} below gate"})
            continue
        if min(fcounts) < int(gate["minimumFundingEventsPer180Days"]):
            errors.append({"symbol":symbol,"dataset":"fundingRate","error":f"funding 180d counts {fcounts} below gate"})
            continue

        cmeta = write_csv(contract, CONTRACT_DIR / f"{symbol}_1h.csv")
        pmeta = write_csv(premium, PREMIUM_DIR / f"{symbol}_1h.csv")
        fmeta = write_csv(funding, FUNDING_DIR / f"{symbol}.csv")
        manifest.append({
            "symbol":symbol,
            "contract":cmeta,
            "premium":pmeta,
            "funding":fmeta,
            "contractCoverage":cov_c,
            "premiumCoverage":cov_p,
            "funding180dCounts":fcounts,
        })

    status = {
        "schema":"mgpt_mie1_data_freeze_v1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS_DATA_FREEZE" if not errors and len(manifest)==len(LOCK["universe"]) else "BLOCKED_DATA_FREEZE",
        "performanceOutcomesComputed":False,
        "freshValidationOpened":False,
        "sealedHoldoutOpened":False,
        "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "sourceAuthority":LOCK["sourceAuthority"],
        "symbols":manifest,
        "archives":archive_evidence,
        "errors":errors,
    }
    sp = OUT / "MGPT_MIE1_DATA_FREEZE_STATUS_20260916.json"
    sp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    sums = []
    for m in manifest:
        for k in ["contract","premium","funding"]:
            sums.append(f"{m[k]['sha256']}  {m[k]['file']}")
    sums.append(f"{hashlib.sha256(sp.read_bytes()).hexdigest()}  results/{sp.name}")
    (OUT / "MGPT_MIE1_DATA_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")

    print(json.dumps({
        "status":status["status"],
        "symbolsFrozen":len(manifest),
        "archiveObjectsVerified":len(archive_evidence),
        "errors":errors[:20],
        "errorCount":len(errors),
    }, indent=2))
    if status["status"] != "PASS_DATA_FREEZE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
