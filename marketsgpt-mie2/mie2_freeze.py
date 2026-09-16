#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import io
import json
import math
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "MGPT_MIE2_POSITIONING_OI_LOCK_20260916.json"
LOCK = json.loads(LOCK_PATH.read_text())
OUT = ROOT / "results"
DATA = ROOT / "data" / "positioning_hourly"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
START = pd.Timestamp(LOCK["windows"]["positioningWarmupStart"])
END = pd.Timestamp(LOCK["windows"]["developmentScreen"]["endExclusive"])
REQ = list(LOCK["sourceAuthority"]["requiredColumns"])

def date_strings():
    d = START.normalize()
    out = []
    while d < END:
        out.append(d.strftime("%Y-%m-%d"))
        d += pd.Timedelta(days=1)
    return out

DATES = date_strings()

def get_bytes(url: str, retries: int = 5) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=45)
            if r.status_code == 200 and r.content:
                return r.content
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"{url}: {last}")

def fetch_one(symbol: str, day: str) -> tuple[str,str,pd.DataFrame,str]:
    url = f"{BASE}/{symbol}/{symbol}-metrics-{day}.zip"
    zbytes = get_bytes(url)
    check = get_bytes(url + ".CHECKSUM").decode("utf-8", errors="replace")
    m = re.search(r"\b([0-9a-fA-F]{64})\b", check)
    if not m:
        raise RuntimeError(f"{symbol} {day}: invalid checksum sidecar")
    expected = m.group(1).lower()
    actual = hashlib.sha256(zbytes).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{symbol} {day}: checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"{symbol} {day}: expected one CSV, got {names}")
        raw = zf.read(names[0])
    x = pd.read_csv(io.BytesIO(raw))
    x.columns = [str(c).strip().lower() for c in x.columns]
    missing = [c for c in REQ if c not in x.columns]
    if missing:
        raise RuntimeError(f"{symbol} {day}: missing {missing}; got {list(x.columns)}")
    x = x[REQ].copy()
    x["create_time"] = pd.to_datetime(x["create_time"], utc=True, format="mixed")
    if pd.Timestamp(day, tz="UTC") >= pd.Timestamp("2026-06-25T00:00:00Z"):
        x["availability_time"] = x["create_time"] + pd.Timedelta(minutes=5)
    else:
        x["availability_time"] = x["create_time"]
    for c in REQ:
        if c not in {"create_time","symbol"}:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["availability_time"] + [c for c in REQ if c not in {"create_time","symbol"}])
    return symbol, day, x, actual

def numeric_equal(a: pd.Series, b: pd.Series) -> bool:
    cols = [c for c in REQ if c not in {"create_time","symbol"}]
    av = pd.to_numeric(a[cols], errors="coerce").to_numpy(float)
    bv = pd.to_numeric(b[cols], errors="coerce").to_numpy(float)
    return np.allclose(av, bv, rtol=1e-12, atol=1e-12, equal_nan=True)

def collapse_duplicates(x: pd.DataFrame) -> tuple[pd.DataFrame,int]:
    x = x.sort_values(["availability_time","create_time"]).reset_index(drop=True)
    dup_count = 0
    rows = []
    for _, g in x.groupby("availability_time", sort=True):
        if len(g) == 1:
            rows.append(g.iloc[-1])
            continue
        dup_count += len(g)-1
        first = g.iloc[0]
        if not all(numeric_equal(first, g.iloc[j]) for j in range(1,len(g))):
            raise RuntimeError(f"non-identical normalized duplicate at {g.iloc[0]['availability_time']}")
        rows.append(g.iloc[-1])
    y = pd.DataFrame(rows).reset_index(drop=True)
    return y, dup_count

def hourly_snapshot(x: pd.DataFrame) -> pd.DataFrame:
    # Snapshot at each UTC hour boundary, using only rows available at or before that boundary.
    x = x.sort_values("availability_time").reset_index(drop=True)
    grid = pd.DataFrame({"timestamp": pd.date_range(START, END, freq="1h", inclusive="left", tz="UTC")})
    y = pd.merge_asof(
        grid, x.rename(columns={"availability_time":"metric_time"}).sort_values("metric_time"),
        left_on="timestamp", right_on="metric_time", direction="backward",
        tolerance=pd.Timedelta(minutes=10)
    )
    keep = ["timestamp","metric_time"] + [c for c in REQ if c not in {"create_time","symbol"}]
    return y[keep]

def main():
    errors = []
    by_symbol: dict[str,list[pd.DataFrame]] = {s:[] for s in LOCK["universe"]}
    archive_rows = []

    tasks = [(s,d) for s in LOCK["universe"] for d in DATES]
    with cf.ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(fetch_one,s,d):(s,d) for s,d in tasks}
        for fut in cf.as_completed(futs):
            s,d = futs[fut]
            try:
                symbol,day,df,sha = fut.result()
                by_symbol[symbol].append(df)
                archive_rows.append({"symbol":symbol,"day":day,"zipSha256":sha})
            except Exception as e:
                errors.append({"symbol":s,"day":d,"error":str(e)})

    status_symbols = []
    gate = LOCK["dataFreezeGate"]
    expected_5m = int((END-START)/pd.Timedelta(minutes=5))
    expected_h = int((END-START)/pd.Timedelta(hours=1))

    if not errors:
        for symbol in LOCK["universe"]:
            try:
                raw = pd.concat(by_symbol[symbol], ignore_index=True)
                raw, normalized_dups = collapse_duplicates(raw)
                raw = raw[(raw.availability_time >= START) & (raw.availability_time < END)].copy()
                five_cov = raw.availability_time.nunique() / max(1, expected_5m)
                h = hourly_snapshot(raw)
                h_cov = h["metric_time"].notna().sum() / max(1, expected_h)
                if five_cov < float(gate["minimumMetric5mCoverage"]):
                    raise RuntimeError(f"5m coverage {five_cov:.6f} below gate")
                if h_cov < float(gate["minimumHourlyAvailabilityCoverage"]):
                    raise RuntimeError(f"hourly coverage {h_cov:.6f} below gate")
                p = DATA / f"{symbol}_1h.csv"
                h.to_csv(p, index=False, float_format="%.12g")
                status_symbols.append({
                    "symbol":symbol,
                    "rows5m":int(len(raw)),
                    "metric5mCoverage":float(five_cov),
                    "hourlyRows":int(len(h)),
                    "hourlyAvailabilityCoverage":float(h_cov),
                    "normalizedDuplicateRowsCollapsed":int(normalized_dups),
                    "file":str(p.relative_to(ROOT)),
                    "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                })
            except Exception as e:
                errors.append({"symbol":symbol,"stage":"aggregate","error":str(e)})

    archive_rows = sorted(archive_rows, key=lambda r:(r["symbol"],r["day"]))
    manifest_path = OUT / "MGPT_MIE2_ARCHIVE_MANIFEST_20260916.jsonl"
    manifest_path.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in archive_rows))

    status = {
        "schema":"mgpt_mie2_data_freeze_v1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS_DATA_FREEZE" if not errors and len(status_symbols)==len(LOCK["universe"]) else "BLOCKED_DATA_FREEZE",
        "performanceOutcomesComputed":False,
        "freshValidationOpened":False,
        "sealedHoldoutOpened":False,
        "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "sourceAuthority":LOCK["sourceAuthority"],
        "window":{"start":START.isoformat(),"endExclusive":END.isoformat()},
        "archiveDays":len(DATES),
        "archiveObjectsVerified":len(archive_rows),
        "symbols":status_symbols,
        "archiveManifestFile":manifest_path.name,
        "archiveManifestSha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "errors":errors,
    }
    sp = OUT / "MGPT_MIE2_DATA_FREEZE_STATUS_20260916.json"
    sp.write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    sums = []
    for s in status_symbols:
        sums.append(f"{s['sha256']}  {s['file']}")
    sums.append(f"{status['archiveManifestSha256']}  results/{manifest_path.name}")
    sums.append(f"{hashlib.sha256(sp.read_bytes()).hexdigest()}  results/{sp.name}")
    (OUT/"MGPT_MIE2_DATA_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")

    print(json.dumps({
        "status":status["status"],
        "archiveDays":len(DATES),
        "archiveObjectsVerified":len(archive_rows),
        "symbolsFrozen":len(status_symbols),
        "errorCount":len(errors),
        "errors":errors[:20]
    },indent=2))
    if status["status"] != "PASS_DATA_FREEZE":
        raise SystemExit(3)

if __name__ == "__main__":
    main()
