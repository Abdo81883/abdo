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
LOCK = json.loads((ROOT / "MGPT_ITE1_E1_DEVELOPMENT_LOCK_V3_20260916.json").read_text())
DATA = ROOT / "data" / "e1_v3"
DATA.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

START = LOCK["windows"]["warmupStart"][:10]
END = LOCK["windows"]["development"]["endExclusive"][:10]

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        field_names={"Open","High","Low","Close","Adj Close","Volume"}
        new=[]
        for col in x.columns:
            bits=[str(v) for v in col]
            hit=next((b for b in bits if b in field_names), bits[0])
            new.append(hit)
        x.columns=new
    return x

def fetch(symbol: str) -> pd.DataFrame:
    last=None
    for attempt in range(4):
        try:
            df=yf.download(
                symbol,
                start=START,
                end=END,
                interval="1h",
                auto_adjust=True,
                actions=False,
                prepost=False,
                progress=False,
                threads=False,
                repair=False,
                timeout=30,
            )
            df=flatten_columns(df)
            if not df.empty:
                break
            last="empty dataframe"
        except Exception as e:
            last=repr(e)
        time.sleep(2**attempt)
    else:
        raise RuntimeError(f"{symbol}: download failed: {last}")

    req=["Open","High","Low","Close"]
    if any(c not in df.columns for c in req):
        raise RuntimeError(f"{symbol}: missing columns {req}; got {list(df.columns)}")
    x=pd.DataFrame({
        "timestamp":pd.to_datetime(df.index, utc=True),
        "open":pd.to_numeric(df["Open"],errors="coerce"),
        "high":pd.to_numeric(df["High"],errors="coerce"),
        "low":pd.to_numeric(df["Low"],errors="coerce"),
        "close":pd.to_numeric(df["Close"],errors="coerce"),
        "volume":pd.to_numeric(df["Volume"],errors="coerce") if "Volume" in df.columns else 0.0,
    })
    x=x.dropna(subset=["timestamp","open","high","low","close"])
    x=x[(x["open"]>0)&(x["high"]>0)&(x["low"]>0)&(x["close"]>0)]
    x=x.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return x

def main() -> None:
    gate=LOCK["dataFreezeGate"]
    first_limit=pd.Timestamp(gate["firstBarNoLaterThan"])
    last_limit=pd.Timestamp(gate["lastBarNoEarlierThan"])
    manifest=[]
    errors=[]

    for item in LOCK["universe"]:
        symbol=item["symbol"]; family=item["marketFamily"]; key=item["fileKey"]
        try:
            x=fetch(symbol)
            if x.empty:
                raise RuntimeError("no valid rows")
            first=x["timestamp"].iloc[0]
            last=x["timestamp"].iloc[-1]
            min_rows=int(gate["minimumRowsByFamily"][family])
            if len(x)<min_rows:
                raise RuntimeError(f"rows {len(x)} < required {min_rows}")
            if first>first_limit:
                raise RuntimeError(f"first bar {first.isoformat()} later than {first_limit.isoformat()}")
            if last<last_limit:
                raise RuntimeError(f"last bar {last.isoformat()} earlier than {last_limit.isoformat()}")
            p=DATA/f"{key}_1h.csv"
            x.to_csv(p,index=False,float_format="%.12g")
            raw=p.read_bytes()
            manifest.append({
                "symbol":symbol,"marketFamily":family,"file":p.name,
                "rows":int(len(x)),"firstUtc":first.isoformat(),"lastUtc":last.isoformat(),
                "sha256":hashlib.sha256(raw).hexdigest()
            })
        except Exception as e:
            errors.append({"symbol":symbol,"marketFamily":family,"error":str(e)})

    status={
        "schema":"mgpt_ite1_e1_v3_data_freeze_v1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS_DATA_FREEZE" if not errors else "BLOCKED_DATA_FREEZE",
        "performanceOutcomesComputed":False,
        "freshValidationOpened":False,
        "sealedHoldoutOpened":False,
        "lockSha256":hashlib.sha256((ROOT/"MGPT_ITE1_E1_DEVELOPMENT_LOCK_V3_20260916.json").read_bytes()).hexdigest(),
        "files":manifest,
        "errors":errors,
        "provider":"Yahoo Finance via yfinance",
        "window":{"start":START,"endExclusive":END}
    }
    p=OUT/"MGPT_ITE1_E1_V3_DATA_FREEZE_STATUS_20260916.json"
    p.write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    sums=[]
    for row in manifest:
        sums.append(f"{row['sha256']}  data/e1_v3/{row['file']}")
    sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  results/{p.name}")
    (OUT/"MGPT_ITE1_E1_V3_DATA_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps(status,indent=2))
    if errors:
        raise SystemExit(3)

if __name__=="__main__":
    main()
