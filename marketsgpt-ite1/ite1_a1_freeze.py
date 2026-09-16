#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT=Path(__file__).resolve().parent
LOCK=json.loads((ROOT/"MGPT_ITE1_A1_CONTEXTUAL_TREND_LOCK_20260916.json").read_text())
DATA=ROOT/"data"/"a1_fresh"; DATA.mkdir(parents=True,exist_ok=True)
OUT=ROOT/"results"; OUT.mkdir(parents=True,exist_ok=True)
START=LOCK["windows"]["warmupStart"][:10]
END=LOCK["windows"]["freshInstrumentDevelopment"]["endExclusive"][:10]

def flat(df):
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        fields={"Open","High","Low","Close","Adj Close","Volume"}
        x.columns=[next((str(v) for v in col if str(v) in fields),str(col[0])) for col in x.columns]
    return x

def fetch(sym):
    last=None
    for k in range(4):
        try:
            d=yf.download(sym,start=START,end=END,interval="1h",auto_adjust=True,actions=False,prepost=False,progress=False,threads=False,repair=False,timeout=30)
            d=flat(d)
            if not d.empty:break
            last="empty dataframe"
        except Exception as e:last=repr(e)
        time.sleep(2**k)
    else: raise RuntimeError(f"download failed: {last}")
    req=["Open","High","Low","Close"]
    if any(c not in d.columns for c in req):raise RuntimeError(f"missing OHLC: {list(d.columns)}")
    x=pd.DataFrame({
      "timestamp":pd.to_datetime(d.index,utc=True),
      "open":pd.to_numeric(d["Open"],errors="coerce"),
      "high":pd.to_numeric(d["High"],errors="coerce"),
      "low":pd.to_numeric(d["Low"],errors="coerce"),
      "close":pd.to_numeric(d["Close"],errors="coerce"),
      "volume":pd.to_numeric(d["Volume"],errors="coerce") if "Volume" in d.columns else 0.0
    })
    x=x.dropna(subset=["timestamp","open","high","low","close"])
    x=x[(x.open>0)&(x.high>0)&(x.low>0)&(x.close>0)].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return x

def min_rows(item):
    fam=item["marketFamily"]
    return {"CRYPTO_SPOT":5000,"FX":3500,"METALS_FUTURES_PROXY":2500,"EQUITY":900,"INDEX":900}[fam]

def main():
    manifest=[]; errors=[]
    first_lim=pd.Timestamp("2024-11-05T00:00:00Z"); last_lim=pd.Timestamp("2025-06-30T00:00:00Z")
    for item in LOCK["freshUniverse"]:
        try:
            x=fetch(item["symbol"])
            if len(x)<min_rows(item):raise RuntimeError(f"rows {len(x)} < required {min_rows(item)}")
            first=x.timestamp.iloc[0]; last=x.timestamp.iloc[-1]
            if first>first_lim:raise RuntimeError(f"first {first.isoformat()} too late")
            if last<last_lim:raise RuntimeError(f"last {last.isoformat()} too early")
            p=DATA/f"{item['fileKey']}_1h.csv"; x.to_csv(p,index=False,float_format="%.12g")
            raw=p.read_bytes()
            manifest.append({"symbol":item["symbol"],"marketFamily":item["marketFamily"],"file":p.name,"rows":len(x),
                             "firstUtc":first.isoformat(),"lastUtc":last.isoformat(),"sha256":hashlib.sha256(raw).hexdigest()})
        except Exception as e:
            errors.append({"symbol":item["symbol"],"marketFamily":item["marketFamily"],"error":str(e)})
    status={
      "schema":"mgpt_ite1_a1_data_freeze_v1","generatedAtUtc":datetime.now(timezone.utc).isoformat(),
      "status":"PASS_DATA_FREEZE" if not errors else "BLOCKED_DATA_FREEZE",
      "performanceOutcomesComputed":False,"validationOpened":False,"holdoutOpened":False,
      "lockSha256":hashlib.sha256((ROOT/"MGPT_ITE1_A1_CONTEXTUAL_TREND_LOCK_20260916.json").read_bytes()).hexdigest(),
      "provider":"Yahoo Finance via yfinance","files":manifest,"errors":errors
    }
    p=OUT/"MGPT_ITE1_A1_DATA_FREEZE_STATUS_20260916.json"; p.write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    sums=[f"{x['sha256']}  data/a1_fresh/{x['file']}" for x in manifest]
    sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  results/{p.name}")
    (OUT/"MGPT_ITE1_A1_DATA_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps(status,indent=2))
    if errors:raise SystemExit(3)

if __name__=="__main__":main()
