#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
LEDGER=OUT/"MGPT_ITE1_A1_TRADE_LEDGER_20260916.csv"

def pf(a):
    a=np.asarray(a,float);p=float(a[a>0].sum());n=float(-a[a<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)

def summ(z):
    if z.empty:return {"trades":0}
    a=z.netR.to_numpy(float)
    return {
      "trades":int(len(z)),"netExpectancyR":float(a.mean()),"profitFactor":pf(a),
      "hitRate":float(np.mean(a>0)),"stressNetExpectancyR":float(z.stressNetR.mean()),
      "meanMfeR":float(z.mfeR.mean()),"meanMaeR":float(z.maeR.mean()),
      "medianRiskAtr":float(z.riskAtr.median()),"medianTp2R":float(z.tp2R.median()),
      "stopFraction":float(np.mean(z.exitType=="STOP")),"tp2Fraction":float(np.mean(z.exitType=="TP2")),
      "timeoutFraction":float(np.mean(z.exitType=="TIMEOUT"))
    }

def grouped(d,cols):
    out=[]
    for k,z in d.groupby(cols,dropna=False):
        if not isinstance(k,tuple):k=(k,)
        row={cols[i]:str(k[i]) for i in range(len(cols))}
        row.update(summ(z));out.append(row)
    return out

def main():
    d=pd.read_csv(LEDGER)
    d["riskBucket"]=pd.cut(d.riskAtr,[-np.inf,.75,1.0,1.5,2.0,2.5,np.inf],right=True).astype(str)
    d["rewardBucket"]=pd.cut(d.tp2R,[-np.inf,1.5,1.75,2.0,2.5,3.0,np.inf],right=True).astype(str)
    diag={
      "schema":"mgpt_ite1_a1_diagnostic_v1",
      "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
      "status":"CONTAMINATED_DIAGNOSTIC_ONLY",
      "a1Verdict":"REJECT_A1_FRESH_INSTRUMENT_GATE",
      "validationOpened":False,"holdoutOpened":False,
      "overall":summ(d),
      "bySetup":grouped(d,["setupClass"]),
      "byMarketFamily":grouped(d,["marketFamily"]),
      "byTimeframe":grouped(d,["timeframe"]),
      "byDirection":grouped(d,["direction"]),
      "byVolatility":grouped(d,["volState"]),
      "bySetupAndVolatility":grouped(d,["setupClass","volState"]),
      "byRiskBucket":grouped(d,["riskBucket"]),
      "byRewardBucket":grouped(d,["rewardBucket"]),
      "byExit":grouped(d,["exitType"]),
      "byMarketAndSetup":grouped(d,["marketFamily","setupClass"]),
      "interpretationBoundary":"These diagnostics may motivate B1/C1 locks only. No subgroup, threshold, market or direction may be promoted from A1 because it looks profitable here."
    }
    p=OUT/"MGPT_ITE1_A1_DIAGNOSTIC_20260916.json";p.write_text(json.dumps(diag,indent=2,sort_keys=True)+"\n")
    (OUT/"MGPT_ITE1_A1_DIAGNOSTIC_SHA256SUMS_20260916.txt").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    print(json.dumps({"status":diag["status"],"overall":diag["overall"],"bySetup":diag["bySetup"],"byVolatility":diag["byVolatility"]},indent=2))

if __name__=="__main__":main()
