#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ite1_core import CostModel, audit_reference_opportunity, compute_features, construct_plan, generate_candidates, simulate_trade

ROOT=Path(__file__).resolve().parent
LOCK=json.loads((ROOT/"MGPT_ITE1_E1_DEVELOPMENT_LOCK_V3_20260916.json").read_text())
OUT=ROOT/"results"; OUT.mkdir(parents=True,exist_ok=True)
DEV_START=pd.Timestamp(LOCK["windows"]["development"]["start"])
DEV_END=pd.Timestamp(LOCK["windows"]["development"]["endExclusive"])
MODES=list(LOCK["entryChallengers"])

def load(item):
    p=ROOT/"data"/"e1_v3"/f"{item['fileKey']}_1h.csv"
    x=pd.read_csv(p)
    x["timestamp"]=pd.to_datetime(x["timestamp"],utc=True)
    for c in ["open","high","low","close","volume"]:
        if c in x.columns:x[c]=pd.to_numeric(x[c],errors="coerce")
    if "volume" not in x.columns:x["volume"]=np.nan
    return x.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def r4(df):
    z=df.set_index("timestamp")
    n=z.close.resample("4h",origin="epoch",label="left",closed="left").count()
    o=z.resample("4h",origin="epoch",label="left",closed="left").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    return o[n==4].dropna(subset=["open","high","low","close"]).reset_index()

def cost(fam):
    return CostModel(one_way_fee_bps=float(LOCK["costModelOneWayBps"][fam]["base"]),one_way_slippage_bps=0.0)

def pf(a):
    a=np.asarray(a,float); p=float(a[a>0].sum()); n=float(-a[a<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)

def q(x,p):
    a=np.asarray([v for v in x if math.isfinite(v)],float)
    return float(np.quantile(a,p)) if len(a) else None

def summary(z):
    if len(z)==0:return {"trades":0}
    a=z.netR.to_numpy(float)
    return {
      "trades":int(len(z)),
      "netExpectancyR":float(a.mean()),
      "profitFactor":pf(a),
      "hitRate":float(np.mean(a>0)),
      "meanMfeR":float(z.mfeR.mean()),
      "meanMaeR":float(z.maeR.mean()),
      "medianRiskAtr":float(z.riskAtr.median()),
      "medianTp2R":float(z.tp2R.median()),
      "timeoutFraction":float(np.mean(z.exitType=="TIMEOUT")),
      "stopFraction":float(np.mean(z.exitType=="STOP")),
      "targetFraction":float(np.mean(z.exitType=="TP2"))
    }

def grouped(df,cols):
    out=[]
    for key,z in df.groupby(cols,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        row={cols[i]:str(key[i]) for i in range(len(cols))}
        row.update(summary(z))
        out.append(row)
    return out

def main():
    rows=[]; candidate_count=0
    for item in LOCK["universe"]:
        raw=load(item)
        for tf in LOCK["timeframes"]["evaluated"]:
            df=raw if tf=="1h" else r4(raw)
            if len(df)<230:continue
            f=compute_features(df)
            hold=int(LOCK["frozenComponents"]["maximumHoldingBarsByTimeframe"][tf]); wait=int(LOCK["frozenComponents"]["maximumEntryWaitBars"])
            for c in generate_candidates(f,start_index=200):
                ts=pd.Timestamp(f.iloc[c.signal_index]["timestamp"])
                ts=ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                if not(DEV_START<=ts<DEV_END):continue
                if c.signal_index+hold+wait+1>=len(f):continue
                candidate_count+=1
                opp=audit_reference_opportunity(f,c.signal_index,c.direction,
                    positive_barrier_r=float(LOCK["opportunityAudit"]["positiveBarrierR"]),
                    adverse_barrier_r=float(LOCK["opportunityAudit"]["adverseBarrierR"]),horizon_bars=hold)
                for mode in MODES:
                    p=construct_plan(f,c,entry_mode=mode,stop_mode=LOCK["frozenComponents"]["stop"],
                        target_mode=LOCK["frozenComponents"]["target"],max_entry_wait_bars=wait,max_holding_bars=hold)
                    if p is None:continue
                    rr=simulate_trade(f,p,management_mode=LOCK["frozenComponents"]["management"],cost_model=cost(item["marketFamily"]))
                    if not rr.entered:continue
                    risk=abs(float(rr.entry_price)-float(p.stop_loss))
                    risk_atr=risk/float(p.atr_at_signal) if p.atr_at_signal>0 else float("nan")
                    tp2r=abs(float(p.tp2)-float(rr.entry_price))/risk if risk>0 else float("nan")
                    ex="STOP" if rr.stopped else ("TP2" if rr.tp2_hit else ("TIMEOUT" if rr.timed_out else "OTHER"))
                    rows.append({
                      "symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,
                      "entryMode":mode,"setupClass":c.setup_class,"direction":c.direction,
                      "regime":c.regime,"volState":c.vol_state,"auditGoodOpportunity":bool(opp),
                      "netR":float(rr.net_r),"grossR":float(rr.gross_r),"mfeR":float(rr.mfe_r),"maeR":float(rr.mae_r),
                      "riskAtr":float(risk_atr),"tp2R":float(tp2r),"exitType":ex
                    })
    d=pd.DataFrame(rows)
    if d.empty:raise RuntimeError("diagnostic produced no entered trades")
    mode={}
    for m,z in d.groupby("entryMode"):
        s=summary(z)
        good=z.auditGoodOpportunity.astype(bool)
        s["precisionAgainstAuditOpportunity"]=float(good.mean())
        mode[str(m)]=s
    diagnostics={
      "schema":"mgpt_ite1_e1_v3_diagnostic_v1",
      "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
      "status":"DEVELOPMENT_DIAGNOSTIC_ONLY",
      "e1Result":"REJECT_E1_COMPONENT_GATE",
      "validationOpened":False,"holdoutOpened":False,
      "candidateCount":int(candidate_count),"enteredDiagnosticRows":int(len(d)),
      "modeSummary":mode,
      "setupByMode":grouped(d,["entryMode","setupClass"]),
      "regimeByMode":grouped(d,["entryMode","regime"]),
      "volatilityByMode":grouped(d,["entryMode","volState"]),
      "directionByMode":grouped(d,["entryMode","direction"]),
      "marketFamilyByMode":grouped(d,["entryMode","marketFamily"]),
      "exitByMode":grouped(d,["entryMode","exitType"]),
      "geometry":{
        m:{
          "riskAtrQ10":q(z.riskAtr,.10),"riskAtrQ50":q(z.riskAtr,.50),"riskAtrQ90":q(z.riskAtr,.90),
          "tp2RQ10":q(z.tp2R,.10),"tp2RQ50":q(z.tp2R,.50),"tp2RQ90":q(z.tp2R,.90)
        } for m,z in d.groupby("entryMode")
      },
      "interpretationBoundary":"These are contaminated development diagnostics for architecture hypothesis generation only. No subgroup may be promoted because it looks profitable here; redesigned rules require fresh evidence."
    }
    p=OUT/"MGPT_ITE1_E1_V3_DIAGNOSTIC_20260916.json"
    p.write_text(json.dumps(diagnostics,indent=2,sort_keys=True)+"\n")
    (OUT/"MGPT_ITE1_E1_V3_DIAGNOSTIC_ROWS_20260916.csv").write_text(d.to_csv(index=False))
    (OUT/"MGPT_ITE1_E1_V3_DIAGNOSTIC_SHA256SUMS_20260916.txt").write_text(
      f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
    )
    print(json.dumps({"status":diagnostics["status"],"candidateCount":candidate_count,"modeSummary":mode},indent=2))

if __name__=="__main__":main()
