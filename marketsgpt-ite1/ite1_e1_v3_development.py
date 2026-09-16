#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ite1_core import CostModel, audit_reference_opportunity, compute_features, construct_plan, generate_candidates, simulate_trade

ROOT=Path(__file__).resolve().parent
LOCK_PATH=ROOT/"MGPT_ITE1_E1_DEVELOPMENT_LOCK_V3_20260916.json"
FREEZE_PATH=ROOT/"results"/"MGPT_ITE1_E1_V3_DATA_FREEZE_STATUS_20260916.json"
OUT=ROOT/"results"
LOCK=json.loads(LOCK_PATH.read_text())
DEV_START=pd.Timestamp(LOCK["windows"]["development"]["start"])
DEV_END=pd.Timestamp(LOCK["windows"]["development"]["endExclusive"])
ENTRY_MODES=list(LOCK["entryChallengers"])
TIMEFRAMES=list(LOCK["timeframes"]["evaluated"])

def load_frozen(item):
    p=ROOT/"data"/"e1_v3"/f"{item['fileKey']}_1h.csv"
    if not p.exists(): raise RuntimeError(f"missing {p.name}")
    x=pd.read_csv(p)
    req={"timestamp","open","high","low","close"}
    if not req.issubset(x.columns): raise RuntimeError(f"{p.name}: missing {sorted(req-set(x.columns))}")
    x["timestamp"]=pd.to_datetime(x["timestamp"],utc=True)
    for c in ["open","high","low","close","volume"]:
        if c in x.columns: x[c]=pd.to_numeric(x[c],errors="coerce")
    if "volume" not in x.columns: x["volume"]=np.nan
    x=x.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return x

def resample4h(df):
    z=df.set_index("timestamp")
    n=z["close"].resample("4h",origin="epoch",label="left",closed="left").count()
    o=z.resample("4h",origin="epoch",label="left",closed="left").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    return o[n==4].dropna(subset=["open","high","low","close"]).reset_index()

def cost(fam,stress):
    b=float(LOCK["costModelOneWayBps"][fam]["stress" if stress else "base"])
    return CostModel(one_way_fee_bps=b,one_way_slippage_bps=0.0)

def evaluate_block(item,tf,raw):
    df=raw if tf=="1h" else resample4h(raw)
    if len(df)<230: return [],{"symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,"status":"INSUFFICIENT_BARS","bars":len(df)}
    f=compute_features(df)
    hold=int(LOCK["frozenComponents"]["maximumHoldingBarsByTimeframe"][tf]); wait=int(LOCK["frozenComponents"]["maximumEntryWaitBars"])
    rows=[]; ndev=0
    for c in generate_candidates(f,start_index=200):
        ts=pd.Timestamp(f.iloc[c.signal_index]["timestamp"])
        ts=ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if not (DEV_START<=ts<DEV_END): continue
        if c.signal_index+hold+wait+1>=len(f): continue
        ndev+=1
        cid=f"{item['symbol']}|{tf}|{ts.isoformat()}|{c.direction}|{c.setup_class}|{c.signal_index}"
        opp=audit_reference_opportunity(
            f,c.signal_index,c.direction,
            positive_barrier_r=float(LOCK["opportunityAudit"]["positiveBarrierR"]),
            adverse_barrier_r=float(LOCK["opportunityAudit"]["adverseBarrierR"]),
            horizon_bars=hold
        )
        for mode in ENTRY_MODES:
            p=construct_plan(
                f,c,entry_mode=mode,
                stop_mode=LOCK["frozenComponents"]["stop"],
                target_mode=LOCK["frozenComponents"]["target"],
                max_entry_wait_bars=wait,max_holding_bars=hold
            )
            base={"candidateId":cid,"signalUtc":ts.isoformat(),"symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,
                  "direction":c.direction,"setupClass":c.setup_class,"entryMode":mode,"auditGoodOpportunity":bool(opp)}
            if p is None:
                rows.append({**base,"planValid":False,"entered":False,"netR":0.0,"stressNetR":0.0,"grossR":0.0,"mfeR":0.0,"maeR":0.0,"reason":"NO_VALID_PLAN"})
                continue
            rb=simulate_trade(f,p,management_mode=LOCK["frozenComponents"]["management"],cost_model=cost(item["marketFamily"],False))
            rs=simulate_trade(f,p,management_mode=LOCK["frozenComponents"]["management"],cost_model=cost(item["marketFamily"],True))
            rows.append({**base,"planValid":True,"entered":bool(rb.entered),
                         "netR":float(rb.net_r if rb.entered else 0.0),
                         "stressNetR":float(rs.net_r if rs.entered else 0.0),
                         "grossR":float(rb.gross_r if rb.entered else 0.0),
                         "mfeR":float(rb.mfe_r if rb.entered else 0.0),
                         "maeR":float(rb.mae_r if rb.entered else 0.0),"reason":rb.reason})
    return rows,{"symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,"status":"OK","bars":len(df),"developmentCandidates":ndev}

def pf(a):
    a=np.asarray(a,float); p=float(a[a>0].sum()); n=float(-a[a<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)

def metrics(df,mode):
    z=df[df.entryMode==mode].copy(); ent=z[z.entered].copy()
    net=ent.netR.to_numpy(float); stress=ent.stressNetR.to_numpy(float)
    good=z.auditGoodOpportunity.astype(bool); entered=z.entered.astype(bool)
    captured=int((good&entered).sum()); opp=int(good.sum())
    fam=ent.groupby("marketFamily").netR.mean(); blk=ent.groupby(["symbol","timeframe"]).netR.mean()
    return {
      "candidateRows":int(len(z)),"validPlans":int(z.planValid.sum()),"enteredTrades":int(len(ent)),
      "netExpectancyR":float(net.mean()) if len(net) else 0.0,
      "stressNetExpectancyR":float(stress.mean()) if len(stress) else 0.0,
      "netProfitFactor":pf(net) if len(net) else 0.0,
      "hitRate":float(np.mean(net>0)) if len(net) else 0.0,
      "meanMfeR":float(ent.mfeR.mean()) if len(ent) else 0.0,
      "meanMaeR":float(ent.maeR.mean()) if len(ent) else 0.0,
      "opportunityCount":opp,"opportunityCaptured":captured,
      "opportunityRecall":float(captured/max(1,opp)),
      "precision":float(captured/max(1,int(entered.sum()))),
      "activeMarketFamilies":int(len(fam)),"activeInstrumentTimeframeBlocks":int(len(blk)),
      "positiveMarketFamilyFraction":float((fam>0).mean()) if len(fam) else 0.0,
      "positiveBlockFraction":float((blk>0).mean()) if len(blk) else 0.0,
      "familyExpectancyR":{str(k):float(v) for k,v in fam.items()},
      "blockExpectancyR":{f"{k[0]}|{k[1]}":float(v) for k,v in blk.items()}
    }

def circular(n,rng,bl):
    out=[]
    while len(out)<n:
        s=int(rng.integers(0,n)); take=min(bl,n-len(out))
        out.extend(((s+np.arange(take))%n).tolist())
    return np.asarray(out,int)

def bootstrap(df):
    cfg=LOCK["bootstrap"]; iters=int(cfg["iterations"]); rng=np.random.default_rng(int(cfg["seed"])); bl=int(cfg["blockLengthCandidates"])
    blocks=sorted(df[["symbol","timeframe"]].drop_duplicates().itertuples(index=False,name=None))

    # Computational optimization only: freeze each block into aligned candidate arrays once.
    # Statistical method, seed, block sampling, circular candidate blocks and gate values are unchanged.
    packed={}
    for b in blocks:
        z=df[(df.symbol==b[0])&(df.timeframe==b[1])].copy()
        order=z[["candidateId","signalUtc"]].drop_duplicates().sort_values("signalUtc").candidateId.tolist()
        mode_arrays={}
        for m in ENTRY_MODES:
            q=z[z.entryMode==m].set_index("candidateId")
            entered=np.zeros(len(order),dtype=bool)
            net=np.zeros(len(order),dtype=float)
            for i,cid in enumerate(order):
                if cid in q.index:
                    rr=q.loc[cid]
                    if isinstance(rr,pd.DataFrame): rr=rr.iloc[0]
                    entered[i]=bool(rr["entered"])
                    net[i]=float(rr["netR"]) if entered[i] else 0.0
            mode_arrays[m]={"entered":entered,"net":net,"utility":net.copy()}
        packed[b]={"n":len(order),"modes":mode_arrays}

    abs_s={m:np.zeros(iters) for m in ENTRY_MODES}
    delta={m:np.zeros(iters) for m in ENTRY_MODES if m!="trigger_close"}

    for it in range(iters):
        sampled=[blocks[int(rng.integers(0,len(blocks)))] for _ in blocks]
        tr_sum={m:0.0 for m in ENTRY_MODES}; tr_n={m:0 for m in ENTRY_MODES}
        util_sum={m:0.0 for m in ENTRY_MODES}; util_n={m:0 for m in ENTRY_MODES}

        for b in sampled:
            pb=packed[b]; n=pb["n"]
            if n<=0: continue
            idx=circular(n,rng,bl)
            for m in ENTRY_MODES:
                a=pb["modes"][m]
                ent=a["entered"][idx]
                vals=a["net"][idx]
                tr_sum[m]+=float(vals[ent].sum())
                tr_n[m]+=int(ent.sum())
                util_sum[m]+=float(a["utility"][idx].sum())
                util_n[m]+=int(len(idx))

        for m in ENTRY_MODES:
            abs_s[m][it]=tr_sum[m]/tr_n[m] if tr_n[m] else 0.0
        bu=util_sum["trigger_close"]/util_n["trigger_close"] if util_n["trigger_close"] else 0.0
        for m in delta:
            u=util_sum[m]/util_n[m] if util_n[m] else 0.0
            delta[m][it]=u-bu

    out={"method":cfg["method"],"iterations":iters,"seed":int(cfg["seed"]),"blockLengthCandidates":bl,
         "implementation":"array-optimized-equivalent","modes":{}}
    for m in ENTRY_MODES:
        a=abs_s[m]
        d={"probabilityPositiveNetExpectancy":float(np.mean(a>0)),
           "ci95LowNetExpectancyR":float(np.quantile(a,.025)),"ci95HighNetExpectancyR":float(np.quantile(a,.975))}
        if m!="trigger_close":
            q=delta[m]
            d.update({"probabilityPositiveCandidateUtilityDeltaVsBaseline":float(np.mean(q>0)),
                      "ci95LowCandidateUtilityDeltaVsBaseline":float(np.quantile(q,.025)),
                      "ci95HighCandidateUtilityDeltaVsBaseline":float(np.quantile(q,.975))})
        out["modes"][m]=d
    return out

def checks(m,b,baseline,mode):
    g=LOCK["developmentDecisionRule"]; bb=b["modes"][mode]
    c={
      "minimumTotalEnteredTrades":m["enteredTrades"]>=int(g["minimumTotalEnteredTrades"]),
      "minimumActiveMarketFamilies":m["activeMarketFamilies"]>=int(g["minimumActiveMarketFamilies"]),
      "minimumActiveInstrumentTimeframeBlocks":m["activeInstrumentTimeframeBlocks"]>=int(g["minimumActiveInstrumentTimeframeBlocks"]),
      "minimumNetExpectancyR":m["netExpectancyR"]>=float(g["minimumNetExpectancyR"]),
      "minimumStressNetExpectancyR":m["stressNetExpectancyR"]>=float(g["minimumStressNetExpectancyR"]),
      "minimumProfitFactor":m["netProfitFactor"]>=float(g["minimumProfitFactor"]),
      "minimumOpportunityRecall":m["opportunityRecall"]>=float(g["minimumOpportunityRecall"]),
      "minimumPrecision":m["precision"]>=float(g["minimumPrecision"]),
      "minimumPositiveMarketFamilyFraction":m["positiveMarketFamilyFraction"]>=float(g["minimumPositiveMarketFamilyFraction"]),
      "minimumPositiveBlockFraction":m["positiveBlockFraction"]>=float(g["minimumPositiveBlockFraction"]),
      "bootstrapProbabilityPositiveExpectancyMin":bb["probabilityPositiveNetExpectancy"]>=float(g["bootstrapProbabilityPositiveExpectancyMin"])
    }
    if mode!="trigger_close":
        c["challengerBootstrapProbabilityDeltaVsBaselinePositiveMin"]=bb["probabilityPositiveCandidateUtilityDeltaVsBaseline"]>=float(g["challengerBootstrapProbabilityDeltaVsBaselinePositiveMin"])
        c["maximumRelativeRecallDegradationVsBaseline"]=m["opportunityRecall"]>=baseline["opportunityRecall"]*(1.0-float(g["maximumRelativeRecallDegradationVsBaseline"]))
    return {k:bool(v) for k,v in c.items()}

def select(mm,cc):
    passing=[m for m in ENTRY_MODES if all(cc[m].values())]
    if not passing:return None,"NO_ENTRY_MODE_CLEARED_ABSOLUTE_AND_ROBUSTNESS_GATES"
    ch=[m for m in passing if m!="trigger_close"]
    if ch:
        simp={"trigger_close":0,"retest_limit":1,"structure_confirmed_retest":2}
        w=max(ch,key=lambda m:(mm[m]["stressNetExpectancyR"],mm[m]["opportunityRecall"],mm[m]["precision"],-mm[m]["meanMaeR"],-simp[m]))
        return w,"CHALLENGER_CLEARED_ABSOLUTE_AND_INCREMENTAL_GATES"
    return ("trigger_close","BASELINE_CLEARED_ABSOLUTE_GATE_NO_CHALLENGER_PROVED_INCREMENTAL_IMPROVEMENT") if "trigger_close" in passing else (None,"NO_ELIGIBLE_WINNER")

def main():
    if not FREEZE_PATH.exists(): raise RuntimeError("V3 freeze manifest missing")
    freeze=json.loads(FREEZE_PATH.read_text())
    if freeze.get("status")!="PASS_DATA_FREEZE": raise RuntimeError(f"V3 freeze is not PASS: {freeze.get('status')}")
    # Re-hash files against freeze manifest immediately before any outcome computation.
    manifest={x["file"]:x for x in freeze["files"]}
    for item in LOCK["universe"]:
        fn=f"{item['fileKey']}_1h.csv"; p=ROOT/"data"/"e1_v3"/fn
        if fn not in manifest: raise RuntimeError(f"{fn} absent from frozen manifest")
        if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest[fn]["sha256"]: raise RuntimeError(f"{fn} hash mismatch")

    all_rows=[]; block_diag=[]
    for item in LOCK["universe"]:
        raw=load_frozen(item)
        for tf in TIMEFRAMES:
            rr,dd=evaluate_block(item,tf,raw); all_rows.extend(rr); block_diag.append(dd)
    if not all_rows: raise RuntimeError("no E1 rows")
    ledger=pd.DataFrame(all_rows)
    mm={m:metrics(ledger,m) for m in ENTRY_MODES}; boot=bootstrap(ledger); baseline=mm["trigger_close"]
    cc={m:checks(mm[m],boot,baseline,m) for m in ENTRY_MODES}; winner,reason=select(mm,cc)
    status="PASS_E1_FREEZE_ENTRY_WINNER" if winner else "REJECT_E1_COMPONENT_GATE"
    close={
      "schema":"mgpt_ite1_e1_v3_development_closeout_v1","generatedAtUtc":datetime.now(timezone.utc).isoformat(),
      "status":status,"selectedEntryMode":winner,"selectionReason":reason,
      "productionAuthority":False,"r15MutationAllowed":False,"freshValidationOpened":False,"sealedHoldoutOpened":False,
      "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
      "dataFreezeSha256":hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest(),
      "metricsByEntryMode":mm,"bootstrap":boot,"checksByEntryMode":cc,
      "failedChecksByEntryMode":{m:[k for k,v in cc[m].items() if not v] for m in ENTRY_MODES},
      "blockDiagnostics":block_diag,
      "semanticLimitations":LOCK["semanticBoundaries"],
      "nextAction":("Independent provider replication before opening ITE1-S1." if winner else
                    "Keep S1/T1/M1 sealed. Reassess candidate/setup architecture under a new prelock; no E1 threshold rescue and no validation/holdout access.")
    }
    cp=OUT/"MGPT_ITE1_E1_V3_DEVELOPMENT_CLOSEOUT_20260916.json"; lp=OUT/"MGPT_ITE1_E1_V3_DEVELOPMENT_LEDGER_20260916.csv"
    cp.write_text(json.dumps(close,indent=2,sort_keys=True)+"\n"); ledger.to_csv(lp,index=False)
    sums=[f"{hashlib.sha256(cp.read_bytes()).hexdigest()}  {cp.name}",f"{hashlib.sha256(lp.read_bytes()).hexdigest()}  {lp.name}"]
    (OUT/"MGPT_ITE1_E1_V3_DEVELOPMENT_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps({"status":status,"selectedEntryMode":winner,"selectionReason":reason,
                      "failedChecksByEntryMode":close["failedChecksByEntryMode"],
                      "headlineMetrics":{m:{k:mm[m][k] for k in ["enteredTrades","netExpectancyR","stressNetExpectancyR","netProfitFactor","opportunityRecall","precision","positiveMarketFamilyFraction","positiveBlockFraction"]} for m in ENTRY_MODES}},indent=2))

if __name__=="__main__":
    main()
