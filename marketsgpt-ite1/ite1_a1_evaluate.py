#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ite1_core import (
    Candidate,
    CostModel,
    audit_reference_opportunity,
    compute_features,
    construct_plan,
    generate_candidates,
    simulate_trade,
)

ROOT=Path(__file__).resolve().parent
LOCK_PATH=ROOT/"MGPT_ITE1_A1_CONTEXTUAL_TREND_LOCK_20260916.json"
SPEC_PATH=ROOT/"MGPT_ITE1_A1_IMPLEMENTATION_SPEC_V1_20260916.json"
FREEZE_PATH=ROOT/"results"/"MGPT_ITE1_A1_DATA_FREEZE_STATUS_20260916.json"
OUT=ROOT/"results"
LOCK=json.loads(LOCK_PATH.read_text())
SPEC=json.loads(SPEC_PATH.read_text())
DEV_START=pd.Timestamp(LOCK["windows"]["freshInstrumentDevelopment"]["start"])
DEV_END=pd.Timestamp(LOCK["windows"]["freshInstrumentDevelopment"]["endExclusive"])
PRIORITY={"BREAKOUT":0,"PULLBACK":1,"SWEEP_RECLAIM":2}

@dataclass(frozen=True)
class A1Plan:
    symbol:str
    market_family:str
    timeframe:str
    signal_index:int
    signal_time:pd.Timestamp
    direction:str
    setup_class:str
    entry_index:int
    entry:float
    stop:float
    tp1:float
    tp2:float
    tp3:float
    atr:float
    risk_atr:float
    htf_state:str
    vol_state:str

def read_frozen(item):
    p=ROOT/"data"/"a1_fresh"/f"{item['fileKey']}_1h.csv"
    if not p.exists(): raise RuntimeError(f"missing A1 file {p.name}")
    x=pd.read_csv(p)
    req={"timestamp","open","high","low","close"}
    if not req.issubset(x.columns): raise RuntimeError(f"{p.name}: missing {sorted(req-set(x.columns))}")
    x["timestamp"]=pd.to_datetime(x["timestamp"],utc=True)
    for c in ["open","high","low","close","volume"]:
        if c in x.columns: x[c]=pd.to_numeric(x[c],errors="coerce")
    if "volume" not in x.columns: x["volume"]=np.nan
    return x.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def resample_4h(df):
    z=df.set_index("timestamp")
    n=z["close"].resample("4h",origin="epoch",label="left",closed="left").count()
    o=z.resample("4h",origin="epoch",label="left",closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    return o[n==4].dropna(subset=["open","high","low","close"]).reset_index()

def resample_day(df):
    z=df.set_index("timestamp")
    n=z["close"].resample("1D",origin="epoch",label="left",closed="left").count()
    o=z.resample("1D",origin="epoch",label="left",closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    return o[n>0].dropna(subset=["open","high","low","close"]).reset_index()

def context_features(df, duration):
    x=df.copy()
    x["ema20"]=x["close"].ewm(span=20,adjust=False,min_periods=20).mean()
    x["ema50"]=x["close"].ewm(span=50,adjust=False,min_periods=50).mean()
    x["ema50_lag5"]=x["ema50"].shift(5)
    x["end_time"]=x["timestamp"]+duration
    return x

def htf_table(raw,tf,family):
    if tf=="1h" and family not in {"EQUITY","INDEX"}:
        return context_features(resample_4h(raw),pd.Timedelta(hours=4))
    return context_features(resample_day(raw),pd.Timedelta(days=1))

def last_completed_htf_state(htf,decision_time):
    if htf.empty:return "UNKNOWN"
    ends=htf["end_time"].astype("int64").to_numpy()
    pos=int(np.searchsorted(ends,decision_time.value,side="right")-1)
    if pos<0:return "UNKNOWN"
    r=htf.iloc[pos]
    vals=[r["ema20"],r["ema50"],r["ema50_lag5"],r["close"]]
    if any(pd.isna(v) for v in vals):return "UNKNOWN"
    e20,e50,e50p,c=map(float,vals)
    if e20>e50 and c>e20 and e50>e50p:return "LONG"
    if e20<e50 and c<e20 and e50<e50p:return "SHORT"
    return "NEUTRAL"

def market_cost(fam,stress=False):
    b=float(LOCK["costModelOneWayBps"][fam]["stress" if stress else "base"])
    return CostModel(one_way_fee_bps=b,one_way_slippage_bps=0.0)

def candidate_key(symbol,tf,ts,direction):
    return f"{symbol}|{tf}|{ts.isoformat()}|{direction}"

def full_horizon_ok(df,entry_i,hold):
    end_i=entry_i+hold
    return end_i<len(df) and pd.Timestamp(df.iloc[end_i]["timestamp"])<DEV_END

def reference_rows(item,tf,df,f):
    hold=int(LOCK["holdBars"][tf])
    raw=generate_candidates(f,start_index=200)
    dedup={}
    for c in raw:
        ts=pd.Timestamp(f.iloc[c.signal_index]["timestamp"])
        ts=ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if not(DEV_START<=ts<DEV_END):continue
        if c.signal_index+1>=len(df) or not full_horizon_ok(df,c.signal_index+1,hold):continue
        k=candidate_key(item["symbol"],tf,ts,c.direction)
        old=dedup.get(k)
        if old is None or PRIORITY[c.setup_class]<PRIORITY[old.setup_class]:
            dedup[k]=c
    rows=[]
    for k,c in dedup.items():
        ts=pd.Timestamp(f.iloc[c.signal_index]["timestamp"])
        ts=ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        opp=audit_reference_opportunity(
            f,c.signal_index,c.direction,
            positive_barrier_r=float(LOCK["riskGeometry"]["tp2MinimumR"]),
            adverse_barrier_r=1.0,
            horizon_bars=hold
        )
        p=construct_plan(
            f,c,entry_mode="trigger_close",
            stop_mode="structure_plus_atr_noise",
            target_mode="hybrid_structure_volatility_ladder",
            max_entry_wait_bars=4,max_holding_bars=hold
        )
        br=None
        if p is not None:
            br=simulate_trade(f,p,management_mode="fixed_full_exit",cost_model=market_cost(item["marketFamily"],False))
        rows.append({
            "key":k,"symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,
            "signalUtc":ts.isoformat(),"direction":c.direction,"referenceSetup":c.setup_class,
            "auditGoodOpportunity":bool(opp),
            "baselineEntered":bool(br.entered) if br else False,
            "baselineNetR":float(br.net_r) if br and br.entered else 0.0
        })
    return rows

def local_direction(f,i):
    if i<205:return "NEUTRAL"
    r=f.iloc[i]
    vals=[r["ema20"],r["ema50"],r["ema200"],r["close"],f.iloc[i-5]["ema50"]]
    if any(pd.isna(v) for v in vals):return "NEUTRAL"
    e20,e50,e200,c,e50p=map(float,vals)
    if e20>e50>e200 and c>e20 and e50>e50p:return "LONG"
    if e20<e50<e200 and c<e20 and e50<e50p:return "SHORT"
    return "NEUTRAL"

def target_candidates(f,i,entry,direction,atr):
    r=f.iloc[i]
    ph20=float(r["prior_high20"]); pl20=float(r["prior_low20"])
    ph50=float(r["prior_high50"]) if not pd.isna(r["prior_high50"]) else math.nan
    pl50=float(r["prior_low50"]) if not pd.isna(r["prior_low50"]) else math.nan
    ph120=float(r["prior_high120"]) if not pd.isna(r["prior_high120"]) else math.nan
    pl120=float(r["prior_low120"]) if not pd.isna(r["prior_low120"]) else math.nan
    rg=ph20-pl20
    if not(math.isfinite(rg) and rg>0):return []
    if direction=="LONG":
        vals=[ph50,ph120,ph20+rg,ph20+1.5*rg,entry+atr,entry+2*atr,entry+3*atr]
        vals=[float(v) for v in vals if math.isfinite(v) and v>entry]
        return sorted(set(round(v,12) for v in vals))
    vals=[pl50,pl120,pl20-rg,pl20-1.5*rg,entry-atr,entry-2*atr,entry-3*atr]
    vals=[float(v) for v in vals if math.isfinite(v) and v<entry]
    return sorted(set(round(v,12) for v in vals),reverse=True)

def choose_targets(vals,direction,entry,risk,atr):
    floors=[float(LOCK["riskGeometry"]["tp1MinimumR"]),float(LOCK["riskGeometry"]["tp2MinimumR"]),float(LOCK["riskGeometry"]["tp3MinimumR"])]
    chosen=[]; last=None
    for floor in floors:
        pick=None
        for v in vals:
            rr=(v-entry)/risk if direction=="LONG" else (entry-v)/risk
            if rr+1e-12<floor:continue
            if last is not None:
                if direction=="LONG" and v<=last+0.10*atr:continue
                if direction=="SHORT" and v>=last-0.10*atr:continue
            pick=float(v);break
        if pick is None:return None
        chosen.append(pick);last=pick
        vals=[v for v in vals if (v>pick if direction=="LONG" else v<pick)]
    return tuple(chosen)

def detect_a1(item,tf,df,f,htf):
    out=[]
    hold=int(LOCK["holdBars"][tf])
    for i in range(205,len(f)-1):
        ts=pd.Timestamp(f.iloc[i]["timestamp"])
        ts=ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if not(DEV_START<=ts<DEV_END):continue
        entry_i=i+1
        if not full_horizon_ok(df,entry_i,hold):continue
        r=f.iloc[i]
        atr=float(r["atr14"]) if not pd.isna(r["atr14"]) else math.nan
        if not(math.isfinite(atr) and atr>0):continue
        if str(r["vol_state"]) not in {"NORMAL","EXPANSION"}:continue
        ld=local_direction(f,i)
        if ld not in {"LONG","SHORT"}:continue
        decision_time=pd.Timestamp(df.iloc[entry_i]["timestamp"])
        decision_time=decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        hd=last_completed_htf_state(htf,decision_time)
        if hd!=ld:continue
        o,h,l,c=map(float,[r["open"],r["high"],r["low"],r["close"]])
        bar_range=h-l
        if bar_range<=0:continue
        body=abs(c-o);clv=(c-l)/bar_range
        ph=float(r["prior_high20"]);pl=float(r["prior_low20"])
        breakout=False;pullback=False
        if ld=="LONG":
            breakout=c>ph and bar_range>=0.80*atr and body>=0.35*atr and clv>=0.65
            if not breakout:
                e20=float(r["ema20"])
                pullback=l<=e20<=h and c>e20 and c>o and clv>=0.60
        else:
            breakout=c<pl and bar_range>=0.80*atr and body>=0.35*atr and clv<=0.35
            if not breakout:
                e20=float(r["ema20"])
                pullback=l<=e20<=h and c<e20 and c<o and clv<=0.40
        if not(breakout or pullback):continue
        setup="BREAKOUT_CONTINUATION" if breakout else "PULLBACK_CONTINUATION"
        entry=float(df.iloc[entry_i]["open"])
        if breakout:
            stop=ph-0.35*atr if ld=="LONG" else pl+0.35*atr
        else:
            if i<5:continue
            if ld=="LONG":
                prior5=float(df.iloc[i-5:i]["low"].min());stop=min(l,prior5)-0.15*atr
            else:
                prior5=float(df.iloc[i-5:i]["high"].max());stop=max(h,prior5)+0.15*atr
        if (ld=="LONG" and not stop<entry) or (ld=="SHORT" and not stop>entry):continue
        risk=abs(entry-stop); risk_atr=risk/atr
        if risk_atr<0.50 or risk_atr>2.50:continue
        vals=target_candidates(f,i,entry,ld,atr)
        t=choose_targets(vals,ld,entry,risk,atr)
        if t is None:continue
        out.append(A1Plan(item["symbol"],item["marketFamily"],tf,i,ts,ld,setup,entry_i,entry,float(stop),t[0],t[1],t[2],atr,risk_atr,hd,str(r["vol_state"])))
    return out

def simulate_a1(df,p,stress=False):
    sign=1 if p.direction=="LONG" else -1
    risk=abs(p.entry-p.stop);hold=int(LOCK["holdBars"][p.timeframe])
    end_i=p.entry_index+hold
    if end_i>=len(df):return None
    mfe=0.0;mae=0.0;exit_price=None;exit_i=None;etype=None
    for j in range(p.entry_index,end_i+1):
        h=float(df.iloc[j]["high"]);l=float(df.iloc[j]["low"])
        mfe=max(mfe,(h-p.entry)/risk if sign==1 else (p.entry-l)/risk)
        mae=max(mae,(p.entry-l)/risk if sign==1 else (h-p.entry)/risk)
        stop_hit=l<=p.stop if sign==1 else h>=p.stop
        if stop_hit:
            exit_price=p.stop;exit_i=j;etype="STOP";break
        tp_hit=h>=p.tp2 if sign==1 else l<=p.tp2
        if tp_hit:
            exit_price=p.tp2;exit_i=j;etype="TP2";break
    if exit_price is None:
        exit_i=end_i;exit_price=float(df.iloc[end_i]["close"]);etype="TIMEOUT"
    gross=sign*(exit_price-p.entry)/risk
    rate=market_cost(p.market_family,stress).one_way_total_rate
    friction=rate*(abs(p.entry)+abs(exit_price))/risk
    return {
      "grossR":float(gross),"netR":float(gross-friction),"mfeR":float(mfe),"maeR":float(mae),
      "exitType":etype,"exitIndex":int(exit_i),"exitPrice":float(exit_price),
      "tp1":p.tp1,"tp2":p.tp2,"tp3":p.tp3,
      "tp1R":abs(p.tp1-p.entry)/risk,"tp2R":abs(p.tp2-p.entry)/risk,"tp3R":abs(p.tp3-p.entry)/risk
    }

def pf(a):
    a=np.asarray(a,float);p=float(a[a>0].sum());n=float(-a[a<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)

def max_drawdown_r(a):
    eq=0.0;peak=0.0;dd=0.0
    for r in a:
        eq+=float(r);peak=max(peak,eq);dd=max(dd,peak-eq)
    return float(dd)

def bootstrap(ref):
    cfg=LOCK["bootstrap"];iters=int(cfg["iterations"]);seed=int(cfg["seed"]);bl=int(cfg["blockLengthCandidates"])
    rng=np.random.default_rng(seed)
    blocks=sorted(ref[["symbol","timeframe"]].drop_duplicates().itertuples(index=False,name=None))
    packed={}
    for b in blocks:
        z=ref[(ref.symbol==b[0])&(ref.timeframe==b[1])].sort_values("signalUtc").reset_index(drop=True)
        packed[b]={
          "n":len(z),
          "entered":z.a1Entered.to_numpy(bool),
          "a1":z.a1NetR.to_numpy(float),
          "base":z.baselineNetR.to_numpy(float)
        }
    abs_s=np.zeros(iters);delta=np.zeros(iters)
    for it in range(iters):
        tsum=0.0;tn=0;du=0.0;un=0
        sampled=[blocks[int(rng.integers(0,len(blocks)))] for _ in blocks]
        for b in sampled:
            p=packed[b];n=p["n"]
            if n<=0:continue
            idx=[]
            while len(idx)<n:
                st=int(rng.integers(0,n));take=min(bl,n-len(idx))
                idx.extend(((st+np.arange(take))%n).tolist())
            idx=np.asarray(idx,int)
            ent=p["entered"][idx];aa=p["a1"][idx];bb=p["base"][idx]
            tsum+=float(aa[ent].sum());tn+=int(ent.sum())
            du+=float((aa-bb).sum());un+=len(idx)
        abs_s[it]=tsum/tn if tn else 0.0
        delta[it]=du/un if un else 0.0
    return {
      "method":cfg["method"],"iterations":iters,"seed":seed,"blockLengthCandidates":bl,
      "probabilityPositiveNetExpectancy":float(np.mean(abs_s>0)),
      "ci95NetExpectancyR":[float(np.quantile(abs_s,.025)),float(np.quantile(abs_s,.975))],
      "probabilityPositiveUtilityDeltaVsFrozenE1Baseline":float(np.mean(delta>0)),
      "ci95UtilityDeltaVsFrozenE1Baseline":[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))]
    }

def main():
    freeze=json.loads(FREEZE_PATH.read_text())
    if freeze.get("status")!="PASS_DATA_FREEZE":raise RuntimeError("A1 data freeze not PASS")
    manifest={x["file"]:x for x in freeze["files"]}
    for item in LOCK["freshUniverse"]:
        fn=f"{item['fileKey']}_1h.csv";p=ROOT/"data"/"a1_fresh"/fn
        if fn not in manifest:raise RuntimeError(f"{fn} not in manifest")
        if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest[fn]["sha256"]:raise RuntimeError(f"{fn} hash mismatch")

    ref_rows=[];trade_rows=[];block_diag=[]
    for item in LOCK["freshUniverse"]:
        raw=read_frozen(item)
        for tf in item["timeframes"]:
            df=raw if tf=="1h" else resample_4h(raw)
            if len(df)<230:
                block_diag.append({"symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,"status":"INSUFFICIENT_BARS","bars":len(df)})
                continue
            f=compute_features(df)
            refs=reference_rows(item,tf,df,f)
            ref_rows.extend(refs)
            ref_keys={r["key"] for r in refs}
            htf=htf_table(raw,tf,item["marketFamily"])
            plans=detect_a1(item,tf,df,f,htf)
            n_entered=0
            for p in plans:
                k=candidate_key(item["symbol"],tf,p.signal_time,p.direction)
                if k not in ref_keys:
                    raise RuntimeError(f"A1 plan escaped reference universe: {k}")
                rb=simulate_a1(df,p,False);rs=simulate_a1(df,p,True)
                if rb is None or rs is None:continue
                n_entered+=1
                trade_rows.append({
                  "key":k,"signalUtc":p.signal_time.isoformat(),"symbol":p.symbol,"marketFamily":p.market_family,
                  "timeframe":p.timeframe,"direction":p.direction,"setupClass":p.setup_class,
                  "volState":p.vol_state,"htfState":p.htf_state,"entry":p.entry,"stop":p.stop,
                  "riskAtr":p.risk_atr,"tp1":p.tp1,"tp2":p.tp2,"tp3":p.tp3,
                  "tp1R":rb["tp1R"],"tp2R":rb["tp2R"],"tp3R":rb["tp3R"],
                  "grossR":rb["grossR"],"netR":rb["netR"],"stressNetR":rs["netR"],
                  "mfeR":rb["mfeR"],"maeR":rb["maeR"],"exitType":rb["exitType"]
                })
            block_diag.append({"symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,
                               "status":"OK","bars":len(df),"referenceCandidates":len(refs),"a1EnteredTrades":n_entered})

    ref=pd.DataFrame(ref_rows);tr=pd.DataFrame(trade_rows)
    if ref.empty:raise RuntimeError("A1 reference universe empty")
    if tr.empty:raise RuntimeError("A1 produced no entered trades")
    dup=tr.key.duplicated().sum()
    if dup:raise RuntimeError(f"A1 duplicate trade keys: {dup}")

    amap=tr.set_index("key")["netR"].to_dict()
    ref["a1Entered"]=ref["key"].isin(amap)
    ref["a1NetR"]=ref["key"].map(amap).fillna(0.0).astype(float)

    net=tr.netR.to_numpy(float);stress=tr.stressNetR.to_numpy(float)
    good=ref.auditGoodOpportunity.astype(bool);captured=good & ref.a1Entered.astype(bool)
    fam=tr.groupby("marketFamily").netR.mean();blk=tr.groupby(["symbol","timeframe"]).netR.mean()
    fam_total=tr.groupby("marketFamily").netR.sum();pos=fam_total[fam_total>0]
    conc=float(pos.max()/pos.sum()) if len(pos) else 0.0

    s=tr.sort_values("signalUtc").reset_index(drop=True);half=len(s)//2
    first=float(s.iloc[:half].netR.mean()) if half else 0.0
    second=float(s.iloc[half:].netR.mean()) if len(s)-half else 0.0
    tr["month"]=pd.to_datetime(tr.signalUtc,utc=True).dt.strftime("%Y-%m")
    months=tr.groupby("month").netR.sum()
    pos_month=float((months>0).mean()) if len(months) else 0.0

    base_enter=ref[ref.baselineEntered.astype(bool)]
    metrics={
      "referenceCandidateCount":int(len(ref)),
      "referenceGoodOpportunityCount":int(good.sum()),
      "enteredTrades":int(len(tr)),
      "netExpectancyR":float(net.mean()),
      "stressNetExpectancyR":float(stress.mean()),
      "profitFactor":pf(net),
      "hitRate":float(np.mean(net>0)),
      "meanMfeR":float(tr.mfeR.mean()),"meanMaeR":float(tr.maeR.mean()),
      "maxDrawdownR":max_drawdown_r(s.netR.to_numpy(float)),
      "referenceOpportunityRecall":float(captured.sum()/max(1,good.sum())),
      "referenceOpportunityPrecision":float(captured.sum()/max(1,len(tr))),
      "activeMarketFamilies":int(len(fam)),
      "activeBlocks":int(len(blk)),
      "positiveMarketFamilyFraction":float((fam>0).mean()) if len(fam) else 0.0,
      "positiveBlockFraction":float((blk>0).mean()) if len(blk) else 0.0,
      "firstHalfNetExpectancyR":first,"secondHalfNetExpectancyR":second,
      "positiveMonthFraction":pos_month,
      "singleMarketFamilyPositiveContribution":conc,
      "familyExpectancyR":{str(k):float(v) for k,v in fam.items()},
      "blockExpectancyR":{f"{k[0]}|{k[1]}":float(v) for k,v in blk.items()},
      "setupExpectancyR":{str(k):float(v) for k,v in tr.groupby("setupClass").netR.mean().items()},
      "baselineOnFreshUniverse":{
        "enteredTrades":int(len(base_enter)),
        "netExpectancyR":float(base_enter.baselineNetR.mean()) if len(base_enter) else 0.0,
        "candidateUtilityMeanR":float(ref.baselineNetR.mean()),
        "a1CandidateUtilityMeanR":float(ref.a1NetR.mean()),
        "candidateUtilityDeltaR":float((ref.a1NetR-ref.baselineNetR).mean())
      }
    }
    boot=bootstrap(ref)
    g=LOCK["developmentGate"]
    checks={
      "minimumEnteredTrades":metrics["enteredTrades"]>=int(g["minimumEnteredTrades"]),
      "minimumActiveMarketFamilies":metrics["activeMarketFamilies"]>=int(g["minimumActiveMarketFamilies"]),
      "minimumActiveBlocks":metrics["activeBlocks"]>=int(g["minimumActiveBlocks"]),
      "netExpectancyRMin":metrics["netExpectancyR"]>=float(g["netExpectancyRMin"]),
      "stressNetExpectancyRMin":metrics["stressNetExpectancyR"]>=float(g["stressNetExpectancyRMin"]),
      "profitFactorMin":metrics["profitFactor"]>=float(g["profitFactorMin"]),
      "referenceOpportunityRecallMin":metrics["referenceOpportunityRecall"]>=float(g["referenceOpportunityRecallMin"]),
      "referenceOpportunityPrecisionMin":metrics["referenceOpportunityPrecision"]>=float(g["referenceOpportunityPrecisionMin"]),
      "positiveMarketFamilyFractionMin":metrics["positiveMarketFamilyFraction"]>=float(g["positiveMarketFamilyFractionMin"]),
      "positiveBlockFractionMin":metrics["positiveBlockFraction"]>=float(g["positiveBlockFractionMin"]),
      "firstHalfNetExpectancyPositive":metrics["firstHalfNetExpectancyR"]>0,
      "secondHalfNetExpectancyPositive":metrics["secondHalfNetExpectancyR"]>0,
      "positiveMonthFractionMin":metrics["positiveMonthFraction"]>=float(g["positiveMonthFractionMin"]),
      "bootstrapProbabilityPositiveNetExpectancyMin":boot["probabilityPositiveNetExpectancy"]>=float(g["bootstrapProbabilityPositiveNetExpectancyMin"]),
      "bootstrapProbabilityPositiveUtilityDeltaVsFrozenE1BaselineMin":boot["probabilityPositiveUtilityDeltaVsFrozenE1Baseline"]>=float(g["bootstrapProbabilityPositiveUtilityDeltaVsFrozenE1BaselineMin"]),
      "singleMarketFamilyPositiveContributionMax":metrics["singleMarketFamilyPositiveContribution"]<=float(g["singleMarketFamilyPositiveContributionMax"])
    }
    checks={k:bool(v) for k,v in checks.items()}
    passed=all(checks.values())
    status="PASS_A1_FRESH_INSTRUMENT_GATE_REPLICATION_REQUIRED" if passed else "REJECT_A1_FRESH_INSTRUMENT_GATE"
    close={
      "schema":"mgpt_ite1_a1_development_closeout_v1","generatedAtUtc":datetime.now(timezone.utc).isoformat(),
      "status":status,"productionAuthority":False,"r15MutationAllowed":False,
      "timeValidationOpened":False,"sealedHoldoutOpened":False,
      "providerReplicationRequiredBeforeNextPhase":bool(passed),
      "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
      "implementationSpecSha256":hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
      "dataFreezeSha256":hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest(),
      "metrics":metrics,"bootstrap":boot,"checks":checks,
      "failedChecks":[k for k,v in checks.items() if not v],
      "blockDiagnostics":block_diag,
      "nextAction":(
        "Run independent provider/execution replication of the exact frozen A1 architecture. Do not open time validation until replication passes."
        if passed else
        "Do not rescue A1 thresholds or open time validation/holdout. Register rejection, diagnose only on contaminated A1 evidence, and keep subsequent hypotheses under a new prelock and fresh evidence budget."
      )
    }
    cp=OUT/"MGPT_ITE1_A1_DEVELOPMENT_CLOSEOUT_20260916.json"
    tp=OUT/"MGPT_ITE1_A1_TRADE_LEDGER_20260916.csv"
    rp=OUT/"MGPT_ITE1_A1_REFERENCE_LEDGER_20260916.csv"
    cp.write_text(json.dumps(close,indent=2,sort_keys=True)+"\n");tr.to_csv(tp,index=False);ref.to_csv(rp,index=False)
    sums=[]
    for p in [cp,tp,rp]:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/"MGPT_ITE1_A1_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps({"status":status,"failedChecks":close["failedChecks"],"metrics":{k:v for k,v in metrics.items() if k not in {"familyExpectancyR","blockExpectancyR","setupExpectancyR"}},"bootstrap":boot},indent=2))

if __name__=="__main__":
    main()
