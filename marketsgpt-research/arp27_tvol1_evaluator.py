#!/usr/bin/env python3
from __future__ import annotations
import json, math, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parent
LOCK=json.loads((ROOT/"ARP27_TVOL1_TURNOVER_VOLATILITY_LOCK_20260915.json").read_text())
DAY=86400000
WEEK=7*DAY

def mean(a): return float(np.mean(a)) if len(a) else 0.0
def pf(a):
    a=np.asarray(a,float); p=float(a[a>0].sum()); n=float(-a[a<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)
def maxdd(a):
    e=1.0; peak=1.0; dd=0.0
    for r in a:
        e=max(1e-12,e*(1+float(r))); peak=max(peak,e); dd=max(dd,1-e/peak)
    return dd
def boot(a,seed,iterations=10000,block=4):
    a=np.asarray(a,float); n=len(a)
    if not n:return {"probPositive":0.0,"ci95Low":0.0,"ci95High":0.0}
    rng=np.random.default_rng(seed); z=np.empty(iterations)
    for k in range(iterations):
        vals=[]
        while len(vals)<n:
            st=int(rng.integers(0,n)); take=min(block,n-len(vals))
            vals.extend(a[(st+np.arange(take))%n])
        z[k]=float(np.mean(vals))
    z.sort()
    return {"probPositive":float(np.mean(z>0)),"ci95Low":float(z[int(.025*(iterations-1))]),"ci95High":float(z[int(.975*(iterations-1))])}
def stats(a,dates,seed):
    a=np.asarray(a,float); m=float(a.mean()) if len(a) else 0.0
    sd=float(a.std(ddof=1)) if len(a)>1 else 0.0
    months={}
    for r,d in zip(a,dates):
        k=str(d)[:7]; months[k]=months.get(k,0.0)+float(r)
    return {
      "n":len(a),"meanWeeklyReturn":m,"annualizedMean":m*52,
      "annualizedSharpe":float(m/sd*math.sqrt(52)) if sd>0 else 0.0,
      "profitFactor":pf(a),"maxDrawdown":maxdd(a),
      "positiveMonthFraction":sum(x>0 for x in months.values())/max(1,len(months)),
      "monthSums":months,"bootstrap":boot(a,seed)
    }

def parse_panel(p):
    # Required JSON:
    # {"symbols":{"BTCUSDT":{"daily":[{"t":..., "marketCap":..., "volumeUsd":..., "binanceOpen":...}, ...]}, ...}}
    data={}
    for sym,obj in p.get("symbols",{}).items():
        m={}
        for r in obj.get("daily",[]):
            try:
                t=int(r["t"]); mc=float(r["marketCap"]); vol=float(r["volumeUsd"]); op=float(r["binanceOpen"])
            except Exception: continue
            if mc>0 and vol>=0 and op>0 and all(map(math.isfinite,[mc,vol,op])):
                m[t]={"marketCap":mc,"volumeUsd":vol,"open":op}
        data[sym]=m
    return data

def monday_grid(start_ms,end_ms):
    # lock dates are Monday 00:00 UTC.
    return list(range(start_ms,end_ms,WEEK))

def evaluate(payload):
    d=parse_panel(payload)
    syms=LOCK["cohort"]["symbols"]
    start=int(datetime.fromisoformat(LOCK["windows"]["development"]["start"].replace("Z","+00:00")).timestamp()*1000)
    end=int(datetime.fromisoformat(LOCK["windows"]["development"]["endExclusive"].replace("Z","+00:00")).timestamp()*1000)
    prev=np.zeros(len(syms)); gross=[]; turns=[]; dates=[]; elig_counts=[]; contrib=np.zeros(len(syms))
    for t in monday_grid(start,end):
        cand=[]
        for i,s in enumerate(syms):
            sm=d.get(s,{})
            obs=[]
            ok=True
            # 30 strictly completed daily snapshots: t-30d ... t-1d.
            for k in range(30,0,-1):
                r=sm.get(t-k*DAY)
                if r is None: ok=False; break
                obs.append(r["volumeUsd"]/r["marketCap"])
            r0=sm.get(t); r1=sm.get(t+WEEK)
            if not ok or r0 is None or r1 is None: continue
            tv=float(np.std(np.asarray(obs,float),ddof=1))
            ret=r1["open"]/r0["open"]-1.0
            if math.isfinite(tv) and math.isfinite(ret):
                cand.append((i,s,tv,ret))
        elig_counts.append(len(cand))
        w=np.zeros(len(syms))
        if len(cand)>=LOCK["cohort"]["minimumEligibleAtRebalance"]:
            cand.sort(key=lambda x:(x[2],x[1]))
            q=len(cand)//4
            if q>=1:
                for i,_,_,_ in cand[:q]:w[i]=0.5/q
                for i,_,_,_ in cand[-q:]:w[i]=-0.5/q
        r=0.0
        for i,_,_,ret in cand:
            c=w[i]*ret; r+=c; contrib[i]+=c
        gross.append(r); turns.append(float(np.abs(w-prev).sum()))
        dates.append(datetime.fromtimestamp(t/1000,timezone.utc).date().isoformat())
        prev=w
    if turns:turns[-1]+=float(np.abs(prev).sum())
    gross=np.asarray(gross); turns=np.asarray(turns)
    base=gross-turns*LOCK["costs"]["baseOneWayRate"]
    s2=gross-turns*LOCK["costs"]["stress2xOneWayRate"]
    s4=gross-turns*LOCK["costs"]["stress4xOneWayRate"]
    bs=stats(base,dates,27101); gs=stats(gross,dates,27104); s2s=stats(s2,dates,27102); s4s=stats(s4,dates,27103)
    n=len(base); half=n//2
    first=float(base[:half].sum()) if half else 0.0; second=float(base[half:].sum()) if n-half else 0.0
    pos=contrib[contrib>0]; dom=float(pos.max()/pos.sum()) if len(pos) and pos.sum()>0 else 0.0
    coverage=sum(x>=LOCK["cohort"]["minimumEligibleAtRebalance"] for x in elig_counts)/max(1,len(elig_counts))
    avg=float(np.mean(elig_counts)) if elig_counts else 0.0
    g=LOCK["developmentGate"]
    checks={
      "minimumEvaluatedWeeks":n>=g["minimumEvaluatedWeeks"],
      "minimumCoverageFraction":coverage>=g["minimumCoverageFraction"],
      "averageEligibleSymbolsMin":avg>=g["averageEligibleSymbolsMin"],
      "netMeanWeeklyReturnMin":bs["meanWeeklyReturn"]>=g["netMeanWeeklyReturnMin"],
      "netAnnualizedSharpeMin":bs["annualizedSharpe"]>=g["netAnnualizedSharpeMin"],
      "netProfitFactorMin":bs["profitFactor"]>=g["netProfitFactorMin"],
      "stress2xMeanWeeklyReturnMin":s2s["meanWeeklyReturn"]>=g["stress2xMeanWeeklyReturnMin"],
      "stress4xMeanWeeklyReturnMin":s4s["meanWeeklyReturn"]>=g["stress4xMeanWeeklyReturnMin"],
      "positiveMonthFractionMin":bs["positiveMonthFraction"]>=g["positiveMonthFractionMin"],
      "firstHalfNetReturnPositive":first>0,
      "secondHalfNetReturnPositive":second>0,
      "bootstrapProbabilityPositiveMin":bs["bootstrap"]["probPositive"]>=g["bootstrapProbabilityPositiveMin"],
      "bootstrapCi95LowMin":bs["bootstrap"]["ci95Low"]>=g["bootstrapCi95LowMin"],
      "maxDrawdownMax":bs["maxDrawdown"]<=g["maxDrawdownMax"],
      "singleSymbolPositiveContributionMax":dom<=g["singleSymbolPositiveContributionMax"]
    }
    return {
      "schema":"mgpt_arp27_tvol1_development_result_v1","generation":"ARP27-TVOL1",
      "status":"PASS_DEVELOPMENT_GATE" if all(checks.values()) else "REJECT_DEVELOPMENT_GATE",
      "productionAuthority":False,"r15MutationAllowed":False,
      "development":{"evaluatedWeeks":n,"coverageFraction":coverage,"averageEligibleSymbols":avg,
        "meanTurnover":float(turns.mean()) if len(turns) else 0.0,
        "firstHalfNetReturn":first,"secondHalfNetReturn":second,
        "singleSymbolPositiveContribution":dom,"gross":gs,"baseCost":bs,"stress2x":s2s,"stress4x":s4s,
        "symbolGrossContributions":{s:float(contrib[i]) for i,s in enumerate(syms)}},
      "gate":checks,"failedChecks":[k for k,v in checks.items() if not v],
      "validationAuthorized":all(checks.values()),"sealedHoldoutOpened":False
    }

if __name__=="__main__":
    if len(sys.argv)!=2: raise SystemExit("usage: arp27_tvol1_evaluator.py NORMALIZED_PANEL.json")
    p=json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(evaluate(p),indent=2))