#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, io, json, math, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "ARP30_XCR1_BTC_LEADLAG_LOCK_20260916.json"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOCK = json.loads(LOCK_PATH.read_text())
PRED = LOCK["universe"]["predictor"]
TARGETS = LOCK["universe"]["targets"]
SYMS = [PRED] + TARGETS
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
FETCH_START = datetime.fromisoformat(LOCK["windows"]["archiveFetchStart"].replace("Z","+00:00"))
DEV_START = datetime.fromisoformat(LOCK["windows"]["development"]["start"].replace("Z","+00:00"))
DEV_END = datetime.fromisoformat(LOCK["windows"]["development"]["endExclusive"].replace("Z","+00:00"))
TRAIN = int(LOCK["methodAuthority"]["trainingWindowMinutes"])
CADENCE = int(LOCK["methodAuthority"]["decisionCadenceMinutes"])
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"MarketsGPT-ARP30-XCR1/1.0"})

def month_keys(a: datetime, b: datetime):
    y,m=a.year,a.month
    out=[]
    while (y,m) <= (b.year,b.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out

MONTHS = month_keys(FETCH_START, DEV_END + timedelta(days=1))

def get_bytes(url: str, tries: int = 5):
    last = None
    for k in range(tries):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(8.0, 0.75 * (2**k)))
    raise RuntimeError(f"download failed {url}: {last}")

def download_month(sym: str, ym: str):
    name = f"{sym}-1m-{ym}.zip"
    url = f"{BASE}/{sym}/1m/{name}"
    blob = get_bytes(url)
    if blob is None:
        return {"symbol":sym,"ym":ym,"status":"404","rows":[]}
    chk = get_bytes(url + ".CHECKSUM")
    if chk is None:
        raise RuntimeError(f"missing CHECKSUM {url}")
    expected = chk.decode("utf-8","replace").strip().split()[0].lower()
    got = hashlib.sha256(blob).hexdigest()
    if expected != got:
        raise RuntimeError(f"checksum mismatch {name}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise RuntimeError(f"unexpected archive members {name}: {names}")
        raw = z.read(names[0]).decode("utf-8-sig","replace")
    rows=[]
    for rec in csv.reader(io.StringIO(raw)):
        if not rec:
            continue
        try:
            t=int(float(rec[0]))
        except Exception:
            continue
        if t > 10**14:
            t //= 1000
        if len(rec) < 8:
            continue
        try:
            o=float(rec[1]); c=float(rec[4]); qv=float(rec[7])
        except Exception:
            continue
        if o > 0 and c > 0 and qv >= 0 and all(map(math.isfinite,[o,c,qv])):
            rows.append((t,o,c,qv))
    return {"symbol":sym,"ym":ym,"status":"ok","sha256":got,"rows":rows}

def transport():
    jobs=[]
    with ThreadPoolExecutor(max_workers=20) as ex:
        for s in SYMS:
            for ym in MONTHS:
                jobs.append(ex.submit(download_month,s,ym))
        got=[f.result() for f in as_completed(jobs)]
    data={s:{} for s in SYMS}
    duplicates={s:0 for s in SYMS}
    manifest=[]
    for x in got:
        manifest.append({k:v for k,v in x.items() if k!="rows"})
        for row in x["rows"]:
            if row[0] in data[x["symbol"]]:
                duplicates[x["symbol"]] += 1
            data[x["symbol"]][row[0]] = row[1:]
    frames={}
    diagnostics=[]
    for s in SYMS:
        items=sorted(data[s].items())
        if items:
            idx=pd.to_datetime([t for t,_ in items],unit="ms",utc=True)
            arr=np.asarray([v for _,v in items],float)
            df=pd.DataFrame(arr,index=idx,columns=["open","close","qv"])
            df=df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df=pd.DataFrame(columns=["open","close","qv"])
        frames[s]=df
        ts=sorted(data[s])
        gaps=sum(1 for a,b in zip(ts,ts[1:]) if b-a != 60000)
        diagnostics.append({
            "symbol":s,"rows":len(ts),
            "firstUtc":datetime.fromtimestamp(ts[0]/1000,timezone.utc).isoformat() if ts else None,
            "lastUtc":datetime.fromtimestamp(ts[-1]/1000,timezone.utc).isoformat() if ts else None,
            "duplicateRows":duplicates[s],"postFirstMinuteGapCount":gaps
        })
    return frames, manifest, diagnostics

def fit_ols(x: np.ndarray, y: np.ndarray):
    mask=np.isfinite(x) & np.isfinite(y)
    n=int(mask.sum())
    if n < LOCK["universe"]["minimumTrainingPairsPerTarget"]:
        return None
    xx=x[mask]; yy=y[mask]
    xm=float(xx.mean()); ym=float(yy.mean())
    vx=float(np.sum((xx-xm)**2))
    if vx <= 1e-18:
        return None
    beta=float(np.sum((xx-xm)*(yy-ym))/vx)
    alpha=float(ym-beta*xm)
    return alpha,beta,n

def pf(a):
    a=np.asarray(a,float)
    pos=float(a[a>0].sum()); neg=float(-a[a<0].sum())
    return pos/neg if neg>0 else (999.0 if pos>0 else 0.0)

def maxdd(a):
    eq=1.0; peak=1.0; dd=0.0
    for r in np.asarray(a,float):
        eq=max(1e-12,eq*(1.0+float(r)))
        peak=max(peak,eq)
        dd=max(dd,1.0-eq/peak)
    return dd

def bootstrap(a, seed, it=10000, block=5):
    a=np.asarray(a,float)
    n=len(a)
    if n==0:
        return {"probPositive":0.0,"ci95Low":0.0,"ci95High":0.0}
    rng=np.random.default_rng(seed)
    means=np.empty(it,float)
    for k in range(it):
        vals=[]
        while len(vals)<n:
            st=int(rng.integers(0,n))
            take=min(block,n-len(vals))
            vals.extend(a[(st+np.arange(take))%n])
        means[k]=float(np.mean(vals))
    return {
        "probPositive":float(np.mean(means>0)),
        "ci95Low":float(np.quantile(means,0.025)),
        "ci95High":float(np.quantile(means,0.975))
    }

def daily_stats(slot_returns, slot_times, seed):
    byday={}
    for r,t in zip(slot_returns,slot_times):
        day=t.strftime("%Y-%m-%d")
        byday.setdefault(day,[]).append(float(r))
    days=sorted(byday)
    d=np.asarray([float(np.prod(1.0+np.asarray(byday[k],float))-1.0) for k in days],float)
    m=float(d.mean()) if len(d) else 0.0
    sd=float(d.std(ddof=1)) if len(d)>1 else 0.0
    return {
        "nDays":int(len(d)),
        "meanDailyReturn":m,
        "annualizedMean":m*365.0,
        "annualizedSharpe":float(m/sd*math.sqrt(365.0)) if sd>0 else 0.0,
        "profitFactor":pf(d),
        "maxDrawdown":maxdd(d),
        "positiveDayFraction":float(np.mean(d>0)) if len(d) else 0.0,
        "bootstrap":bootstrap(d,seed),
        "dailyReturns":{k:float(v) for k,v in zip(days,d)}
    }

def main():
    frames, manifest, diagnostics = transport()
    if PRED not in frames or frames[PRED].empty:
        raise RuntimeError("predictor transport unavailable")
    idx=frames[PRED].index
    panel_close=pd.DataFrame(index=idx)
    panel_open=pd.DataFrame(index=idx)
    panel_liq=pd.DataFrame(index=idx)
    for s in SYMS:
        df=frames[s].reindex(idx)
        panel_close[s]=df["close"]
        panel_open[s]=df["open"]
        panel_liq[s]=df["qv"].rolling(60,min_periods=60).mean()
    rets=panel_close.pct_change(fill_method=None)
    btc=rets[PRED].to_numpy(float)
    target_ret=rets[TARGETS].to_numpy(float)
    opens=panel_open[TARGETS].to_numpy(float)
    liq=panel_liq[TARGETS].to_numpy(float)

    start=pd.Timestamp(DEV_START); end=pd.Timestamp(DEV_END)
    positions=np.flatnonzero((idx>=start-pd.Timedelta(minutes=1)) & (idx<end))
    decisions=[]
    for p in positions:
        ts=idx[p]
        entry_ts=ts+pd.Timedelta(minutes=1)
        exit_ts=entry_ts+pd.Timedelta(minutes=CADENCE)
        if not (start <= entry_ts < end and exit_ts <= end):
            continue
        if entry_ts.minute % CADENCE != 0:
            continue
        if p < TRAIN+1 or p+CADENCE+1 >= len(idx):
            continue
        decisions.append(p)

    gross=[]; base=[]; stress=[]; extreme=[]; times=[]
    eligible_counts=[]; positive_beta_fracs=[]; ledgers=[]
    prev_w=np.zeros(len(TARGETS),float)
    contrib=np.zeros(len(TARGETS),float)
    expected_slots=len(pd.date_range(start=start,end=end-pd.Timedelta(minutes=CADENCE),freq=f"{CADENCE}min"))

    for p in decisions:
        xtrain=btc[p-TRAIN:p]
        ytrain=target_ret[p-TRAIN+1:p+1,:]
        xnow=btc[p]
        if not math.isfinite(float(xnow)):
            continue
        forecasts=np.full(len(TARGETS),np.nan)
        betas=np.full(len(TARGETS),np.nan)
        for j in range(len(TARGETS)):
            fit=fit_ols(xtrain,ytrain[:,j])
            if fit is None:
                continue
            a,b,_=fit
            forecasts[j]=a+b*xnow
            betas[j]=b
        entry=opens[p+1,:]
        exitp=opens[p+1+CADENCE,:]
        liquid=liq[p,:] >= float(LOCK["universe"]["prior60mMeanQuoteVolumeUsdMin"])
        eligible=np.isfinite(forecasts)&np.isfinite(entry)&np.isfinite(exitp)&(entry>0)&(exitp>0)&liquid
        ids=np.flatnonzero(eligible)
        if len(ids) < int(LOCK["universe"]["minimumEligibleTargetsPerDecision"]):
            continue
        q=max(1,len(ids)//5)
        ordered=ids[np.argsort(forecasts[ids],kind="mergesort")]
        short_ids=ordered[:q]
        long_ids=ordered[-q:]
        w=np.zeros(len(TARGETS),float)
        w[long_ids]=0.5/q
        w[short_ids]=-0.5/q
        asset_r=exitp/entry-1.0
        g=float(np.nansum(w*asset_r))
        turnover=float(np.sum(np.abs(w-prev_w)))
        bret=g-float(LOCK["costs"]["baseOneWayRate"])*turnover
        sret=g-float(LOCK["costs"]["stressOneWayRate"])*turnover
        eret=g-float(LOCK["costs"]["extremeOneWayRate"])*turnover
        gross.append(g); base.append(bret); stress.append(sret); extreme.append(eret); times.append(idx[p+1])
        eligible_counts.append(int(len(ids)))
        positive_beta_fracs.append(float(np.mean(betas[ids]>0)))
        contrib += np.nan_to_num(w*asset_r,nan=0.0)
        ledgers.append({
            "decisionUtc":idx[p].isoformat(),"entryUtc":idx[p+1].isoformat(),"exitUtc":idx[p+1+CADENCE].isoformat(),
            "eligible":int(len(ids)),"positiveBetaFraction":positive_beta_fracs[-1],
            "long":[TARGETS[j] for j in long_ids],"short":[TARGETS[j] for j in short_ids],
            "grossReturn":g,"turnover":turnover,"baseNetReturn":bret
        })
        prev_w=w

    if base:
        terminal=float(np.sum(np.abs(prev_w)))
        base[-1]-=float(LOCK["costs"]["baseOneWayRate"])*terminal
        stress[-1]-=float(LOCK["costs"]["stressOneWayRate"])*terminal
        extreme[-1]-=float(LOCK["costs"]["extremeOneWayRate"])*terminal
        ledgers[-1]["terminalLiquidationTurnover"]=terminal

    seed=int(LOCK["statistics"]["bootstrap"]["developmentSeed"])
    st_g=daily_stats(gross,times,seed)
    st_b=daily_stats(base,times,seed)
    st_s=daily_stats(stress,times,seed)
    st_e=daily_stats(extreme,times,seed)
    nslots=len(base)
    coverage=nslots/max(1,expected_slots)
    dvals=np.asarray(list(st_b["dailyReturns"].values()),float)
    half=max(1,len(dvals)//2)
    first=float(np.prod(1+dvals[:half])-1) if len(dvals) else 0.0
    second=float(np.prod(1+dvals[half:])-1) if len(dvals)>half else 0.0
    poscon=np.maximum(contrib,0)
    dominance=float(poscon.max()/poscon.sum()) if poscon.sum()>0 else 0.0
    avg_beta=float(np.mean(positive_beta_fracs)) if positive_beta_fracs else 0.0
    avg_elig=float(np.mean(eligible_counts)) if eligible_counts else 0.0

    gate=LOCK["developmentGate"]
    checks={
      "minimumEvaluatedDays":st_b["nDays"]>=gate["minimumEvaluatedDays"],
      "minimumDecisionCoverageFraction":coverage>=gate["minimumDecisionCoverageFraction"],
      "averageEligibleTargetsMin":avg_elig>=gate["averageEligibleTargetsMin"],
      "baseNetMeanDailyReturnMin":st_b["meanDailyReturn"]>=gate["baseNetMeanDailyReturnMin"],
      "baseNetAnnualizedSharpeMin":st_b["annualizedSharpe"]>=gate["baseNetAnnualizedSharpeMin"],
      "baseNetProfitFactorMin":st_b["profitFactor"]>=gate["baseNetProfitFactorMin"],
      "stressNetMeanDailyReturnMin":st_s["meanDailyReturn"]>=gate["stressNetMeanDailyReturnMin"],
      "extremeNetMeanDailyReturnMin":st_e["meanDailyReturn"]>=gate["extremeNetMeanDailyReturnMin"],
      "positiveDayFractionMin":st_b["positiveDayFraction"]>=gate["positiveDayFractionMin"],
      "firstHalfNetReturnPositive":first>0,
      "secondHalfNetReturnPositive":second>0,
      "bootstrapProbabilityPositiveMin":st_b["bootstrap"]["probPositive"]>=gate["bootstrapProbabilityPositiveMin"],
      "bootstrapCi95LowMin":st_b["bootstrap"]["ci95Low"]>=gate["bootstrapCi95LowMin"],
      "maxDrawdownMax":st_b["maxDrawdown"]<=gate["maxDrawdownMax"],
      "averagePositiveBetaFractionMin":avg_beta>=gate["averagePositiveBetaFractionMin"],
      "singleSymbolPositiveContributionMax":dominance<=gate["singleSymbolPositiveContributionMax"]
    }
    failed=[k for k,v in checks.items() if not v]
    status="PASS_DEVELOPMENT_GATE_FUNDING_REPLAY_REQUIRED" if not failed else "REJECT_DEVELOPMENT_GATE"
    result={
      "schema":"mgpt_arp30_xcr1_development_closeout_v1",
      "generation":"ARP30-XCR1",
      "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
      "status":status,
      "productionAuthority":False,"r15MutationAllowed":False,
      "validationAuthorized":False,
      "fundingReplayRequiredBeforeValidation":False if failed else True,
      "sealedHoldoutOpened":False,
      "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
      "development":{
        "expectedDecisionSlots":expected_slots,"evaluatedDecisionSlots":nslots,
        "decisionCoverageFraction":coverage,"averageEligibleTargets":avg_elig,
        "averagePositiveBetaFraction":avg_beta,"firstHalfNetReturn":first,"secondHalfNetReturn":second,
        "singleSymbolPositiveContribution":dominance,
        "gross":st_g,"baseCost":st_b,"stress":st_s,"extreme":st_e,
        "symbolGrossContributions":{s:float(v) for s,v in zip(TARGETS,contrib)}
      },
      "checks":checks,"failedChecks":failed,
      "nextAction":("Run exact funding-cashflow replay on the frozen decision ledger; do not open validation yet."
                    if not failed else
                    "Close ARP30-XCR1; no cadence, universe, training-window, direction, cost, liquidity, estimator, or gate rescue; validation and holdout remain sealed.")
    }
    transport_out={
      "schema":"mgpt_arp30_xcr1_transport_diagnostics_v1",
      "generation":"ARP30-XCR1","months":MONTHS,"manifest":manifest,"symbols":diagnostics
    }
    (OUT/"ARP30_XCR1_TRANSPORT_DIAGNOSTICS_20260916.json").write_text(json.dumps(transport_out,indent=2))
    (OUT/"ARP30_XCR1_DECISION_LEDGER_20260916.json").write_text(json.dumps({"schema":"mgpt_arp30_xcr1_decision_ledger_v1","rows":ledgers},indent=2))
    (OUT/"ARP30_XCR1_DEVELOPMENT_CLOSEOUT_20260916.json").write_text(json.dumps(result,indent=2))
    sums=[]
    for p in sorted(OUT.glob("ARP30_XCR1_*_20260916.json")):
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/"ARP30_XCR1_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps({"status":status,"failedChecks":failed,"development":result["development"]},indent=2))

if __name__=="__main__":
    main()
