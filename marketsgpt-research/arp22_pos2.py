#!/usr/bin/env python3
from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "ARP22_POS2_HISTORICAL_POSITIONING_REPLICATION_LOCK_20260915.json"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOCK = json.loads(LOCK_PATH.read_text())
SYMS = LOCK["cohort"]["symbols"]
BASE = "https://data.binance.vision/data/futures/um"
DEV_START = datetime.fromisoformat(LOCK["windows"]["development"]["start"].replace("Z", "+00:00"))
DEV_END = datetime.fromisoformat(LOCK["windows"]["development"]["endExclusive"].replace("Z", "+00:00"))
WARMUP_START = datetime.fromisoformat(LOCK["windows"]["warmupStart"].replace("Z", "+00:00"))
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MarketsGPT-ARP22-POS2/1.0"})

def dt_ms(x): return int(x.timestamp()*1000)
def date_range(a,b):
    d=a
    while d<b:
        yield d
        d+=timedelta(days=1)
def month_range(a,b):
    y,m=a.year,a.month; out=[]
    while (y,m)<=(b.year,b.month):
        out.append(f"{y:04d}-{m:02d}"); m+=1
        if m==13: y+=1; m=1
    return out
def get_bytes(url,tries=5):
    last=None
    for k in range(tries):
        try:
            r=SESSION.get(url,timeout=45)
            if r.status_code==200:return r.content
            if r.status_code==404:return None
            last=f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:last=repr(e)
        time.sleep(min(8,.75*(2**k)))
    raise RuntimeError(f"download failed {url}: {last}")
def verified_zip(url):
    blob=get_bytes(url)
    if blob is None:return None,None
    chk=get_bytes(url+".CHECKSUM")
    if chk is None:raise RuntimeError(f"missing checksum: {url}")
    expected=chk.decode("utf-8","replace").strip().split()[0].lower()
    actual=hashlib.sha256(blob).hexdigest()
    if expected!=actual:raise RuntimeError(f"checksum mismatch: {url}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=[n for n in z.namelist() if not n.endswith("/")]
        if len(names)!=1:raise RuntimeError(f"unexpected zip members: {url}: {names}")
        raw=z.read(names[0]).decode("utf-8-sig","replace")
    return raw,actual
def parse_metric_time(s):
    s=str(s).strip()
    try:
        n=int(float(s))
        if n>10**14:n//=1000
        if n>10**11:return n
    except Exception:pass
    dt=datetime.fromisoformat(s.replace("Z","+00:00"))
    if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
    return dt_ms(dt.astimezone(timezone.utc))
def download_metric(sym,d):
    stamp=d.isoformat(); name=f"{sym}-metrics-{stamp}.zip"; url=f"{BASE}/daily/metrics/{sym}/{name}"
    raw,sha=verified_zip(url)
    if raw is None:return {"kind":"metrics","symbol":sym,"period":stamp,"status":"404","sha256":None,"rows":[]}
    rows=[]
    for rec in csv.reader(io.StringIO(raw)):
        if not rec or str(rec[0]).strip().lower()=="create_time" or len(rec)<5:continue
        try:t=parse_metric_time(rec[0]); ratio=float(rec[4])
        except Exception:continue
        if ratio>0 and math.isfinite(ratio):rows.append((t,ratio))
    return {"kind":"metrics","symbol":sym,"period":stamp,"status":"ok","sha256":sha,"rows":rows}
def download_kline_month(sym,ym):
    name=f"{sym}-1d-{ym}.zip"; url=f"{BASE}/monthly/klines/{sym}/1d/{name}"
    raw,sha=verified_zip(url)
    if raw is None:return {"kind":"klines","symbol":sym,"period":ym,"status":"404","sha256":None,"rows":[]}
    rows=[]
    for rec in csv.reader(io.StringIO(raw)):
        if not rec or len(rec)<8:continue
        try:t=int(float(rec[0])); op=float(rec[1]); qv=float(rec[7])
        except Exception:continue
        if t>10**14:t//=1000
        if op>0 and qv>=0 and math.isfinite(op) and math.isfinite(qv):rows.append((t,op,qv))
    return {"kind":"klines","symbol":sym,"period":ym,"status":"ok","sha256":sha,"rows":rows}
def transport():
    metric_start=(DEV_START-timedelta(days=2)).date(); metric_end=(DEV_END+timedelta(days=1)).date()
    months=month_range(WARMUP_START,DEV_END); jobs=[]
    with ThreadPoolExecutor(max_workers=48) as ex:
        for sym in SYMS:
            for d in date_range(metric_start,metric_end):jobs.append(ex.submit(download_metric,sym,d))
            for ym in months:jobs.append(ex.submit(download_kline_month,sym,ym))
        got=[f.result() for f in as_completed(jobs)]
    metrics={s:{} for s in SYMS}; prices={s:{} for s in SYMS}; manifest=[]; dm={s:0 for s in SYMS}; dp={s:0 for s in SYMS}
    for x in got:
        manifest.append({k:v for k,v in x.items() if k!="rows"})
        if x["kind"]=="metrics":
            for t,ratio in x["rows"]:
                if t in metrics[x["symbol"]]:dm[x["symbol"]]+=1
                metrics[x["symbol"]][t]=ratio
        else:
            for t,op,qv in x["rows"]:
                if t in prices[x["symbol"]]:dp[x["symbol"]]+=1
                prices[x["symbol"]][t]=(op,qv)
    series={}; diag=[]
    for sym in SYMS:
        mts=sorted(metrics[sym]); pts=sorted(prices[sym]); series[sym]=(mts,[metrics[sym][t] for t in mts])
        diag.append({"symbol":sym,"metricSnapshots":len(mts),"dailyPriceBars":len(pts),
                     "firstMetricUtc":datetime.fromtimestamp(mts[0]/1000,timezone.utc).isoformat() if mts else None,
                     "lastMetricUtc":datetime.fromtimestamp(mts[-1]/1000,timezone.utc).isoformat() if mts else None,
                     "firstPriceUtc":datetime.fromtimestamp(pts[0]/1000,timezone.utc).isoformat() if pts else None,
                     "lastPriceUtc":datetime.fromtimestamp(pts[-1]/1000,timezone.utc).isoformat() if pts else None,
                     "duplicateMetricTimestamps":dm[sym],"duplicateDailyPriceTimestamps":dp[sym]})
    return series,prices,manifest,diag
def latest_signal(series,sym,t):
    ts,vals=series[sym]; i=bisect.bisect_left(ts,t)-1
    if i<0:return None
    age=t-ts[i]
    if age<=0 or age>15*60*1000:return None
    return math.log(max(vals[i],1e-8))
def trailing_liquidity(prices,sym,t):
    vals=[]
    for k in range(1,91):
        row=prices[sym].get(t-k*86400000)
        if row is None:return None
        vals.append(row[1])
    return float(np.mean(vals))
def pf(a):
    p=float(np.sum(a[a>0])); n=float(-np.sum(a[a<0]))
    return p/n if n>0 else (999.0 if p>0 else 0.0)
def max_dd(a):
    e=pk=1.; dd=0.
    for r in a:
        e=max(1e-12,e*(1+float(r))); pk=max(pk,e); dd=max(dd,1-e/pk)
    return dd
def bootstrap(a,seed,iterations=10000,block=7):
    rng=np.random.default_rng(seed); a=np.asarray(a,float); n=len(a); out=np.empty(iterations)
    for k in range(iterations):
        vals=[]
        while len(vals)<n:
            st=int(rng.integers(0,n)); take=min(block,n-len(vals)); vals.extend(a[(st+np.arange(take))%n])
        out[k]=float(np.mean(vals))
    out.sort()
    return {"probPositive":float(np.mean(out>0)),"ci95Low":float(out[int(.025*(iterations-1))]),"ci95High":float(out[int(.975*(iterations-1))])}
def stat_block(a,dates,seed):
    a=np.asarray(a,float); m=float(np.mean(a)); sd=float(np.std(a,ddof=1)) if len(a)>1 else 0.; months={}
    for r,d in zip(a,dates):months[d[:7]]=months.get(d[:7],0.)+float(r)
    return {"n":len(a),"meanDailyReturn":m,"annualizedMean":m*365,
            "annualizedSharpe":float(m/sd*math.sqrt(365)) if sd>0 else 0.,
            "profitFactor":pf(a),"maxDrawdown":max_dd(a),
            "positiveMonthFraction":sum(v>0 for v in months.values())/max(1,len(months)),
            "monthSums":months,"bootstrap":bootstrap(a,seed)}
def main():
    series,prices,manifest,diag=transport()
    start=dt_ms(DEV_START); end=dt_ms(DEV_END); expected=int((end-start)//86400000)
    prev=np.zeros(len(SYMS)); gross=[]; turnover=[]; dates=[]; elig_counts=[]; contrib=np.zeros(len(SYMS)); covered=0
    for di in range(expected):
        t=start+di*86400000; nxt=t+86400000; eligible=[]
        for si,sym in enumerate(SYMS):
            p0=prices[sym].get(t); p1=prices[sym].get(nxt)
            if p0 is None or p1 is None:continue
            liq=trailing_liquidity(prices,sym,t)
            if liq is None or liq<5_000_000:continue
            sig=latest_signal(series,sym,t)
            if sig is None:continue
            ret=p1[0]/p0[0]-1.
            if math.isfinite(ret):eligible.append((si,sym,sig,ret))
        desired=np.zeros(len(SYMS)); elig_counts.append(len(eligible))
        if len(eligible)>=LOCK["cohort"]["minimumEligiblePerDecision"]:
            covered+=1; eligible.sort(key=lambda x:(-x[2],x[1])); q=max(1,int(math.floor(len(eligible)*.10)))
            for si,_,_,_ in eligible[:q]:desired[si]+=.5/q
            for si,_,_,_ in eligible[-q:]:desired[si]-=.5/q
        r=0.; by=np.zeros(len(SYMS))
        for si,_,_,ret in eligible:
            c=desired[si]*ret; r+=c; by[si]=c
        contrib+=by; turnover.append(float(np.abs(desired-prev).sum())); gross.append(r)
        dates.append(datetime.fromtimestamp(t/1000,timezone.utc).date().isoformat()); prev=desired
    if turnover:turnover[-1]+=float(np.abs(prev).sum())
    gross=np.asarray(gross); turnover=np.asarray(turnover); costs=LOCK["costs"]
    net=gross-turnover*costs["baseOneWayRate"]; n15=gross-turnover*costs["stress1_5xOneWayRate"]; n2=gross-turnover*costs["stress2xOneWayRate"]
    base=stat_block(net,dates,22201); s15=stat_block(n15,dates,22211); s2=stat_block(n2,dates,22212); gs=stat_block(gross,dates,22213)
    pos=contrib[contrib>0]; dom=float(pos.max()/pos.sum()) if len(pos) and pos.sum()>0 else 0.
    coverage=covered/expected; avg=float(np.mean(elig_counts)); half=len(net)//2
    first=float(np.sum(net[:half])); second=float(np.sum(net[half:])); g=LOCK["developmentGate"]
    gate={
      "minimumEvaluatedDays":len(net)>=g["minimumEvaluatedDays"],"minimumCoverageFraction":coverage>=g["minimumCoverageFraction"],
      "netMeanDailyReturnMin":base["meanDailyReturn"]>=g["netMeanDailyReturnMin"],"netAnnualizedSharpeMin":base["annualizedSharpe"]>=g["netAnnualizedSharpeMin"],
      "netProfitFactorMin":base["profitFactor"]>=g["netProfitFactorMin"],"stress1_5xMeanDailyReturnMin":s15["meanDailyReturn"]>=g["stress1_5xMeanDailyReturnMin"],
      "stress2xMeanDailyReturnMin":s2["meanDailyReturn"]>=g["stress2xMeanDailyReturnMin"],"positiveMonthFractionMin":base["positiveMonthFraction"]>=g["positiveMonthFractionMin"],
      "firstHalfNetReturnPositive":first>0,"secondHalfNetReturnPositive":second>0,
      "bootstrapProbabilityPositiveMin":base["bootstrap"]["probPositive"]>=g["bootstrapProbabilityPositiveMin"],
      "bootstrapCi95LowMin":base["bootstrap"]["ci95Low"]>=g["bootstrapCi95LowMin"],"maxDrawdownMax":base["maxDrawdown"]<=g["maxDrawdownMax"],
      "averageEligibleSymbolsMin":avg>=g["averageEligibleSymbolsMin"],"singleSymbolPositiveContributionMax":dom<=g["singleSymbolPositiveContributionMax"]}
    gate={k:bool(v) for k,v in gate.items()}; failed=[k for k,v in gate.items() if not v]
    close={"schema":"mgpt_arp22_pos2_development_closeout_v1","generation":"ARP22-POS2","generatedAtUtc":datetime.now(timezone.utc).isoformat(),
           "status":"PASS_DEVELOPMENT_GATE" if not failed else "REJECT_DEVELOPMENT_GATE","productionAuthority":False,"r15MutationAllowed":False,
           "validationAuthorized":not failed,"sealedHoldoutOpened":False,"lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
           "development":{"expectedDays":expected,"evaluatedDays":len(net),"decisionCoverageDays":covered,"coverageFraction":coverage,
             "averageEligibleSymbols":avg,"meanTurnover":float(np.mean(turnover)),"singleSymbolPositiveContribution":dom,
             "firstHalfNetReturn":first,"secondHalfNetReturn":second,"gross":gs,"baseCost":base,"stress1_5x":s15,"stress2x":s2,
             "symbolGrossContributions":{s:float(contrib[i]) for i,s in enumerate(SYMS)}},
           "gate":gate,"failedChecks":failed,
           "nextAction":"Open frozen fresh validation only." if not failed else "Close ARP22-POS2; validation and holdout remain sealed."}
    trans={"schema":"mgpt_arp22_pos2_transport_diagnostics_v1","generation":"ARP22-POS2","generatedAtUtc":datetime.now(timezone.utc).isoformat(),
           "checksumRequired":True,"manifestCount":len(manifest),"symbols":diag}
    cp=OUT/"ARP22_POS2_DEVELOPMENT_CLOSEOUT_20260915.json"; tp=OUT/"ARP22_POS2_TRANSPORT_DIAGNOSTICS_20260915.json"
    cp.write_text(json.dumps(close,indent=2)+"\n"); tp.write_text(json.dumps(trans,indent=2)+"\n")
    sums=[f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(OUT.glob("ARP22_POS2_*_20260915.json"))]
    (OUT/"ARP22_POS2_SHA256SUMS_20260915.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps({"status":close["status"],"failedChecks":failed,"development":close["development"]},indent=2))
if __name__=="__main__":main()
