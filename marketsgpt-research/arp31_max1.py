#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, io, json, math, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import requests

ROOT=Path(__file__).resolve().parent
LOCK_PATH=ROOT/"ARP31_MAX1_WEEKLY_EXTREME_RETURN_LOCK_20260916.json"
OUT=ROOT/"results"; OUT.mkdir(parents=True,exist_ok=True)
LOCK=json.loads(LOCK_PATH.read_text())
SYMS=LOCK["universe"]["symbols"]
BASE="https://data.binance.vision/data/futures/um/monthly/klines"
WARM=datetime.fromisoformat(LOCK["windows"]["warmupStart"].replace("Z","+00:00"))
DEV_START=datetime.fromisoformat(LOCK["windows"]["development"]["start"].replace("Z","+00:00"))
DEV_END=datetime.fromisoformat(LOCK["windows"]["development"]["endExclusive"].replace("Z","+00:00"))
DAY=86400000
SESSION=requests.Session(); SESSION.headers.update({"User-Agent":"MarketsGPT-ARP31-MAX1/1.0"})

def ms(x): return int(x.timestamp()*1000)

def month_keys(a,b):
    y,m=a.year,a.month; out=[]
    while (y,m)<=(b.year,b.month):
        out.append(f"{y:04d}-{m:02d}"); m+=1
        if m==13: y+=1; m=1
    return out

MONTHS=month_keys(WARM,DEV_END+timedelta(days=7))

def get_bytes(url,tries=5):
    last=None
    for k in range(tries):
        try:
            r=SESSION.get(url,timeout=45)
            if r.status_code==200: return r.content
            if r.status_code==404: return None
            last=f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e: last=repr(e)
        time.sleep(min(8,.75*(2**k)))
    raise RuntimeError(f"download failed {url}: {last}")

def download_month(sym,ym):
    name=f"{sym}-1d-{ym}.zip"; url=f"{BASE}/{sym}/1d/{name}"
    blob=get_bytes(url)
    if blob is None: return {"symbol":sym,"ym":ym,"status":"404","rows":[]}
    chk=get_bytes(url+".CHECKSUM")
    if chk is None: raise RuntimeError(f"missing CHECKSUM {url}")
    exp=chk.decode("utf-8","replace").strip().split()[0].lower()
    got=hashlib.sha256(blob).hexdigest()
    if exp!=got: raise RuntimeError(f"checksum mismatch {name}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        ns=[n for n in z.namelist() if not n.endswith("/")]
        if len(ns)!=1: raise RuntimeError(f"unexpected members {name}: {ns}")
        raw=z.read(ns[0]).decode("utf-8-sig","replace")
    rows=[]
    for rec in csv.reader(io.StringIO(raw)):
        if not rec: continue
        try: t=int(float(rec[0]))
        except Exception: continue
        if t>10**14: t//=1000
        if len(rec)<8: continue
        try: o=float(rec[1]); c=float(rec[4]); qv=float(rec[7])
        except Exception: continue
        if min(o,c)>0 and qv>=0 and all(map(math.isfinite,[o,c,qv])):
            rows.append((t,o,c,qv))
    return {"symbol":sym,"ym":ym,"status":"ok","sha256":got,"rows":rows}

def transport():
    jobs=[]
    with ThreadPoolExecutor(max_workers=32) as ex:
        for s in SYMS:
            for ym in MONTHS: jobs.append(ex.submit(download_month,s,ym))
        got=[f.result() for f in as_completed(jobs)]
    data={s:{} for s in SYMS}; manifest=[]; dup={s:0 for s in SYMS}
    for x in got:
        manifest.append({k:v for k,v in x.items() if k!="rows"})
        for row in x["rows"]:
            if row[0] in data[x["symbol"]]: dup[x["symbol"]]+=1
            data[x["symbol"]][row[0]]=row[1:]
    diag=[]
    for s in SYMS:
        ts=sorted(data[s]); gaps=sum(1 for a,b in zip(ts,ts[1:]) if b-a!=DAY)
        diag.append({
            "symbol":s,"rows":len(ts),
            "firstUtc":datetime.fromtimestamp(ts[0]/1000,timezone.utc).isoformat() if ts else None,
            "lastUtc":datetime.fromtimestamp(ts[-1]/1000,timezone.utc).isoformat() if ts else None,
            "duplicateRows":dup[s],"postFirstGapCount":gaps
        })
    return data,manifest,diag

def pf(a):
    a=np.asarray(a,float); p=float(a[a>0].sum()); n=float(-a[a<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)

def maxdd(a):
    eq=1.; peak=1.; d=0.
    for r in np.asarray(a,float):
        eq=max(1e-12,eq*(1+float(r))); peak=max(peak,eq); d=max(d,1-eq/peak)
    return d

def boot(a,seed,it=10000,block=4):
    a=np.asarray(a,float); n=len(a)
    if n==0: return {"probPositive":0.0,"ci95Low":0.0,"ci95High":0.0}
    rng=np.random.default_rng(seed); z=np.empty(it)
    for k in range(it):
        vals=[]
        while len(vals)<n:
            st=int(rng.integers(0,n)); take=min(block,n-len(vals))
            vals.extend(a[(st+np.arange(take))%n])
        z[k]=float(np.mean(vals))
    z.sort()
    return {"probPositive":float(np.mean(z>0)),"ci95Low":float(z[int(.025*(it-1))]),"ci95High":float(z[int(.975*(it-1))])}

def stats(a,dates,seed):
    a=np.asarray(a,float)
    m=float(a.mean()) if len(a) else 0.; sd=float(a.std(ddof=1)) if len(a)>1 else 0.
    months={}
    for r,d in zip(a,dates): months[d[:7]]=months.get(d[:7],0.)+float(r)
    return {
        "n":int(len(a)),"meanWeeklyReturn":m,"annualizedMean":m*52,
        "annualizedSharpe":float(m/sd*math.sqrt(52)) if sd>0 else 0.,
        "profitFactor":pf(a),"maxDrawdown":maxdd(a),
        "positiveMonthFraction":sum(v>0 for v in months.values())/max(1,len(months)),
        "monthSums":months,"bootstrap":boot(a,seed)
    }

def monday_grid(a,b):
    out=[]; t=a
    while t<b: out.append(t); t+=timedelta(days=7)
    return out

def main():
    data,manifest,diag=transport()
    grid=monday_grid(DEV_START,DEV_END)
    prev=np.zeros(len(SYMS))
    gross=[]; turnovers=[]; dates=[]; elig_counts=[]; active=0
    contrib=np.zeros(len(SYMS))
    selection_counts={s:{"long":0,"short":0} for s in SYMS}
    rows=[]

    minliq=10_000_000.0
    minage=int(LOCK["method"]["minimumArchiveAgeDays"])
    topn=50

    first_ts={s:(min(data[s]) if data[s] else None) for s in SYMS}

    for dt in grid:
        t=ms(dt); nxt=t+7*DAY
        cand=[]
        for i,s in enumerate(SYMS):
            d=data[s]
            if first_ts[s] is None or t-first_ts[s] < minage*DAY: continue
            entry=d.get(t); exitrow=d.get(nxt)
            if entry is None or exitrow is None: continue

            qv=[]; ok=True
            for k in range(1,31):
                row=d.get(t-k*DAY)
                if row is None: ok=False; break
                qv.append(row[2])
            if not ok: continue
            liq=float(np.mean(qv))
            if liq<minliq: continue

            closes=[]
            for k in range(29,0,-1):
                row=d.get(t-k*DAY)
                if row is None: ok=False; break
                closes.append(float(row[1]))
            if not ok or len(closes)!=29: continue
            daily=np.asarray(closes[1:],float)/np.asarray(closes[:-1],float)-1.0
            if not np.isfinite(daily).all(): continue
            sig=float(np.max(daily))
            ret=float(exitrow[0]/entry[0]-1.0)
            if math.isfinite(sig) and math.isfinite(ret): cand.append((i,s,sig,liq,ret))

        cand.sort(key=lambda x:(-x[3],x[1]))
        cand=cand[:topn]
        elig_counts.append(len(cand))
        desired=np.zeros(len(SYMS))
        longs=[]; shorts=[]
        if len(cand)>=LOCK["universe"]["minimumEligiblePerRebalance"]:
            active+=1
            ranked=sorted(cand,key=lambda x:(-x[2],x[1]))
            qn=max(1,len(ranked)//10)
            longs=ranked[:qn]; shorts=ranked[-qn:]
            for i,s,_,_,_ in longs:
                desired[i]+=0.5/qn; selection_counts[s]["long"]+=1
            for i,s,_,_,_ in shorts:
                desired[i]-=0.5/qn; selection_counts[s]["short"]+=1

        retmap={i:r for i,_,_,_,r in cand}
        g=0.; by=np.zeros(len(SYMS))
        for i,w in enumerate(desired):
            if w and i in retmap:
                z=w*retmap[i]; g+=z; by[i]=z
        contrib+=by
        turnover=float(np.abs(desired-prev).sum())
        gross.append(g); turnovers.append(turnover); dates.append(dt.date().isoformat())
        rows.append({
            "rebalanceUtc":dt.isoformat(),"eligible":len(cand),
            "long":[s for _,s,_,_,_ in longs],"short":[s for _,s,_,_,_ in shorts],
            "longSignals":[float(z) for _,_,z,_,_ in longs],"shortSignals":[float(z) for _,_,z,_,_ in shorts],
            "grossReturn":g,"turnover":turnover
        })
        prev=desired

    if turnovers:
        terminal=float(np.abs(prev).sum()); turnovers[-1]+=terminal; rows[-1]["terminalLiquidationTurnover"]=terminal

    gross=np.asarray(gross,float); turnovers=np.asarray(turnovers,float)
    c=LOCK["costs"]
    net=gross-turnovers*c["baseOneWayRate"]
    n15=gross-turnovers*c["stress1_5xOneWayRate"]
    n2=gross-turnovers*c["stress2xOneWayRate"]

    gs=stats(gross,dates,31100); ns=stats(net,dates,31101); s15=stats(n15,dates,31111); s2=stats(n2,dates,31112)
    pos=contrib[contrib>0]
    dom=float(pos.max()/pos.sum()) if len(pos) and pos.sum()>0 else 0.
    cov=active/max(1,len(grid)); avg=float(np.mean(elig_counts)) if elig_counts else 0.
    half=len(net)//2
    fh=float(net[:half].sum()); sh=float(net[half:].sum())
    g=LOCK["developmentGate"]
    checks={
        "minimumEvaluatedWeeks":len(net)>=g["minimumEvaluatedWeeks"],
        "minimumCoverageFraction":cov>=g["minimumCoverageFraction"],
        "averageEligibleSymbolsMin":avg>=g["averageEligibleSymbolsMin"],
        "netMeanWeeklyReturnMin":ns["meanWeeklyReturn"]>=g["netMeanWeeklyReturnMin"],
        "netAnnualizedSharpeMin":ns["annualizedSharpe"]>=g["netAnnualizedSharpeMin"],
        "netProfitFactorMin":ns["profitFactor"]>=g["netProfitFactorMin"],
        "stress1_5xMeanWeeklyReturnMin":s15["meanWeeklyReturn"]>=g["stress1_5xMeanWeeklyReturnMin"],
        "stress2xMeanWeeklyReturnMin":s2["meanWeeklyReturn"]>=g["stress2xMeanWeeklyReturnMin"],
        "positiveMonthFractionMin":ns["positiveMonthFraction"]>=g["positiveMonthFractionMin"],
        "firstHalfNetReturnPositive":fh>0,
        "secondHalfNetReturnPositive":sh>0,
        "bootstrapProbabilityPositiveMin":ns["bootstrap"]["probPositive"]>=g["bootstrapProbabilityPositiveMin"],
        "bootstrapCi95LowMin":ns["bootstrap"]["ci95Low"]>=g["bootstrapCi95LowMin"],
        "maxDrawdownMax":ns["maxDrawdown"]<=g["maxDrawdownMax"],
        "singleSymbolPositiveContributionMax":dom<=g["singleSymbolPositiveContributionMax"]
    }
    checks={k:bool(v) for k,v in checks.items()}
    failed=[k for k,v in checks.items() if not v]
    status="PASS_DEVELOPMENT_GATE" if not failed else "REJECT_DEVELOPMENT_GATE"
    close={
        "schema":"mgpt_arp31_max1_development_closeout_v1","generation":"ARP31-MAX1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),"status":status,
        "productionAuthority":False,"r15MutationAllowed":False,
        "freshValidationAuthorized":not failed,"sealedHoldoutOpened":False,
        "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "development":{
            "expectedWeeks":len(grid),"evaluatedWeeks":len(net),"activeCoverageWeeks":active,
            "coverageFraction":cov,"averageEligibleSymbols":avg,"meanTurnover":float(turnovers.mean()) if len(turnovers) else 0.,
            "singleSymbolPositiveContribution":dom,"firstHalfNetReturn":fh,"secondHalfNetReturn":sh,
            "gross":gs,"baseCost":ns,"stress1_5x":s15,"stress2x":s2,
            "symbolGrossContributions":{s:float(contrib[i]) for i,s in enumerate(SYMS)},
            "selectionCounts":selection_counts
        },
        "checks":checks,"failedChecks":failed,
        "nextAction":"Run the unchanged predeclared fresh validation window only." if not failed else "Close ARP31-MAX1; no rescue; validation and holdout remain sealed."
    }
    trans={
        "schema":"mgpt_arp31_max1_transport_diagnostics_v1","generation":"ARP31-MAX1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),"checksumRequired":True,
        "manifestCount":len(manifest),"symbols":diag
    }
    (OUT/"ARP31_MAX1_DEVELOPMENT_CLOSEOUT_20260916.json").write_text(json.dumps(close,indent=2)+"\n")
    (OUT/"ARP31_MAX1_TRANSPORT_DIAGNOSTICS_20260916.json").write_text(json.dumps(trans,indent=2)+"\n")
    (OUT/"ARP31_MAX1_WEEKLY_LEDGER_20260916.json").write_text(json.dumps({"schema":"mgpt_arp31_max1_weekly_ledger_v1","rows":rows},indent=2)+"\n")
    sums=[]
    for p in sorted(OUT.glob("ARP31_MAX1_*_20260916.json")):
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/"ARP31_MAX1_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps({"status":status,"failedChecks":failed,"development":{k:v for k,v in close["development"].items() if k not in ("symbolGrossContributions","selectionCounts")}},indent=2))

if __name__=="__main__":
    main()
