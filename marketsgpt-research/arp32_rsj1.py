#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, io, json, math, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "ARP32_RSJ1_REALIZED_SIGNED_JUMP_LOCK_20260916.json"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

LOCK = json.loads(LOCK_PATH.read_text())
SYMS = LOCK["universe"]["symbols"]
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
FETCH_START = datetime.fromisoformat(LOCK["windows"]["archiveFetchStart"].replace("Z","+00:00"))
DEV_START = datetime.fromisoformat(LOCK["windows"]["development"]["start"].replace("Z","+00:00"))
DEV_END = datetime.fromisoformat(LOCK["windows"]["development"]["endExclusive"].replace("Z","+00:00"))
MIN_BARS = int(LOCK["dataAuthority"]["minimumMinuteBarsPerSignalDay"])
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"MarketsGPT-ARP32-RSJ1/1.0"})

def month_keys(a,b):
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

def get_bytes(url, tries=5):
    last=None
    for k in range(tries):
        try:
            r=SESSION.get(url,timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            last=f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            last=repr(e)
        time.sleep(min(8.0,0.75*(2**k)))
    raise RuntimeError(f"download failed {url}: {last}")

def download_month(sym,ym):
    name=f"{sym}-1m-{ym}.zip"
    url=f"{BASE}/{sym}/1m/{name}"
    blob=get_bytes(url)
    if blob is None:
        return {"symbol":sym,"ym":ym,"status":"404","sha256":None,"days":[]}
    chk=get_bytes(url+".CHECKSUM")
    if chk is None:
        raise RuntimeError(f"missing CHECKSUM {url}")
    expected=chk.decode("utf-8","replace").strip().split()[0].lower()
    got=hashlib.sha256(blob).hexdigest()
    if expected != got:
        raise RuntimeError(f"checksum mismatch {name}")

    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=[n for n in z.namelist() if not n.endswith("/")]
        if len(names)!=1:
            raise RuntimeError(f"unexpected members {name}: {names}")
        raw=z.read(names[0]).decode("utf-8-sig","replace")

    # Aggregate minute data to one row per UTC day while preserving the
    # source-defined semivariance concept. The day-boundary return is excluded.
    cur_day=None
    first_open=None
    last_close=None
    qv_sum=0.0
    rv_plus=0.0
    rv_minus=0.0
    bars=0
    days=[]

    def flush():
        nonlocal cur_day,first_open,last_close,qv_sum,rv_plus,rv_minus,bars
        if cur_day is not None:
            days.append({
                "day":cur_day,
                "open":first_open,
                "close":last_close,
                "qv":qv_sum,
                "rvPlus":rv_plus,
                "rvMinus":rv_minus,
                "rsj":rv_plus-rv_minus,
                "bars":bars
            })

    for rec in csv.reader(io.StringIO(raw)):
        if not rec:
            continue
        try:
            t=int(float(rec[0]))
            if t > 10**14:
                t//=1000
            o=float(rec[1]); c=float(rec[4]); qv=float(rec[7])
        except Exception:
            continue
        if not (o>0 and c>0 and qv>=0 and all(map(math.isfinite,[o,c,qv]))):
            continue
        dt=datetime.fromtimestamp(t/1000,timezone.utc)
        day=dt.date().isoformat()
        if day != cur_day:
            flush()
            cur_day=day
            first_open=o
            last_close=c
            qv_sum=qv
            rv_plus=0.0
            rv_minus=0.0
            bars=1
            continue
        r=math.log(c/last_close) if last_close and last_close>0 else float("nan")
        if math.isfinite(r):
            z=r*r
            if r>0:
                rv_plus+=z
            elif r<0:
                rv_minus+=z
        last_close=c
        qv_sum+=qv
        bars+=1
    flush()
    return {"symbol":sym,"ym":ym,"status":"ok","sha256":got,"days":days}

def transport():
    futures=[]
    with ThreadPoolExecutor(max_workers=20) as ex:
        for s in SYMS:
            for ym in MONTHS:
                futures.append(ex.submit(download_month,s,ym))
        got=[f.result() for f in as_completed(futures)]

    daily={s:{} for s in SYMS}
    manifest=[]
    duplicates={s:0 for s in SYMS}
    for x in got:
        manifest.append({k:v for k,v in x.items() if k!="days"})
        for row in x["days"]:
            d=row["day"]
            if d in daily[x["symbol"]]:
                duplicates[x["symbol"]]+=1
            daily[x["symbol"]][d]=row

    diagnostics=[]
    for s in SYMS:
        days=sorted(daily[s])
        complete=sum(1 for d in days if int(daily[s][d]["bars"])>=MIN_BARS)
        diagnostics.append({
            "symbol":s,
            "days":len(days),
            "completeSignalDays":complete,
            "firstDay":days[0] if days else None,
            "lastDay":days[-1] if days else None,
            "duplicateDays":duplicates[s],
            "minBars":min((int(daily[s][d]["bars"]) for d in days),default=0),
            "maxBars":max((int(daily[s][d]["bars"]) for d in days),default=0)
        })
    return daily,manifest,diagnostics

def pf(a):
    a=np.asarray(a,float)
    p=float(a[a>0].sum()); n=float(-a[a<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)

def maxdd(a):
    eq=1.0; peak=1.0; d=0.0
    for r in np.asarray(a,float):
        eq=max(1e-12,eq*(1.0+float(r)))
        peak=max(peak,eq)
        d=max(d,1.0-eq/peak)
    return d

def bootstrap(a,seed,it=10000,block=7):
    a=np.asarray(a,float)
    n=len(a)
    if n==0:
        return {"probPositive":0.0,"ci95Low":0.0,"ci95High":0.0}
    rng=np.random.default_rng(seed)
    z=np.empty(it,float)
    for k in range(it):
        vals=[]
        while len(vals)<n:
            st=int(rng.integers(0,n))
            take=min(block,n-len(vals))
            vals.extend(a[(st+np.arange(take))%n])
        z[k]=float(np.mean(vals))
    z.sort()
    return {
        "probPositive":float(np.mean(z>0)),
        "ci95Low":float(z[int(.025*(it-1))]),
        "ci95High":float(z[int(.975*(it-1))])
    }

def stats(a,dates,seed):
    a=np.asarray(a,float)
    m=float(a.mean()) if len(a) else 0.0
    sd=float(a.std(ddof=1)) if len(a)>1 else 0.0
    months={}
    for r,d in zip(a,dates):
        months[d[:7]]=months.get(d[:7],0.0)+float(r)
    return {
        "n":int(len(a)),
        "meanDailyReturn":m,
        "annualizedMean":m*365.0,
        "annualizedSharpe":float(m/sd*math.sqrt(365.0)) if sd>0 else 0.0,
        "profitFactor":pf(a),
        "maxDrawdown":maxdd(a),
        "positiveMonthFraction":sum(v>0 for v in months.values())/max(1,len(months)),
        "monthSums":months,
        "bootstrap":bootstrap(a,seed)
    }

def date_grid(a,b):
    out=[]
    d=a
    while d<b:
        out.append(d)
        d+=timedelta(days=1)
    return out

def compound(a):
    a=np.asarray(a,float)
    return float(np.prod(1.0+a)-1.0) if len(a) else 0.0

def main():
    daily,manifest,diagnostics=transport()
    decisions=date_grid(DEV_START,DEV_END)
    prev=np.zeros(len(SYMS),float)
    gross=[]; turnovers=[]; dates=[]; elig_counts=[]
    contrib=np.zeros(len(SYMS),float)
    rows=[]; active=0

    minliq=float(LOCK["universe"]["prior30CompletedDaysMeanDailyQuoteVolumeUsdMin"])
    minelig=int(LOCK["universe"]["minimumEligiblePerDecision"])

    for dt in decisions:
        day=dt.date().isoformat()
        prev_day=(dt-timedelta(days=1)).date().isoformat()
        next_day=(dt+timedelta(days=1)).date().isoformat()
        cand=[]
        for i,s in enumerate(SYMS):
            d=daily[s]
            sigrow=d.get(prev_day); erow=d.get(day); xrow=d.get(next_day)
            if sigrow is None or erow is None or xrow is None:
                continue
            if int(sigrow["bars"]) < MIN_BARS:
                continue
            qvs=[]; ok=True
            for k in range(1,31):
                kd=(dt-timedelta(days=k)).date().isoformat()
                rr=d.get(kd)
                if rr is None or int(rr["bars"])<MIN_BARS:
                    ok=False; break
                qvs.append(float(rr["qv"]))
            if not ok:
                continue
            liq=float(np.mean(qvs))
            if liq < minliq:
                continue
            sig=float(sigrow["rsj"])
            o0=float(erow["open"]); o1=float(xrow["open"])
            ret=o1/o0-1.0 if o0>0 and o1>0 else float("nan")
            if math.isfinite(sig) and math.isfinite(ret):
                cand.append((i,s,sig,liq,ret))

        elig_counts.append(len(cand))
        desired=np.zeros(len(SYMS),float)
        longs=[]; shorts=[]
        if len(cand) >= minelig:
            active+=1
            ranked=sorted(cand,key=lambda x:(x[2],x[1]))  # lowest RSJ first
            q=max(1,len(ranked)//5)
            longs=ranked[:q]
            shorts=ranked[-q:]
            for i,s,_,_,_ in longs:
                desired[i]+=0.5/q
            for i,s,_,_,_ in shorts:
                desired[i]-=0.5/q

        retmap={i:r for i,_,_,_,r in cand}
        g=0.0; by=np.zeros(len(SYMS),float)
        for i,w in enumerate(desired):
            if w and i in retmap:
                z=w*retmap[i]
                g+=z
                by[i]=z
        contrib+=by
        turnover=float(np.abs(desired-prev).sum())
        gross.append(g); turnovers.append(turnover); dates.append(day)
        rows.append({
            "decisionUtc":dt.isoformat(),
            "signalDay":prev_day,
            "eligible":len(cand),
            "long":[s for _,s,_,_,_ in longs],
            "short":[s for _,s,_,_,_ in shorts],
            "longRsj":[float(sig) for _,_,sig,_,_ in longs],
            "shortRsj":[float(sig) for _,_,sig,_,_ in shorts],
            "grossReturn":g,
            "turnover":turnover
        })
        prev=desired

    if turnovers:
        terminal=float(np.abs(prev).sum())
        turnovers[-1]+=terminal
        rows[-1]["terminalLiquidationTurnover"]=terminal

    gross=np.asarray(gross,float)
    turnovers=np.asarray(turnovers,float)
    c=LOCK["costs"]
    net=gross-turnovers*float(c["baseOneWayRate"])
    n15=gross-turnovers*float(c["stress1_5xOneWayRate"])
    n2=gross-turnovers*float(c["stress2xOneWayRate"])

    gs=stats(gross,dates,32100)
    ns=stats(net,dates,32101)
    s15=stats(n15,dates,32111)
    s2=stats(n2,dates,32112)

    pos=np.maximum(contrib,0.0)
    dom=float(pos.max()/pos.sum()) if pos.sum()>0 else 0.0
    coverage=active/max(1,len(decisions))
    avg=float(np.mean(elig_counts)) if elig_counts else 0.0
    half=len(net)//2
    fh=compound(net[:half])
    sh=compound(net[half:])

    g=LOCK["developmentGate"]
    checks={
        "minimumEvaluatedDays":len(net)>=g["minimumEvaluatedDays"],
        "minimumCoverageFraction":coverage>=g["minimumCoverageFraction"],
        "averageEligibleSymbolsMin":avg>=g["averageEligibleSymbolsMin"],
        "baseNetMeanDailyReturnMin":ns["meanDailyReturn"]>=g["baseNetMeanDailyReturnMin"],
        "baseNetAnnualizedSharpeMin":ns["annualizedSharpe"]>=g["baseNetAnnualizedSharpeMin"],
        "baseNetProfitFactorMin":ns["profitFactor"]>=g["baseNetProfitFactorMin"],
        "stress1_5xMeanDailyReturnMin":s15["meanDailyReturn"]>=g["stress1_5xMeanDailyReturnMin"],
        "stress2xMeanDailyReturnMin":s2["meanDailyReturn"]>=g["stress2xMeanDailyReturnMin"],
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
    passed=not failed

    close={
        "schema":"mgpt_arp32_rsj1_development_closeout_v1",
        "generation":"ARP32-RSJ1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS_DEVELOPMENT_GATE_FUNDING_REPLAY_REQUIRED" if passed else "REJECT_DEVELOPMENT_GATE_TERMINAL_STOP_RULE",
        "productionAuthority":False,
        "r15MutationAllowed":False,
        "fundingReplayRequiredBeforeValidation":passed,
        "freshValidationOpened":False,
        "sealedHoldoutOpened":False,
        "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "development":{
            "expectedDays":len(decisions),
            "evaluatedDays":len(net),
            "activeCoverageDays":active,
            "coverageFraction":coverage,
            "averageEligibleSymbols":avg,
            "meanTurnover":float(turnovers.mean()) if len(turnovers) else 0.0,
            "singleSymbolPositiveContribution":dom,
            "firstHalfNetReturn":fh,
            "secondHalfNetReturn":sh,
            "gross":gs,
            "baseCost":ns,
            "stress1_5x":s15,
            "stress2x":s2,
            "symbolGrossContributions":{s:float(contrib[i]) for i,s in enumerate(SYMS)}
        },
        "checks":checks,
        "failedChecks":failed,
        "researchProgramStopRuleTriggered":not passed,
        "nextAction":(
            "Run exact historical funding-cashflow replay over the frozen development ledger. Do not open validation until funding-adjusted development remains inside the unchanged gate contract."
            if passed else
            "Trigger the predeclared terminal research-program stop rule. Do not launch another price-alpha family in this closure campaign; preserve RC4S3H5R4R15 unchanged and close profitability as not proven."
        )
    }

    transport_out={
        "schema":"mgpt_arp32_rsj1_transport_diagnostics_v1",
        "generation":"ARP32-RSJ1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "checksumRequired":True,
        "months":MONTHS,
        "manifest":manifest,
        "symbols":diagnostics
    }
    (OUT/"ARP32_RSJ1_DEVELOPMENT_CLOSEOUT_20260916.json").write_text(json.dumps(close,indent=2)+"\n")
    (OUT/"ARP32_RSJ1_TRANSPORT_DIAGNOSTICS_20260916.json").write_text(json.dumps(transport_out,indent=2)+"\n")
    (OUT/"ARP32_RSJ1_DAILY_LEDGER_20260916.json").write_text(json.dumps({"schema":"mgpt_arp32_rsj1_daily_ledger_v1","rows":rows},indent=2)+"\n")

    sums=[]
    for p in sorted(OUT.glob("ARP32_RSJ1_*_20260916.json")):
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/"ARP32_RSJ1_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")

    print(json.dumps({
        "status":close["status"],
        "failedChecks":failed,
        "researchProgramStopRuleTriggered":close["researchProgramStopRuleTriggered"],
        "development":{k:v for k,v in close["development"].items() if k!="symbolGrossContributions"}
    },indent=2))

if __name__=="__main__":
    main()
