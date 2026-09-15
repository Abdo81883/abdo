#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, io, json, math, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "ARP21_FPRV_EXACT_R2_LOCK_20260915.json"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOCK = json.loads(LOCK_PATH.read_text())
SYMS = LOCK["cohort"]["symbols"]
TF_MS = 8 * 60 * 60 * 1000
FORM_DAYS = LOCK["methodAuthority"]["formationDays"]
FORM_BARS = [d * 3 for d in FORM_DAYS]
FAMILIES = ["meanPV", "meanSPVI", "varPV", "varSPVI"]
FACTOR_NAMES = [f"{fam}_{d}d" for fam in FAMILIES for d in FORM_DAYS]
BASE = "https://data.binance.vision/data/futures/um/monthly"
FETCH_START = datetime.fromisoformat(LOCK["windows"]["archiveFetchStart"].replace("Z","+00:00"))
DEV_START = datetime.fromisoformat(LOCK["windows"]["development"]["start"].replace("Z","+00:00"))
DEV_END = datetime.fromisoformat(LOCK["windows"]["development"]["endExclusive"].replace("Z","+00:00"))
FETCH_END = DEV_END
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"MarketsGPT-ARP21-FPRV-R2/1.0"})

def ms(dt): return int(dt.timestamp()*1000)

def month_keys(start, end):
    y,m=start.year,start.month
    out=[]
    while (y,m) <= (end.year,end.month):
        out.append(f"{y:04d}-{m:02d}")
        m+=1
        if m==13:
            y+=1
            m=1
    return out

MONTHS = month_keys(FETCH_START, FETCH_END)

def get_bytes(url, tries=5):
    last=None
    for k in range(tries):
        try:
            r=SESSION.get(url, timeout=45)
            if r.status_code==200:
                return r.content
            if r.status_code==404:
                return None
            last=f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            last=repr(e)
        time.sleep(min(8, 0.75*(2**k)))
    raise RuntimeError(f"download failed {url}: {last}")

def download_one(kind, sym, ym):
    name=f"{sym}-8h-{ym}.zip"
    url=f"{BASE}/{kind}/{sym}/8h/{name}"
    blob=get_bytes(url)
    if blob is None:
        return {"kind":kind,"symbol":sym,"ym":ym,"status":"404","rows":[]}
    chk=get_bytes(url+".CHECKSUM")
    if chk is None:
        raise RuntimeError(f"missing CHECKSUM {url}")
    expected=chk.decode("utf-8","replace").strip().split()[0].lower()
    actual=hashlib.sha256(blob).hexdigest()
    if expected!=actual:
        raise RuntimeError(f"checksum mismatch {name}: {expected} != {actual}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=[n for n in z.namelist() if not n.endswith("/")]
        if len(names)!=1:
            raise RuntimeError(f"unexpected zip members {name}: {names}")
        text=z.read(names[0]).decode("utf-8-sig","replace")
    rows=[]
    for rec in csv.reader(io.StringIO(text)):
        if not rec:
            continue
        try:
            int(float(rec[0]))
        except Exception:
            continue
        if len(rec)>=5:
            rows.append(rec)
    return {"kind":kind,"symbol":sym,"ym":ym,"status":"ok","sha256":actual,"rows":rows}

def transport():
    jobs=[]
    with ThreadPoolExecutor(max_workers=16) as ex:
        for sym in SYMS:
            for ym in MONTHS:
                for kind in ("klines","indexPriceKlines"):
                    jobs.append(ex.submit(download_one,kind,sym,ym))
        got=[f.result() for f in as_completed(jobs)]
    per={s:{} for s in SYMS}
    idx={s:{} for s in SYMS}
    manifest=[]
    dupes={s:{"klines":0,"indexPriceKlines":0} for s in SYMS}
    for x in got:
        manifest.append({k:v for k,v in x.items() if k!="rows"})
        target=per if x["kind"]=="klines" else idx
        for r in x["rows"]:
            t=int(float(r[0]))
            if t in target[x["symbol"]]:
                dupes[x["symbol"]][x["kind"]]+=1
            target[x["symbol"]][t]=r
    data={}
    diagnostics=[]
    for sym in SYMS:
        pts=[]
        common=sorted(set(per[sym]) & set(idx[sym]))
        for t in common:
            pr=per[sym][t]
            ir=idx[sym][t]
            try:
                o=float(pr[1]); c=float(pr[4]); qv=float(pr[7]); tbq=float(pr[10]); ic=float(ir[4])
            except Exception:
                continue
            if min(o,c,ic)<=0 or qv<0 or tbq<0:
                continue
            pts.append({"t":t,"open":o,"close":c,"basis":math.log(c/ic),"pv":qv,"spvi":2.0*tbq-qv})
        pts.sort(key=lambda x:x["t"])
        gaps=sum(1 for a,b in zip(pts,pts[1:]) if b["t"]-a["t"]!=TF_MS)
        data[sym]=pts
        diagnostics.append({
            "symbol":sym,
            "perpRows":len(per[sym]),
            "indexRows":len(idx[sym]),
            "alignedRows":len(pts),
            "firstUtc":datetime.fromtimestamp(pts[0]["t"]/1000,timezone.utc).isoformat() if pts else None,
            "lastUtc":datetime.fromtimestamp(pts[-1]["t"]/1000,timezone.utc).isoformat() if pts else None,
            "postFirstTimestampGapCount":gaps,
            "duplicateArchiveRows":dupes[sym]
        })
    return data, manifest, diagnostics

def prep_symbol(rows):
    t=np.array([r["t"] for r in rows],dtype=np.int64)
    op=np.array([r["open"] for r in rows],float)
    basis=np.array([r["basis"] for r in rows],float)
    pv=np.array([r["pv"] for r in rows],float)
    sp=np.array([r["spvi"] for r in rows],float)
    def cs(x):
        return np.concatenate(([0.0],np.cumsum(x,dtype=float)))
    return {
        "t":t,"open":op,"basis":basis,"pv":pv,"sp":sp,
        "cpv":cs(pv),"cpv2":cs(pv*pv),"csp":cs(sp),"csp2":cs(sp*sp),
        "pos":{int(x):i for i,x in enumerate(t)}
    }

def win_stats(c,c2,start,end):
    n=end-start
    s=c[end]-c[start]
    m=s/n
    if n<2:
        return m,0.0
    ss=c2[end]-c2[start]
    v=max(0.0,(ss-s*s/n)/(n-1))
    return m,v

def portfolio_factor(group, name):
    g=sorted(group,key=lambda z:(z["vars"][name],z["sym"]))
    q=len(g)//5
    if q<1:
        return None
    return g[:q],g[-q:]

def bootstrap(a, iterations=10000, block=21, seed=21202):
    rng=np.random.default_rng(seed)
    a=np.asarray(a,float)
    n=len(a)
    out=np.empty(iterations)
    for k in range(iterations):
        vals=[]
        while len(vals)<n:
            st=int(rng.integers(0,n))
            take=min(block,n-len(vals))
            vals.extend(a[(st+np.arange(take))%n])
        out[k]=float(np.mean(vals))
    out.sort()
    return {
        "probPositive":float(np.mean(out>0)),
        "ci95Low":float(out[int(.025*(iterations-1))]),
        "ci95High":float(out[int(.975*(iterations-1))])
    }

def stats(a,times,seed):
    a=np.asarray(a,float)
    m=float(np.mean(a))
    sd=float(np.std(a,ddof=1)) if len(a)>1 else 0.0
    pos=float(np.sum(a[a>0]))
    neg=float(-np.sum(a[a<0]))
    e=1.0; pk=1.0; dd=0.0
    months={}
    for r,t in zip(a,times):
        e=max(1e-12,e*(1+r))
        pk=max(pk,e)
        dd=max(dd,1-e/pk)
        key=datetime.fromtimestamp(t/1000,timezone.utc).strftime("%Y-%m")
        months[key]=months.get(key,0.0)+float(r)
    return {
        "n":len(a),
        "meanPer8h":m,
        "annualizedMean":m*1095,
        "annualizedSharpe":float(m/sd*math.sqrt(1095)) if sd>0 else 0.0,
        "profitFactor":pos/neg if neg>0 else (999.0 if pos>0 else 0.0),
        "maxDrawdown":dd,
        "positiveMonthFraction":sum(v>0 for v in months.values())/max(1,len(months)),
        "monthSums":months,
        "bootstrap":bootstrap(a,seed=seed)
    }

def main():
    data_raw,manifest,transport_diag=transport()
    data={s:prep_symbol(data_raw[s]) for s in SYMS}
    dev_start=ms(DEV_START)
    dev_end=ms(DEV_END)
    all_times=sorted({int(t) for s in SYMS for t in data[s]["t"] if dev_start-TF_MS <= t < dev_end})
    rows=[]
    weight_cube=[]
    under_min=0
    maxw=max(FORM_BARS)
    for t in all_times:
        decision=t+TF_MS
        if not (dev_start <= decision < dev_end):
            continue
        elig=[]
        for si,sym in enumerate(SYMS):
            d=data[sym]
            i=d["pos"].get(t)
            if i is None or i<maxw-1 or i+2>=len(d["t"]):
                continue
            if int(d["t"][i+1])!=t+TF_MS or int(d["t"][i+2])!=t+2*TF_MS:
                continue
            ret=d["open"][i+2]/d["open"][i+1]-1.0
            vars_={}
            ok=True
            for days,w in zip(FORM_DAYS,FORM_BARS):
                st=i-w+1
                en=i+1
                mpv,vpv=win_stats(d["cpv"],d["cpv2"],st,en)
                msp,vsp=win_stats(d["csp"],d["csp2"],st,en)
                if not (mpv>0 and np.isfinite([mpv,vpv,msp,vsp]).all()):
                    ok=False
                    break
                vars_[f"meanPV_{days}d"]=math.log(mpv)
                vars_[f"meanSPVI_{days}d"]=msp
                vars_[f"varPV_{days}d"]=vpv
                vars_[f"varSPVI_{days}d"]=vsp
            if ok and math.isfinite(ret):
                elig.append({"si":si,"sym":sym,"basis":float(d["basis"][i]),"ret":float(ret),"vars":vars_})
        if len(elig)<LOCK["cohort"]["minimumEligiblePerTimestamp"]:
            under_min+=1
            continue
        elig.sort(key=lambda z:(z["basis"],z["sym"]))
        if len(elig)%2:
            elig=elig[:-1]
        h=len(elig)//2
        low=elig[:h]
        high=elig[h:]
        fr=[]
        fweights=[]
        good=True
        for name in FACTOR_NAMES:
            L=portfolio_factor(low,name)
            H=portfolio_factor(high,name)
            if not L or not H:
                good=False
                break
            l1,l5=L
            h1,h5=H
            r=.5*(np.mean([x["ret"] for x in l1])+np.mean([x["ret"] for x in h1]))-.5*(np.mean([x["ret"] for x in l5])+np.mean([x["ret"] for x in h5]))
            w=np.zeros(len(SYMS),float)
            for grp,coef in ((l1,.5),(h1,.5),(l5,-.5),(h5,-.5)):
                for x in grp:
                    w[x["si"]]+=coef/len(grp)
            fr.append(float(r))
            fweights.append(w)
        if good and len(fr)==32:
            rows.append((decision,fr))
            weight_cube.append(np.stack(fweights))
    if not rows:
        raise RuntimeError("No development rows evaluated")

    times=[x[0] for x in rows]
    X=np.asarray([x[1] for x in rows],float)
    T,P=X.shape
    cov=np.cov(X,rowvar=False,ddof=1)
    vals,vecs=np.linalg.eigh(cov)
    order=np.argsort(vals)[::-1]
    vals=vals[order]
    vecs=vecs[:,order]
    top=vecs[:,:3].copy()
    for k in range(3):
        s=float(np.sum(top[:,k]))
        if s<0:
            top[:,k]*=-1
        elif abs(s)<1e-12:
            j=int(np.argmax(np.abs(top[:,k])))
            if top[j,k]<0:
                top[:,k]*=-1

    total=float(np.trace(cov))
    explained=(vals[:3]/total) if total>0 else np.zeros(3)
    vw=explained/explained.sum() if explained.sum()>0 else np.ones(3)/3
    raw_pc=X@top
    gross=raw_pc@vw

    cube=np.stack(weight_cube)
    compw=np.zeros((T,len(SYMS)))
    for k in range(3):
        compw += vw[k] * np.einsum("tpn,p->tn",cube,top[:,k])

    turnover=np.zeros(T)
    prev=np.zeros(len(SYMS))
    for i,w in enumerate(compw):
        turnover[i]=np.abs(w-prev).sum()
        prev=w
    turnover[-1]+=np.abs(compw[-1]).sum()

    costs=LOCK["costs"]
    net=gross-turnover*costs["baseOneWayRate"]
    net15=gross-turnover*costs["stress1_5xOneWayRate"]
    net2=gross-turnover*costs["stress2xOneWayRate"]

    gs=stats(gross,times,21201)
    ns=stats(net,times,21202)
    n15=stats(net15,times,21203)
    n2=stats(net2,times,21204)
    expected=max(1,int((dev_end-dev_start)//TF_MS))
    coverage=T/expected
    ve3=float(explained.sum())
    gate_cfg=LOCK["developmentGate"]
    gate={
        "minimumEvaluated8hBars": T>=gate_cfg["minimumEvaluated8hBars"],
        "minimum32FactorCoverageFraction": coverage>=gate_cfg["minimum32FactorCoverageFraction"],
        "grossCompositeAnnualizedSharpeMin": gs["annualizedSharpe"]>=gate_cfg["grossCompositeAnnualizedSharpeMin"],
        "grossCompositeMeanPer8hMin": gs["meanPer8h"]>=gate_cfg["grossCompositeMeanPer8hMin"],
        "baseCostNetMeanPer8hMin": ns["meanPer8h"]>=gate_cfg["baseCostNetMeanPer8hMin"],
        "stress1_5xNetMeanPer8hMin": n15["meanPer8h"]>=gate_cfg["stress1_5xNetMeanPer8hMin"],
        "baseCostBootstrapProbabilityPositiveMin": ns["bootstrap"]["probPositive"]>=gate_cfg["baseCostBootstrapProbabilityPositiveMin"],
        "baseCostBootstrapCi95LowMin": ns["bootstrap"]["ci95Low"]>=gate_cfg["baseCostBootstrapCi95LowMin"],
        "positiveMonthFractionMin": ns["positiveMonthFraction"]>=gate_cfg["positiveMonthFractionMin"],
        "first3VarianceExplainedMin": ve3>=gate_cfg["first3VarianceExplainedMin"],
        "baseCostMaxDrawdownMax": ns["maxDrawdown"]<=gate_cfg["baseCostMaxDrawdownMax"]
    }
    gate={k:bool(v) for k,v in gate.items()}
    failed=[k for k,v in gate.items() if not v]

    factor_stats=[]
    for j,name in enumerate(FACTOR_NAMES):
        factor_stats.append({
            "name":name,
            "meanPer8h":float(X[:,j].mean()),
            "annualizedSharpe":stats(X[:,j],times,21300+j)["annualizedSharpe"]
        })
    factor_stats.sort(key=lambda z:z["annualizedSharpe"],reverse=True)

    closeout={
        "schema":"mgpt_arp21_fprv_exact_r2_development_closeout_v1",
        "generation":"ARP21-FPRV-EXACT-R2",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS_DEVELOPMENT_GATE" if not failed else "REJECT_DEVELOPMENT_GATE",
        "productionAuthority":False,
        "r15MutationAllowed":False,
        "validationAuthorized":not failed,
        "sealedHoldoutOpened":False,
        "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "development":{
            "evaluated8hBars":T,
            "expected8hBars":expected,
            "factorCoverageFraction":coverage,
            "timestampsBelowMinimumEligible":under_min,
            "firstDecisionUtc":datetime.fromtimestamp(times[0]/1000,timezone.utc).isoformat(),
            "lastDecisionUtc":datetime.fromtimestamp(times[-1]/1000,timezone.utc).isoformat(),
            "first3VarianceExplained":ve3,
            "pcaEigenvalues":vals[:3].tolist(),
            "pcaExplainedVariance":explained.tolist(),
            "pcaCompositeWeights":vw.tolist(),
            "meanCompositeTurnover":float(turnover.mean()),
            "gross":gs,
            "baseCost":ns,
            "stress1_5x":n15,
            "stress2x":n2,
            "strongestIndividualFactors":factor_stats[:8],
            "weakestIndividualFactors":factor_stats[-5:]
        },
        "gate":gate,
        "failedChecks":failed,
        "nextAction":"Freeze PCA transform and open fresh validation only." if not failed else "Close R2; validation and holdout remain sealed."
    }

    transport_out={
        "schema":"mgpt_arp21_fprv_exact_r2_transport_diagnostics_v1",
        "generation":"ARP21-FPRV-EXACT-R2",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "archiveMonths":MONTHS,
        "checksumRequired":True,
        "manifestFileCount":len(manifest),
        "symbols":transport_diag
    }

    cpath=OUT/"ARP21_FPRV_EXACT_R2_DEVELOPMENT_CLOSEOUT_20260915.json"
    tpath=OUT/"ARP21_FPRV_EXACT_R2_TRANSPORT_DIAGNOSTICS_20260915.json"
    cpath.write_text(json.dumps(closeout,indent=2)+"\n")
    tpath.write_text(json.dumps(transport_out,indent=2)+"\n")

    if not failed:
        freeze={
            "schema":"mgpt_arp21_fprv_exact_r2_pca_freeze_v1",
            "generation":"ARP21-FPRV-EXACT-R2",
            "createdAtUtc":datetime.now(timezone.utc).isoformat(),
            "sourceDevelopmentCloseout":cpath.name,
            "factorNames":FACTOR_NAMES,
            "columnMeans":X.mean(axis=0).tolist(),
            "eigenvectorsFirst3":top.tolist(),
            "eigenvaluesFirst3":vals[:3].tolist(),
            "explainedVarianceFirst3":explained.tolist(),
            "compositeVarianceWeights":vw.tolist(),
            "signRule":LOCK["methodAuthority"]["pcSignRule"],
            "immutableForFreshValidation":True
        }
        (OUT/"ARP21_FPRV_EXACT_R2_PCA_FREEZE_20260915.json").write_text(json.dumps(freeze,indent=2)+"\n")

    sums=[]
    for p in sorted(OUT.glob("ARP21_FPRV_EXACT_R2_*_20260915.json")):
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/"ARP21_FPRV_EXACT_R2_SHA256SUMS_20260915.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps({"status":closeout["status"],"failedChecks":failed,"development":closeout["development"]},indent=2))

if __name__=="__main__":
    main()
