#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, io, json, math, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import ElasticNet

ROOT=Path(__file__).resolve().parent
LOCK_PATH=ROOT/"ARP29_CTREND_B1_BINANCE_ADAPTATION_LOCK_20260916.json"
OUT=ROOT/"results"; OUT.mkdir(parents=True,exist_ok=True)
LOCK=json.loads(LOCK_PATH.read_text())
SYMS=LOCK["universe"]["symbols"]
FEATURES=(LOCK["methodAuthority"]["signalFamilies"]["momentum"]+
          LOCK["methodAuthority"]["signalFamilies"]["movingAverage"]+
          LOCK["methodAuthority"]["signalFamilies"]["volume"]+
          LOCK["methodAuthority"]["signalFamilies"]["volatility"])
DAY_MS=86400000
BASE="https://data.binance.vision/data/futures/um/monthly/klines"
FETCH_START=datetime.fromisoformat(LOCK["windows"]["archiveFetchStart"].replace("Z","+00:00"))
DEV_START=datetime.fromisoformat(LOCK["windows"]["development"]["start"].replace("Z","+00:00"))
DEV_END=datetime.fromisoformat(LOCK["windows"]["development"]["endExclusive"].replace("Z","+00:00"))
SESSION=requests.Session(); SESSION.headers.update({"User-Agent":"MarketsGPT-ARP29-CTREND-B1/1.0"})

def month_keys(a,b):
    y,m=a.year,a.month; out=[]
    while (y,m)<=(b.year,b.month):
        out.append(f"{y:04d}-{m:02d}"); m+=1
        if m==13: y+=1; m=1
    return out
MONTHS=month_keys(FETCH_START, DEV_END+timedelta(days=7))

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

def download_month(sym,ym):
    name=f"{sym}-1d-{ym}.zip"; url=f"{BASE}/{sym}/1d/{name}"
    blob=get_bytes(url)
    if blob is None:return {"symbol":sym,"ym":ym,"status":"404","rows":[]}
    chk=get_bytes(url+".CHECKSUM")
    if chk is None:raise RuntimeError(f"missing CHECKSUM {url}")
    exp=chk.decode("utf-8","replace").strip().split()[0].lower()
    got=hashlib.sha256(blob).hexdigest()
    if exp!=got:raise RuntimeError(f"checksum mismatch {name}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        ns=[n for n in z.namelist() if not n.endswith("/")]
        if len(ns)!=1:raise RuntimeError(f"unexpected members {name}: {ns}")
        raw=z.read(ns[0]).decode("utf-8-sig","replace")
    rows=[]
    for rec in csv.reader(io.StringIO(raw)):
        if not rec:continue
        try:t=int(float(rec[0]))
        except Exception:continue
        if t>10**14:t//=1000
        if len(rec)<8:continue
        try:o,h,l,c,qv=map(float,[rec[1],rec[2],rec[3],rec[4],rec[7]])
        except Exception:continue
        if min(o,h,l,c)>0 and qv>0 and all(map(math.isfinite,[o,h,l,c,qv])):
            rows.append((t,o,h,l,c,qv))
    return {"symbol":sym,"ym":ym,"status":"ok","sha256":got,"rows":rows}

def transport():
    jobs=[]
    with ThreadPoolExecutor(max_workers=32) as ex:
        for s in SYMS:
            for ym in MONTHS:jobs.append(ex.submit(download_month,s,ym))
        got=[f.result() for f in as_completed(jobs)]
    data={s:{} for s in SYMS}; manifest=[]; dup={s:0 for s in SYMS}
    for x in got:
        manifest.append({k:v for k,v in x.items() if k!="rows"})
        for row in x["rows"]:
            if row[0] in data[x["symbol"]]:dup[x["symbol"]]+=1
            data[x["symbol"]][row[0]]=row[1:]
    frames={}; diag=[]
    for s in SYMS:
        items=sorted(data[s].items())
        if items:
            idx=pd.to_datetime([t for t,_ in items],unit="ms",utc=True)
            arr=np.asarray([v for _,v in items],float)
            df=pd.DataFrame(arr,index=idx,columns=["open","high","low","close","qv"])
        else:
            df=pd.DataFrame(columns=["open","high","low","close","qv"])
        frames[s]=df
        ts=sorted(data[s]); gaps=sum(1 for a,b in zip(ts,ts[1:]) if b-a!=DAY_MS)
        diag.append({"symbol":s,"rows":len(ts),
                     "firstUtc":datetime.fromtimestamp(ts[0]/1000,timezone.utc).isoformat() if ts else None,
                     "lastUtc":datetime.fromtimestamp(ts[-1]/1000,timezone.utc).isoformat() if ts else None,
                     "duplicateRows":dup[s],"postFirstGapCount":gaps})
    return frames,manifest,diag

def ema(x,L):
    return x.ewm(alpha=1.0/(1.0+L),adjust=False,min_periods=L).mean()

def indicators(df):
    if df.empty:return pd.DataFrame(columns=FEATURES+["liq30"])
    c=df["close"];h=df["high"];l=df["low"];v=df["qv"]
    out=pd.DataFrame(index=df.index)
    delta=c.diff(); gain=delta.clip(lower=0); loss=(-delta).clip(lower=0)
    ag=gain.rolling(14,min_periods=14).mean(); al=loss.rolling(14,min_periods=14).mean()
    rs=ag/al.replace(0,np.nan); rsi=100-100/(1+rs)
    rsi=rsi.where(~((al==0)&(ag>0)),100.0).where(~((al==0)&(ag==0)),50.0)
    out["rsi14"]=rsi
    rmin=rsi.rolling(14,min_periods=14).min(); rmax=rsi.rolling(14,min_periods=14).max()
    out["stochrsi14"]=(rsi-rmin)/(rmax-rmin).replace(0,np.nan)
    ll=l.rolling(14,min_periods=14).min(); hh=h.rolling(14,min_periods=14).max()
    sk=(c-ll)/(hh-ll).replace(0,np.nan); out["stochk14"]=sk; out["stochd3"]=sk.rolling(3,min_periods=3).mean()
    tp=(h+l+c)/3.0; tpma=tp.rolling(20,min_periods=20).mean()
    mad=tp.rolling(20,min_periods=20).apply(lambda a: float(np.mean(np.abs(a-np.mean(a)))),raw=True)
    out["cci20"]=(tp-tpma)/(0.015*mad.replace(0,np.nan))
    for n in [3,5,10,20,50,100,200]:out[f"sma{n}"]=c.rolling(n,min_periods=n).mean()/c
    e12=ema(c,12);e26=ema(c,26); macd=(e12-e26)/e12.replace(0,np.nan)
    out["macd12_26"]=macd;out["macd_diff_signal9"]=macd-ema(macd,9)
    for n in [3,5,10,20,50,100,200]:out[f"volsma{n}"]=v.rolling(n,min_periods=n).mean()/v
    ve12=ema(v,12);ve26=ema(v,26);vm=(ve12-ve26)/ve12.replace(0,np.nan)
    out["volmacd12_26"]=vm;out["volmacd_diff_signal9"]=vm-ema(vm,9)
    denom=(h-l); mult=((c-l)-(h-c))/denom.replace(0,np.nan); ad=(mult.fillna(0))*v
    out["chaikin21"]=ad.rolling(21,min_periods=21).sum()/v.rolling(21,min_periods=21).sum().replace(0,np.nan)
    mid=c.rolling(20,min_periods=20).mean(); sd=c.rolling(20,min_periods=20).std(ddof=1)
    low=(mid-2*sd)/c; high=(mid+2*sd)/c; midscaled=mid/c
    out["boll_low20"]=low;out["boll_mid20"]=midscaled;out["boll_high20"]=high
    out["boll_width20"]=(high-low)/midscaled.replace(0,np.nan)
    out["liq30"]=v.rolling(30,min_periods=30).mean()
    return out.replace([np.inf,-np.inf],np.nan)

def rank_map(vals):
    s=pd.Series(vals,dtype=float)
    r=s.rank(method="average",ascending=True)
    n=int(r.notna().sum())
    if n<=1:return np.full(len(s),np.nan)
    return ((r-1)/(n-1)-0.5).to_numpy(float)

def ols1(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    xm=x.mean();ym=y.mean();vx=float(np.sum((x-xm)**2))
    if vx<=1e-18:return (ym,0.0)
    b=float(np.sum((x-xm)*(y-ym))/vx);a=float(ym-b*xm)
    return (a,b)

def build_weekly(frames,feats):
    start=FETCH_START+timedelta(days=(7-FETCH_START.weekday())%7)
    if start.weekday()!=0: raise RuntimeError("Monday alignment failure")
    end=DEV_END
    mondays=[];d=start
    while d<end:
        mondays.append(d);d+=timedelta(days=7)
    weeks=[]
    for mon in mondays:
        signal_day=pd.Timestamp(mon-timedelta(days=1))
        m0=pd.Timestamp(mon);m1=pd.Timestamp(mon+timedelta(days=7))
        rows=[]
        for si,s in enumerate(SYMS):
            df=frames[s];fi=feats[s]
            if signal_day not in fi.index or m0 not in df.index or m1 not in df.index:continue
            frow=fi.loc[signal_day]
            if not (float(frow.get("liq30",np.nan))>=LOCK["universe"]["minimumPrior30dMeanQuoteVolumeUsd"]):continue
            fv=np.asarray([frow.get(k,np.nan) for k in FEATURES],float)
            if not np.isfinite(fv).all():continue
            o0=float(df.loc[m0,"open"]);o1=float(df.loc[m1,"open"])
            if not (o0>0 and o1>0):continue
            ret=o1/o0-1.0
            if math.isfinite(ret):rows.append({"si":si,"sym":s,"raw":fv,"ret":ret})
        if len(rows)>=LOCK["universe"]["minimumEligiblePerWeek"]:
            Xraw=np.stack([r["raw"] for r in rows]); Xrank=np.column_stack([rank_map(Xraw[:,j]) for j in range(Xraw.shape[1])])
            y=np.asarray([r["ret"] for r in rows],float)
            if np.isfinite(Xrank).all():
                coeff=np.asarray([ols1(Xrank[:,j],y) for j in range(Xrank.shape[1])],float)
                weeks.append({"date":mon,"rows":rows,"Xrank":Xrank,"y":y,"coeff":coeff})
            else:weeks.append({"date":mon,"rows":rows,"Xrank":None,"y":None,"coeff":None})
        else:weeks.append({"date":mon,"rows":rows,"Xrank":None,"y":None,"coeff":None})
    return weeks

def add_univariate_forecasts(weeks):
    for i,w in enumerate(weeks):
        if w["coeff"] is None or i<52:continue
        prev=weeks[i-52:i]
        if any(x["coeff"] is None for x in prev):continue
        c=np.mean(np.stack([x["coeff"] for x in prev]),axis=0)
        w["F"]=c[:,0][None,:]+w["Xrank"]*c[:,1][None,:]
    return weeks

def fit_selector(trainF,trainY):
    X=np.asarray(trainF,float);y=np.asarray(trainY,float)
    mu=X.mean(axis=0);sd=X.std(axis=0,ddof=1);sd=np.where(sd>1e-12,sd,1.0);Xs=(X-mu)/sd
    l1=LOCK["elasticNet"]["l1Ratio"];yc=y-y.mean()
    lmax=float(np.max(np.abs(Xs.T@yc))/(len(y)*max(l1,1e-12)))
    if not math.isfinite(lmax) or lmax<=1e-12:return np.zeros(X.shape[1]),{"lambda":None,"aicc":None}
    grid=np.geomspace(lmax,lmax*LOCK["elasticNet"]["lambdaMaxFractionMin"],LOCK["elasticNet"]["lambdaGridCount"])
    best=None
    for a in grid:
        model=ElasticNet(alpha=float(a),l1_ratio=l1,fit_intercept=True,max_iter=20000,tol=1e-8,selection="cyclic")
        model.fit(Xs,y);pred=model.predict(Xs);rss=float(np.sum((y-pred)**2));k=int(np.sum(np.abs(model.coef_)>1e-10))+1;n=len(y)
        if rss<=0 or n<=k+1:continue
        aicc=n*math.log(rss/n)+2*k+(2*k*(k+1))/(n-k-1)
        cand=(aicc,float(a),model.coef_.copy(),k)
        if best is None or cand[0]<best[0]:best=cand
    if best is None:return np.zeros(X.shape[1]),{"lambda":None,"aicc":None}
    return best[2],{"lambda":best[1],"aicc":best[0],"k":best[3]}

def pf(a):
    a=np.asarray(a,float);p=float(a[a>0].sum());n=float(-a[a<0].sum());return p/n if n>0 else (999. if p>0 else 0.)
def maxdd(a):
    e=1.;pk=1.;dd=0.
    for r in a:e=max(1e-12,e*(1+float(r)));pk=max(pk,e);dd=max(dd,1-e/pk)
    return dd
def bootstrap(a,seed,it=10000,block=4):
    a=np.asarray(a,float);n=len(a)
    if not n:return {"probPositive":0.,"ci95Low":0.,"ci95High":0.}
    rng=np.random.default_rng(seed);z=np.empty(it)
    for k in range(it):
        vals=[]
        while len(vals)<n:
            st=int(rng.integers(0,n));take=min(block,n-len(vals));vals.extend(a[(st+np.arange(take))%n])
        z[k]=float(np.mean(vals))
    z.sort();return {"probPositive":float(np.mean(z>0)),"ci95Low":float(z[int(.025*(it-1))]),"ci95High":float(z[int(.975*(it-1))])}
def stats(a,dates,seed):
    a=np.asarray(a,float);m=float(a.mean()) if len(a) else 0.;sd=float(a.std(ddof=1)) if len(a)>1 else 0.;months={}
    for r,d in zip(a,dates):months[d.strftime("%Y-%m")]=months.get(d.strftime("%Y-%m"),0.)+float(r)
    return {"n":len(a),"meanWeeklyReturn":m,"annualizedMean":m*52,"annualizedSharpe":float(m/sd*math.sqrt(52)) if sd>0 else 0.,"profitFactor":pf(a),"maxDrawdown":maxdd(a),"positiveMonthFraction":sum(v>0 for v in months.values())/max(1,len(months)),"monthSums":months,"bootstrap":bootstrap(a,seed)}

def main():
    frames,manifest,diag=transport()
    feats={s:indicators(frames[s]) for s in SYMS}
    weeks=add_univariate_forecasts(build_weekly(frames,feats))
    dev=[i for i,w in enumerate(weeks) if DEV_START<=w["date"]<DEV_END]
    prevw=np.zeros(len(SYMS),float);gross=[];base=[];stress=[];extreme=[];dates=[];elig=[];active=0;selected_counts=[];zero_sel=0;contrib=np.zeros(len(SYMS));selector_meta=[]
    for i in dev:
        w=weeks[i];desired=np.zeros(len(SYMS),float);g=0.;seln=0
        if w.get("F") is not None and i>=52:
            hist=weeks[i-52:i]
            if all(x.get("F") is not None for x in hist):
                trainF=np.concatenate([x["F"] for x in hist],axis=0);trainY=np.concatenate([x["y"] for x in hist],axis=0)
                coef,meta=fit_selector(trainF,trainY);selected=np.where(coef>LOCK["elasticNet"]["positiveCoefficientThreshold"])[0];seln=len(selected);selector_meta.append({"date":w["date"].date().isoformat(),"selected":seln,**meta})
                if seln>0:
                    score=np.mean(w["F"][:,selected],axis=1);order=np.argsort(score);q=len(order)//5
                    if q>=1:
                        for ix in order[-q:]:desired[w["rows"][ix]["si"]]+=.5/q
                        for ix in order[:q]:desired[w["rows"][ix]["si"]]-=.5/q
                        active+=1
        if seln==0:zero_sel+=1
        selected_counts.append(seln);elig.append(len(w["rows"]))
        retmap={r["si"]:w["y"][k] for k,r in enumerate(w["rows"])}
        by=np.zeros(len(SYMS),float)
        for si,ww in enumerate(desired):
            if ww and si in retmap:by[si]=ww*retmap[si];g+=by[si]
        contrib+=by
        pos_prev=np.clip(prevw,0,None);pos_new=np.clip(desired,0,None);short_prev=np.clip(-prevw,0,None);short_new=np.clip(-desired,0,None)
        lt=float(np.abs(pos_new-pos_prev).sum());st=float(np.abs(short_new-short_prev).sum())
        c=LOCK["costs"];gross.append(g);base.append(g-c["baseLongOneWayRate"]*lt-c["baseShortOneWayRate"]*st);stress.append(g-c["stressLongOneWayRate"]*lt-c["stressShortOneWayRate"]*st);extreme.append(g-c["extremeLongOneWayRate"]*lt-c["extremeShortOneWayRate"]*st);dates.append(w["date"]);prevw=desired
    if dates:
        pos_prev=np.clip(prevw,0,None);short_prev=np.clip(-prevw,0,None);c=LOCK["costs"];base[-1]-=c["baseLongOneWayRate"]*float(pos_prev.sum())+c["baseShortOneWayRate"]*float(short_prev.sum());stress[-1]-=c["stressLongOneWayRate"]*float(pos_prev.sum())+c["stressShortOneWayRate"]*float(short_prev.sum());extreme[-1]-=c["extremeLongOneWayRate"]*float(pos_prev.sum())+c["extremeShortOneWayRate"]*float(short_prev.sum())
    gs=stats(gross,dates,29110);bs=stats(base,dates,29111);ss=stats(stress,dates,29112);xs=stats(extreme,dates,29113)
    n=len(base);half=n//2;fh=float(np.sum(base[:half]));sh=float(np.sum(base[half:]));pos=contrib[contrib>0];dom=float(pos.max()/pos.sum()) if len(pos) and pos.sum()>0 else 0.;coverage=active/max(1,n);avgsel=float(np.mean(selected_counts)) if selected_counts else 0.;zerof=zero_sel/max(1,n);avgel=float(np.mean(elig)) if elig else 0.
    g=LOCK["developmentGate"];gate={
      "minimumEvaluatedWeeks":n>=g["minimumEvaluatedWeeks"],"minimumCoverageFraction":coverage>=g["minimumCoverageFraction"],"averageEligibleSymbolsMin":avgel>=g["averageEligibleSymbolsMin"],"baseNetMeanWeeklyReturnMin":bs["meanWeeklyReturn"]>=g["baseNetMeanWeeklyReturnMin"],"baseNetAnnualizedSharpeMin":bs["annualizedSharpe"]>=g["baseNetAnnualizedSharpeMin"],"baseNetProfitFactorMin":bs["profitFactor"]>=g["baseNetProfitFactorMin"],"stressNetMeanWeeklyReturnMin":ss["meanWeeklyReturn"]>=g["stressNetMeanWeeklyReturnMin"],"positiveMonthFractionMin":bs["positiveMonthFraction"]>=g["positiveMonthFractionMin"],"firstHalfNetReturnPositive":fh>0,"secondHalfNetReturnPositive":sh>0,"bootstrapProbabilityPositiveMin":bs["bootstrap"]["probPositive"]>=g["bootstrapProbabilityPositiveMin"],"bootstrapCi95LowMin":bs["bootstrap"]["ci95Low"]>=g["bootstrapCi95LowMin"],"maxDrawdownMax":bs["maxDrawdown"]<=g["maxDrawdownMax"],"singleSymbolPositiveContributionMax":dom<=g["singleSymbolPositiveContributionMax"],"averageSelectedForecastCountMin":avgsel>=g["averageSelectedForecastCountMin"],"zeroSelectedForecastWeekFractionMax":zerof<=g["zeroSelectedForecastWeekFractionMax"]
    };gate={k:bool(v) for k,v in gate.items()};failed=[k for k,v in gate.items() if not v]
    close={"schema":"mgpt_arp29_ctrend_b1_development_closeout_v1","generation":"ARP29-CTREND-B1","generatedAtUtc":datetime.now(timezone.utc).isoformat(),"status":"PASS_DEVELOPMENT_GATE_FUNDING_REPLAY_REQUIRED" if not failed else "REJECT_DEVELOPMENT_GATE","productionAuthority":False,"r15MutationAllowed":False,"validationAuthorized":False,"sealedHoldoutOpened":False,"lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),"development":{"evaluatedWeeks":n,"activeWeeks":active,"coverageFraction":coverage,"averageEligibleSymbols":avgel,"averageSelectedForecastCount":avgsel,"zeroSelectedForecastWeekFraction":zerof,"firstHalfNetReturn":fh,"secondHalfNetReturn":sh,"singleSymbolPositiveContribution":dom,"gross":gs,"baseCost":bs,"stressCost":ss,"extremeCost":xs,"symbolGrossContributions":{s:float(contrib[i]) for i,s in enumerate(SYMS)},"selectorMeta":selector_meta},"gate":gate,"failedChecks":failed,"nextAction":"Run exact funding-cashflow replay with frozen signals before external replication validation." if not failed else "Close ARP29-CTREND-B1; no funding replay, validation, or holdout."}
    trans={"schema":"mgpt_arp29_ctrend_b1_transport_diagnostics_v1","generation":"ARP29-CTREND-B1","generatedAtUtc":datetime.now(timezone.utc).isoformat(),"archiveMonths":MONTHS,"checksumRequired":True,"manifestCount":len(manifest),"symbols":diag}
    cp=OUT/"ARP29_CTREND_B1_DEVELOPMENT_CLOSEOUT_20260916.json";tp=OUT/"ARP29_CTREND_B1_TRANSPORT_DIAGNOSTICS_20260916.json";cp.write_text(json.dumps(close,indent=2)+"\n");tp.write_text(json.dumps(trans,indent=2)+"\n")
    sums=[]
    for p in sorted(OUT.glob("ARP29_CTREND_B1_*_20260916.json")):sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/"ARP29_CTREND_B1_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps({"status":close["status"],"failedChecks":failed,"development":{k:v for k,v in close["development"].items() if k not in ("symbolGrossContributions","selectorMeta")}},indent=2))

if __name__=="__main__":main()
