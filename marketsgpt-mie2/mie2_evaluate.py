#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
MIE1=REPO/"marketsgpt-mie1"
LOCK_PATH=ROOT/"MGPT_MIE2_POSITIONING_OI_LOCK_20260916.json"
SPEC_PATH=ROOT/"MGPT_MIE2_IMPLEMENTATION_SPEC_V1_20260916.json"
FREEZE_PATH=ROOT/"results"/"MGPT_MIE2_DATA_FREEZE_STATUS_20260916.json"
MIE1_LOCK_PATH=MIE1/"MGPT_MIE1_MICROSTRUCTURE_LOCK_20260916.json"
MIE1_FREEZE_PATH=MIE1/"results"/"MGPT_MIE1_DATA_FREEZE_STATUS_20260916.json"
OUT=ROOT/"results"

LOCK=json.loads(LOCK_PATH.read_text())
SPEC=json.loads(SPEC_PATH.read_text())
MIE1_LOCK=json.loads(MIE1_LOCK_PATH.read_text())
FIT_START=pd.Timestamp(LOCK["windows"]["modelFit"]["start"])
FIT_END=pd.Timestamp(LOCK["windows"]["modelFit"]["endExclusive"])
DEV_START=pd.Timestamp(LOCK["windows"]["developmentScreen"]["start"])
DEV_END=pd.Timestamp(LOCK["windows"]["developmentScreen"]["endExclusive"])

NUMERIC=list(LOCK["causalPositioningFeatures"])
CATEGORICAL=list(LOCK["categoricalFeatures"])

def read_csv(path:Path)->pd.DataFrame:
    x=pd.read_csv(path)
    x["timestamp"]=pd.to_datetime(x["timestamp"],utc=True,format="mixed")
    if "metric_time" in x.columns:
        x["metric_time"]=pd.to_datetime(x["metric_time"],utc=True,format="mixed")
    for c in x.columns:
        if c not in {"timestamp","metric_time","symbol"}:
            x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def wilder_atr(x:pd.DataFrame,n:int=14)->pd.Series:
    pc=x["close"].shift(1)
    tr=pd.concat([(x.high-x.low).abs(),(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n,adjust=False,min_periods=n).mean()

def prior_z(s:pd.Series,n:int=24)->pd.Series:
    mu=s.shift(1).rolling(n,min_periods=n).mean()
    sd=s.shift(1).rolling(n,min_periods=n).std(ddof=1)
    return (s-mu)/sd.replace(0,np.nan)

def group_for_symbol(s:str)->str:
    return "LARGE" if s in set(LOCK["symbolGroup"]["LARGE"]) else "ALT"

def add_positioning_features(p:pd.DataFrame)->pd.DataFrame:
    x=p.copy()
    oi=x["sum_open_interest"].astype(float)
    oiv=x["sum_open_interest_value"].astype(float)
    for n in [1,3,6,24]:
        x[f"oiContractsLogChange{n}h"]=np.log(oi/oi.shift(n))
        x[f"oiValueLogChange{n}h"]=np.log(oiv/oiv.shift(n))
    x["oiContractsZ24h"]=prior_z(oi,24)
    x["oiValueZ24h"]=prior_z(oiv,24)

    x["topAccountLogRatio"]=np.log(x["count_toptrader_long_short_ratio"].astype(float))
    x["topPositionLogRatio"]=np.log(x["sum_toptrader_long_short_ratio"].astype(float))
    x["globalAccountLogRatio"]=np.log(x["count_long_short_ratio"].astype(float))
    x["takerLogRatio"]=np.log(x["sum_taker_long_short_vol_ratio"].astype(float))

    for base,prefix,ns in [
        ("topAccountLogRatio","topAccountDelta",[1,3,6,24]),
        ("topPositionLogRatio","topPositionDelta",[1,3,6,24]),
        ("globalAccountLogRatio","globalAccountDelta",[1,3,6,24]),
        ("takerLogRatio","takerRatioDelta",[1,3,6]),
    ]:
        for n in ns:
            x[f"{prefix}{n}h"]=x[base]-x[base].shift(n)

    x["topVsGlobalCrowding"]=x["topAccountLogRatio"]-x["globalAccountLogRatio"]
    x["positionVsAccountDivergence"]=x["topPositionLogRatio"]-x["topAccountLogRatio"]
    x["topPositionVsGlobal"]=x["topPositionLogRatio"]-x["globalAccountLogRatio"]
    return x

def funding_state(f:pd.DataFrame,decision:pd.Timestamp)->float:
    times=f["timestamp"].astype("int64").to_numpy()
    pos=int(np.searchsorted(times,decision.value,side="right")-1)
    if pos<0:
        return math.nan
    return float(f.iloc[pos]["funding_rate"])

def funding_cashflow_r(f:pd.DataFrame,entry_time:pd.Timestamp,funding_end:pd.Timestamp,sign:int,entry:float,risk:float)->float:
    z=f[(f.timestamp>entry_time)&(f.timestamp<=funding_end)]
    if z.empty:
        return 0.0
    return float((-sign*z.funding_rate.astype(float).sum()*entry)/risk)

def simulate_direction(c:pd.DataFrame,f:pd.DataFrame,i:int,direction:str,friction_bps:float)->dict:
    sign=1 if direction=="LONG" else -1
    entry_i=i+1
    entry_time=pd.Timestamp(c.iloc[entry_i]["timestamp"])
    entry=float(c.iloc[entry_i]["open"])
    atr=float(c.iloc[i]["atr14"])
    risk=atr
    stop=entry-sign*risk
    tp2=entry+sign*1.5*risk
    hold=int(MIE1_LOCK["candidateEngine"]["maxHoldingBars"])
    last_i=entry_i+hold-1
    exit_price=float(c.iloc[last_i]["close"])
    exit_i=last_i
    exit_type="TIMEOUT"
    mfe=0.0
    mae=0.0
    for j in range(entry_i,last_i+1):
        h=float(c.iloc[j]["high"]); l=float(c.iloc[j]["low"])
        fav=(h-entry)/risk if sign==1 else (entry-l)/risk
        adv=(entry-l)/risk if sign==1 else (h-entry)/risk
        mfe=max(mfe,fav); mae=max(mae,adv)
        if (l<=stop if sign==1 else h>=stop):
            exit_price=stop; exit_i=j; exit_type="STOP"; break
        if (h>=tp2 if sign==1 else l<=tp2):
            exit_price=tp2; exit_i=j; exit_type="TP2"; break
    exit_bar_time=pd.Timestamp(c.iloc[exit_i]["timestamp"])
    funding_end=exit_bar_time+pd.Timedelta(hours=1) if exit_type=="TIMEOUT" else exit_bar_time
    gross=sign*(exit_price-entry)/risk
    fr=(float(friction_bps)/10000.0)*(abs(entry)+abs(exit_price))/risk
    fund=funding_cashflow_r(f,entry_time,funding_end,sign,entry,risk)
    return {
        "netR":float(gross-fr+fund),"grossR":float(gross),"fundingR":float(fund),
        "frictionR":float(fr),"exitType":exit_type,"mfeR":float(mfe),"maeR":float(mae),
        "goodOpportunity":int(exit_type=="TP2")
    }

def split_for_signal(ts:pd.Timestamp)->str|None:
    if FIT_START<=ts<FIT_END: return "FIT"
    if DEV_START<=ts<DEV_END: return "DEVELOPMENT"
    return None

def full_horizon_inside(c:pd.DataFrame,i:int,split:str)->bool:
    entry_i=i+1
    last_i=entry_i+int(MIE1_LOCK["candidateEngine"]["maxHoldingBars"])-1
    if last_i>=len(c): return False
    end=FIT_END if split=="FIT" else DEV_END
    return pd.Timestamp(c.iloc[last_i]["timestamp"])+pd.Timedelta(hours=1)<end

def validate_inputs():
    fr=json.loads(FREEZE_PATH.read_text())
    if fr.get("status")!="PASS_DATA_FREEZE":
        raise RuntimeError("MIE2 data freeze not PASS")
    lock_sha=hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    if fr.get("lockSha256")!=lock_sha:
        raise RuntimeError("MIE2 freeze lock hash mismatch")
    pos_manifest={r["symbol"]:r for r in fr["symbols"]}
    for s in LOCK["universe"]:
        p=ROOT/"data"/"positioning_hourly"/f"{s}_1h.csv"
        if hashlib.sha256(p.read_bytes()).hexdigest()!=pos_manifest[s]["sha256"]:
            raise RuntimeError(f"MIE2 positioning hash mismatch {s}")

    mf=json.loads(MIE1_FREEZE_PATH.read_text())
    if mf.get("status")!="PASS_DATA_FREEZE":
        raise RuntimeError("Inherited MIE1 freeze not PASS")
    mm={r["symbol"]:r for r in mf["symbols"]}
    for s in LOCK["universe"]:
        for key,subdir,fn in [
            ("contract","contract",f"{s}_1h.csv"),
            ("premium","premium",f"{s}_1h.csv"),
            ("funding","funding",f"{s}.csv")
        ]:
            p=MIE1/"data"/subdir/fn
            if hashlib.sha256(p.read_bytes()).hexdigest()!=mm[s][key]["sha256"]:
                raise RuntimeError(f"inherited MIE1 hash mismatch {s} {key}")
    return fr,mf,lock_sha

def build_symbol_rows(symbol:str)->list[dict]:
    c=read_csv(MIE1/"data"/"contract"/f"{symbol}_1h.csv")
    p=read_csv(MIE1/"data"/"premium"/f"{symbol}_1h.csv")
    f=read_csv(MIE1/"data"/"funding"/f"{symbol}.csv")
    pos=add_positioning_features(read_csv(ROOT/"data"/"positioning_hourly"/f"{symbol}_1h.csv"))

    c["atr14"]=wilder_atr(c,14)
    for n in [1,3,6,12,24]:
        c[f"priceRet{n}h"]=c["close"]/c["close"].shift(n)-1.0
    prem=p[["timestamp","close"]].rename(columns={"close":"premiumClose"})
    c=c.merge(prem,on="timestamp",how="left",validate="one_to_one")
    pos_idx=pos.set_index("timestamp",drop=False)
    group=group_for_symbol(symbol)
    rows=[]

    for i in range(50,len(c)-25):
        signal_ts=pd.Timestamp(c.iloc[i]["timestamp"])
        split=split_for_signal(signal_ts)
        if split is None or not full_horizon_inside(c,i,split):
            continue
        decision=signal_ts+pd.Timedelta(hours=1)
        if pd.Timestamp(c.iloc[i+1]["timestamp"])!=decision:
            continue
        if decision not in pos_idx.index:
            continue
        pr=pos_idx.loc[decision]
        if isinstance(pr,pd.DataFrame):
            raise RuntimeError(f"duplicate hourly positioning after freeze {symbol} {decision}")
        if pd.isna(pr["metric_time"]) or pd.Timestamp(pr["metric_time"])>decision:
            continue

        rr=c.iloc[i]
        needed_price=["atr14","priceRet1h","priceRet3h","priceRet6h","priceRet12h","priceRet24h","premiumClose"]
        needed_pos=[
            "oiContractsLogChange1h","oiContractsLogChange3h","oiContractsLogChange6h","oiContractsLogChange24h",
            "oiValueLogChange1h","oiValueLogChange3h","oiValueLogChange6h","oiValueLogChange24h",
            "oiContractsZ24h","oiValueZ24h","topAccountLogRatio","topPositionLogRatio","globalAccountLogRatio","takerLogRatio",
            "topAccountDelta1h","topAccountDelta3h","topAccountDelta6h","topAccountDelta24h",
            "topPositionDelta1h","topPositionDelta3h","topPositionDelta6h","topPositionDelta24h",
            "globalAccountDelta1h","globalAccountDelta3h","globalAccountDelta6h","globalAccountDelta24h",
            "takerRatioDelta1h","takerRatioDelta3h","takerRatioDelta6h",
            "topVsGlobalCrowding","positionVsAccountDivergence","topPositionVsGlobal"
        ]
        if any(pd.isna(rr[k]) for k in needed_price) or any(pd.isna(pr[k]) for k in needed_pos):
            continue
        latest_funding=funding_state(f,decision)
        if not math.isfinite(latest_funding):
            continue

        base_long=simulate_direction(c,f,i,"LONG",float(MIE1_LOCK["costAndFunding"]["oneWayBaseBps"]))
        base_short=simulate_direction(c,f,i,"SHORT",float(MIE1_LOCK["costAndFunding"]["oneWayBaseBps"]))
        stress_long=simulate_direction(c,f,i,"LONG",float(MIE1_LOCK["costAndFunding"]["oneWayStressBps"]))
        stress_short=simulate_direction(c,f,i,"SHORT",float(MIE1_LOCK["costAndFunding"]["oneWayStressBps"]))

        hour=decision.hour+decision.minute/60.0
        wd=decision.weekday()
        row={
            "split":split,"signalUtc":signal_ts.isoformat(),"entryUtc":decision.isoformat(),
            "symbol":symbol,"symbolGroup":group,
            **{k:float(pr[k]) for k in needed_pos},
            "priceRet1h":float(rr["priceRet1h"]),
            "priceRet3h":float(rr["priceRet3h"]),
            "priceRet6h":float(rr["priceRet6h"]),
            "priceRet12h":float(rr["priceRet12h"]),
            "priceRet24h":float(rr["priceRet24h"]),
            "priceOiInteraction3h":float(rr["priceRet3h"])*float(pr["oiContractsLogChange3h"]),
            "priceOiInteraction6h":float(rr["priceRet6h"])*float(pr["oiContractsLogChange6h"]),
            "priceOiInteraction24h":float(rr["priceRet24h"])*float(pr["oiContractsLogChange24h"]),
            "premiumClose":float(rr["premiumClose"]),
            "latestFundingRate":float(latest_funding),
            "hourSin":math.sin(2*math.pi*hour/24.0),
            "hourCos":math.cos(2*math.pi*hour/24.0),
            "weekdaySin":math.sin(2*math.pi*wd/7.0),
            "weekdayCos":math.cos(2*math.pi*wd/7.0),
            "longNetR":base_long["netR"],"shortNetR":base_short["netR"],
            "longStressNetR":stress_long["netR"],"shortStressNetR":stress_short["netR"],
            "longGood":base_long["goodOpportunity"],"shortGood":base_short["goodOpportunity"],
            "longExitType":base_long["exitType"],"shortExitType":base_short["exitType"],
            "targetSpreadR":float(base_long["netR"]-base_short["netR"]),
            "longBetter":int(base_long["netR"]>base_short["netR"]) if base_long["netR"]!=base_short["netR"] else -1,
            "tradableOpportunity":int(max(base_long["netR"],base_short["netR"])>0)
        }
        rows.append(row)
    return rows

def build_table()->pd.DataFrame:
    rows=[]
    for s in LOCK["universe"]:
        rows.extend(build_symbol_rows(s))
    return pd.DataFrame(rows)

def preprocessor():
    num=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([("num",num,NUMERIC),("cat",cat,CATEGORICAL)],remainder="drop",sparse_threshold=0.0)

def fit_score(fit:pd.DataFrame,dev:pd.DataFrame):
    pre=preprocessor()
    xf=pre.fit_transform(fit[NUMERIC+CATEGORICAL])
    xd=pre.transform(dev[NUMERIC+CATEGORICAL])
    tie=fit["longBetter"]!=-1
    xfd=xf[tie.to_numpy()]
    yd=fit.loc[tie,"longBetter"].astype(int).to_numpy()
    ys=fit.loc[tie,"targetSpreadR"].astype(float).to_numpy()
    yt=fit["tradableOpportunity"].astype(int).to_numpy()
    if len(np.unique(yd))<2 or len(np.unique(yt))<2:
        raise RuntimeError("MIE2 fit target lacks both classes")

    lc=LOCK["model"]["longBetterLogit"]
    logit=LogisticRegression(C=float(lc["C"]),class_weight=lc["class_weight"],max_iter=int(lc["max_iter"]),solver=lc["solver"],random_state=int(lc["random_state"]))
    hc=LOCK["model"]["longBetterHgb"]
    hgb=HistGradientBoostingClassifier(loss=hc["loss"],learning_rate=float(hc["learning_rate"]),max_iter=int(hc["max_iter"]),max_depth=int(hc["max_depth"]),min_samples_leaf=int(hc["min_samples_leaf"]),l2_regularization=float(hc["l2_regularization"]),early_stopping=bool(hc["early_stopping"]),random_state=int(hc["random_state"]))
    sc=LOCK["model"]["spreadHgb"]
    spread=HistGradientBoostingRegressor(loss=sc["loss"],learning_rate=float(sc["learning_rate"]),max_iter=int(sc["max_iter"]),max_depth=int(sc["max_depth"]),min_samples_leaf=int(sc["min_samples_leaf"]),l2_regularization=float(sc["l2_regularization"]),early_stopping=bool(sc["early_stopping"]),random_state=int(sc["random_state"]))
    tc=LOCK["model"]["tradableHgb"]
    trad=HistGradientBoostingClassifier(loss=tc["loss"],learning_rate=float(tc["learning_rate"]),max_iter=int(tc["max_iter"]),max_depth=int(tc["max_depth"]),min_samples_leaf=int(tc["min_samples_leaf"]),l2_regularization=float(tc["l2_regularization"]),early_stopping=bool(tc["early_stopping"]),random_state=int(tc["random_state"]))

    logit.fit(xfd,yd); hgb.fit(xfd,yd); spread.fit(xfd,ys); trad.fit(xf,yt)

    z=dev.copy()
    z["pLongLogit"]=logit.predict_proba(xd)[:,1]
    z["pLongHgb"]=hgb.predict_proba(xd)[:,1]
    z["predSpreadR"]=spread.predict(xd)
    z["pTradableHgb"]=trad.predict_proba(xd)[:,1]
    z["directionalScore"]=0.35*(2*z.pLongLogit-1)+0.35*(2*z.pLongHgb-1)+0.30*np.tanh(z.predSpreadR/1.25)
    z["chosenDirection"]=np.where(z.directionalScore>=0,"LONG","SHORT")
    z["strength"]=np.abs(z.directionalScore)*(0.50+0.50*z.pTradableHgb)
    return z,{"preprocessor":pre,"longBetterLogit":logit,"longBetterHgb":hgb,"spreadHgb":spread,"tradableHgb":trad}

def select(dev:pd.DataFrame)->pd.DataFrame:
    z=dev.copy()
    z["selected"]=False
    for ts,g in z.groupby("entryUtc",sort=True):
        idxs=g.sort_values(["strength","symbol"],ascending=[False,True]).head(int(LOCK["selection"]["maxTradesPerHour"])).index
        z.loc[idxs,"selected"]=True
    z["chosenNetR"]=np.where(z.chosenDirection=="LONG",z.longNetR,z.shortNetR)
    z["chosenStressNetR"]=np.where(z.chosenDirection=="LONG",z.longStressNetR,z.shortStressNetR)
    z["oppositeNetR"]=np.where(z.chosenDirection=="LONG",z.shortNetR,z.longNetR)
    z["chosenGood"]=np.where(z.chosenDirection=="LONG",z.longGood,z.shortGood).astype(int)
    z["pairMeanR"]=0.5*(z.longNetR+z.shortNetR)
    z["chosenDeltaVsPairMeanR"]=z.chosenNetR-z.pairMeanR
    z["directionCorrect"]=(z.chosenNetR>=z.oppositeNetR).astype(int)
    return z

def pf(a)->float:
    x=np.asarray(a,float); p=float(x[x>0].sum()); n=float(-x[x<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)

def maxdd(a)->float:
    eq=peak=dd=0.0
    for v in a:
        eq+=float(v); peak=max(peak,eq); dd=max(dd,peak-eq)
    return float(dd)

def auc(y,s)->float:
    yy=pd.Series(y)
    return float(roc_auc_score(yy,s)) if yy.nunique()==2 else math.nan

def spearman(a,b)->float:
    ra=pd.Series(a).rank(method="average").to_numpy(float)
    rb=pd.Series(b).rank(method="average").to_numpy(float)
    if np.std(ra)==0 or np.std(rb)==0: return math.nan
    return float(np.corrcoef(ra,rb)[0,1])

def metrics(dev:pd.DataFrame)->dict:
    sel=dev[dev.selected].copy()
    sym=sel.groupby("symbol").chosenNetR.mean() if len(sel) else pd.Series(dtype=float)
    st=sel.groupby("symbol").chosenNetR.sum() if len(sel) else pd.Series(dtype=float)
    pos=st[st>0]; conc=float(pos.max()/pos.sum()) if len(pos) else 0.0
    s=sel.sort_values("signalUtc").reset_index(drop=True); half=len(s)//2
    mon=sel.assign(month=pd.to_datetime(sel.signalUtc,utc=True).dt.strftime("%Y-%m")).groupby("month").chosenNetR.sum() if len(sel) else pd.Series(dtype=float)
    all_good=int(dev.longGood.sum()+dev.shortGood.sum()); sel_good=int(sel.chosenGood.sum()) if len(sel) else 0
    dclass=dev[dev.longBetter!=-1]
    return {
        "developmentSymbolHours":int(len(dev)),
        "selectedTrades":int(len(sel)),
        "selectionCoverage":float(len(sel)/max(1,len(dev))),
        "activeSymbols":int(sel.symbol.nunique()) if len(sel) else 0,
        "activeDirections":sorted(sel.chosenDirection.unique().tolist()) if len(sel) else [],
        "netExpectancyR":float(sel.chosenNetR.mean()) if len(sel) else 0.0,
        "stressNetExpectancyR":float(sel.chosenStressNetR.mean()) if len(sel) else 0.0,
        "profitFactor":pf(sel.chosenNetR) if len(sel) else 0.0,
        "hitRate":float((sel.chosenNetR>0).mean()) if len(sel) else 0.0,
        "maxDrawdownR":maxdd(s.chosenNetR.to_numpy(float)) if len(sel) else 0.0,
        "pairDirectionAccuracy":float(sel.directionCorrect.mean()) if len(sel) else 0.0,
        "opportunityRecall":float(sel_good/max(1,all_good)),
        "opportunityPrecision":float(sel_good/max(1,len(sel))),
        "positiveSymbolFraction":float((sym>0).mean()) if len(sym) else 0.0,
        "positiveMonthFraction":float((mon>0).mean()) if len(mon) else 0.0,
        "firstHalfNetExpectancyR":float(s.iloc[:half].chosenNetR.mean()) if half else 0.0,
        "secondHalfNetExpectancyR":float(s.iloc[half:].chosenNetR.mean()) if len(s)-half else 0.0,
        "longBetterLogitAuc":auc(dclass.longBetter,dclass.pLongLogit),
        "longBetterHgbAuc":auc(dclass.longBetter,dclass.pLongHgb),
        "tradableHgbAuc":auc(dev.tradableOpportunity,dev.pTradableHgb),
        "spreadSpearman":spearman(dev.targetSpreadR,dev.predSpreadR),
        "selectedDeltaVsPairMeanR":float(sel.chosenDeltaVsPairMeanR.mean()) if len(sel) else 0.0,
        "portfolioClockBaselineR":float(dev.pairMeanR.mean()),
        "singleSymbolPositiveContribution":conc,
        "symbolExpectancyR":{str(k):float(v) for k,v in sym.items()}
    }

def circ(n,rng,block):
    out=[]
    while len(out)<n:
        st=int(rng.integers(0,n)); take=min(block,n-len(out))
        out.extend(((st+np.arange(take))%n).tolist())
    return np.asarray(out,int)

def bootstrap(dev:pd.DataFrame)->dict:
    cfg=LOCK["bootstrap"]; rng=np.random.default_rng(int(cfg["seed"])); iters=int(cfg["iterations"]); block=int(cfg["blockLengthHours"])
    packed={}
    for sym,g in dev.groupby("symbol"):
        z=g.sort_values("entryUtc").reset_index(drop=True)
        packed[sym]={"n":len(z),"sel":z.selected.to_numpy(bool),"net":z.chosenNetR.to_numpy(float),"delta":z.chosenDeltaVsPairMeanR.to_numpy(float)}
    absx=np.zeros(iters); delx=np.zeros(iters)
    syms=sorted(packed)
    for it in range(iters):
        sv=[]; dv=[]
        for sym in syms:
            p=packed[sym]; idx=circ(p["n"],rng,block); mask=p["sel"][idx]
            if mask.any():
                sv.extend(p["net"][idx][mask].tolist()); dv.extend(p["delta"][idx][mask].tolist())
        absx[it]=float(np.mean(sv)) if sv else 0.0
        delx[it]=float(np.mean(dv)) if dv else 0.0
    return {
        "method":cfg["method"],"iterations":iters,"seed":int(cfg["seed"]),"blockLengthHours":block,
        "probabilityPositiveNetExpectancy":float(np.mean(absx>0)),
        "ci95NetExpectancyR":[float(np.quantile(absx,.025)),float(np.quantile(absx,.975))],
        "probabilityPositiveDeltaVsPairMean":float(np.mean(delx>0)),
        "ci95DeltaVsPairMeanR":[float(np.quantile(delx,.025)),float(np.quantile(delx,.975))]
    }

def checks(fit_rows:int,m:dict,b:dict)->dict:
    g=LOCK["developmentGate"]
    return {
        "minimumFitSymbolHours":bool(fit_rows>=int(g["minimumFitSymbolHours"])),
        "minimumDevelopmentSymbolHours":bool(m["developmentSymbolHours"]>=int(g["minimumDevelopmentSymbolHours"])),
        "minimumSelectedTrades":bool(m["selectedTrades"]>=int(g["minimumSelectedTrades"])),
        "activeSymbolsRequired":bool(m["activeSymbols"]>=int(g["activeSymbolsRequired"])),
        "bothDirectionsActive":bool(set(m["activeDirections"])=={"LONG","SHORT"}),
        "netExpectancyRMin":bool(m["netExpectancyR"]>=float(g["netExpectancyRMin"])),
        "stressNetExpectancyRMin":bool(m["stressNetExpectancyR"]>=float(g["stressNetExpectancyRMin"])),
        "profitFactorMin":bool(m["profitFactor"]>=float(g["profitFactorMin"])),
        "pairDirectionAccuracyMin":bool(m["pairDirectionAccuracy"]>=float(g["pairDirectionAccuracyMin"])),
        "opportunityRecallMin":bool(m["opportunityRecall"]>=float(g["opportunityRecallMin"])),
        "opportunityPrecisionMin":bool(m["opportunityPrecision"]>=float(g["opportunityPrecisionMin"])),
        "positiveSymbolFractionMin":bool(m["positiveSymbolFraction"]>=float(g["positiveSymbolFractionMin"])),
        "positiveMonthFractionMin":bool(m["positiveMonthFraction"]>=float(g["positiveMonthFractionMin"])),
        "firstHalfPositive":bool(m["firstHalfNetExpectancyR"]>0),
        "secondHalfPositive":bool(m["secondHalfNetExpectancyR"]>0),
        "longBetterLogitAucMin":bool(math.isfinite(m["longBetterLogitAuc"]) and m["longBetterLogitAuc"]>=float(g["longBetterLogitAucMin"])),
        "longBetterHgbAucMin":bool(math.isfinite(m["longBetterHgbAuc"]) and m["longBetterHgbAuc"]>=float(g["longBetterHgbAucMin"])),
        "tradableHgbAucMin":bool(math.isfinite(m["tradableHgbAuc"]) and m["tradableHgbAuc"]>=float(g["tradableHgbAucMin"])),
        "spreadSpearmanMin":bool(math.isfinite(m["spreadSpearman"]) and m["spreadSpearman"]>=float(g["spreadSpearmanMin"])),
        "bootstrapProbabilityPositiveNetMin":bool(b["probabilityPositiveNetExpectancy"]>=float(g["bootstrapProbabilityPositiveNetMin"])),
        "bootstrapProbabilityPositiveDeltaVsPairMeanMin":bool(b["probabilityPositiveDeltaVsPairMean"]>=float(g["bootstrapProbabilityPositiveDeltaVsPairMeanMin"])),
        "singleSymbolPositiveContributionMax":bool(m["singleSymbolPositiveContribution"]<=float(g["singleSymbolPositiveContributionMax"]))
    }

def main():
    fr,mf,lock_sha=validate_inputs()
    table=build_table()
    if table.empty: raise RuntimeError("MIE2 table empty")
    fit=table[table.split=="FIT"].copy().reset_index(drop=True)
    dev=table[table.split=="DEVELOPMENT"].copy().reset_index(drop=True)
    if len(fit)<1000 or len(dev)<1000: raise RuntimeError(f"insufficient MIE2 rows fit={len(fit)} dev={len(dev)}")
    devs,models=fit_score(fit,dev)
    devs=select(devs)
    m=metrics(devs); b=bootstrap(devs); c=checks(len(fit),m,b)
    passed=all(c.values())
    status="PASS_MIE2_DEVELOPMENT_REPLICATION_REQUIRED" if passed else "REJECT_MIE2_DEVELOPMENT_GATE"

    model_path=OUT/"MGPT_MIE2_FROZEN_MODEL_20260916.joblib"
    joblib.dump({**models,"lockSha256":lock_sha},model_path)
    summary={
        "schema":"mgpt_mie2_model_summary_v1","fitSymbolHours":int(len(fit)),"developmentSymbolHours":int(len(devs)),
        "numericFeatures":NUMERIC,"categoricalFeatures":CATEGORICAL,
        "modelArtifactSha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "scoreSemantics":"Pairwise ranking scores only; no calibrated probability claim."
    }
    sp=OUT/"MGPT_MIE2_MODEL_SUMMARY_20260916.json"; sp.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")

    close={
        "schema":"mgpt_mie2_development_closeout_v1","generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":status,"productionAuthority":False,"r15MutationAllowed":False,
        "freshValidationOpened":False,"sealedHoldoutOpened":False,
        "independentReplicationRequiredBeforeValidation":bool(passed),
        "lockSha256":lock_sha,"implementationSpecSha256":hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "dataFreezeSha256":hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest(),
        "fitSymbolHours":int(len(fit)),"metrics":m,"bootstrap":b,"checks":c,
        "failedChecks":[k for k,v in c.items() if not v],
        "modelArtifactSha256":summary["modelArtifactSha256"],
        "terminalRule":LOCK["terminalRule"],
        "nextAction":(
            "Replicate exact MIE2 logic through an independent live-authority transport before opening May-Jun validation."
            if passed else
            "Activate terminal stop. Do not tune MIE2 or open May-Aug evidence. Close alpha research for this project and keep R15 frozen."
        )
    }
    cp=OUT/"MGPT_MIE2_DEVELOPMENT_CLOSEOUT_20260916.json"; cp.write_text(json.dumps(close,indent=2,sort_keys=True)+"\n")
    led=OUT/"MGPT_MIE2_DEVELOPMENT_LEDGER_20260916.csv"
    cols=["signalUtc","entryUtc","symbol","symbolGroup","pLongLogit","pLongHgb","predSpreadR","pTradableHgb","directionalScore","chosenDirection","strength","selected","targetSpreadR","tradableOpportunity","chosenNetR","chosenStressNetR","oppositeNetR","pairMeanR","chosenDeltaVsPairMeanR","directionCorrect","chosenGood","longGood","shortGood","longNetR","shortNetR"]
    devs[cols].to_csv(led,index=False)

    sums=[]
    for p in [cp,led,model_path,sp]:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/"MGPT_MIE2_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")
    print(json.dumps({"status":status,"fitSymbolHours":len(fit),"developmentSymbolHours":len(devs),"failedChecks":close["failedChecks"],"metrics":{k:v for k,v in m.items() if k!="symbolExpectancyR"},"bootstrap":b},indent=2))

if __name__=="__main__":
    main()
