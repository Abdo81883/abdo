#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "MGPT_MIE1_MICROSTRUCTURE_LOCK_20260916.json"
SPEC_PATH = ROOT / "MGPT_MIE1_IMPLEMENTATION_SPEC_V1_20260916.json"
FREEZE_PATH = ROOT / "results" / "MGPT_MIE1_DATA_FREEZE_STATUS_20260916.json"
OUT = ROOT / "results"

LOCK = json.loads(LOCK_PATH.read_text())
SPEC = json.loads(SPEC_PATH.read_text())

FIT_START = pd.Timestamp(LOCK["windows"]["modelFit"]["start"])
FIT_END = pd.Timestamp(LOCK["windows"]["modelFit"]["endExclusive"])
DEV_START = pd.Timestamp(LOCK["windows"]["developmentScreen"]["start"])
DEV_END = pd.Timestamp(LOCK["windows"]["developmentScreen"]["endExclusive"])

NUMERIC = [
    "ret1","ret3","ret6","ret12","ret24","atrPct","rv24","range24Atr","closePos24",
    "quoteVolumeZ24","tradeCountZ24","avgTradeSizeZ24",
    "takerBuyQuoteRatio","takerImbalance","takerImbalanceZ24","takerImbalanceDelta3","takerImbalanceDelta6",
    "premiumClose","premiumZ24","premiumDelta1","premiumDelta3","premiumAbsZ24",
    "latestFundingRate","fundingMean3","fundingMean9","fundingZ20Events","hoursSinceFunding",
    "signedRet3","signedRet12","signedTakerImbalance","signedPremium","signedFunding",
]
CATEGORICAL = ["direction","symbolGroup"]


def read_csv(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    for c in x.columns:
        if c != "timestamp":
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def wilder_atr(x: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = x["close"].shift(1)
    tr = pd.concat([
        (x["high"] - x["low"]).abs(),
        (x["high"] - pc).abs(),
        (x["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()


def prior_z(s: pd.Series, n: int = 24) -> pd.Series:
    mu = s.shift(1).rolling(n, min_periods=n).mean()
    sd = s.shift(1).rolling(n, min_periods=n).std(ddof=1)
    return (s - mu) / sd.replace(0, np.nan)


def group_for_symbol(symbol: str) -> str:
    return "LARGE" if symbol in set(LOCK["symbolGroupRule"]["LARGE"]) else "ALT"


def add_contract_features(c: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    x = c.copy()
    prem = p[["timestamp","close"]].rename(columns={"close":"premiumClose"})
    x = x.merge(prem, on="timestamp", how="left", validate="one_to_one")

    x["atr14"] = wilder_atr(x, 14)
    lr = np.log(x["close"] / x["close"].shift(1))
    x["rv24"] = lr.rolling(24, min_periods=24).std(ddof=1)
    for n in [1,3,6,12,24]:
        x[f"ret{n}"] = x["close"] / x["close"].shift(n) - 1.0
    x["atrPct"] = x["atr14"] / x["close"].replace(0, np.nan)
    x["priorHigh24"] = x["high"].shift(1).rolling(24, min_periods=24).max()
    x["priorLow24"] = x["low"].shift(1).rolling(24, min_periods=24).min()
    x["range24Atr"] = (x["priorHigh24"] - x["priorLow24"]) / x["atr14"].replace(0, np.nan)
    x["closePos24"] = (x["close"] - x["priorLow24"]) / (x["priorHigh24"] - x["priorLow24"]).replace(0, np.nan)

    x["quoteVolumeZ24"] = prior_z(x["quote_volume"],24)
    x["tradeCountZ24"] = prior_z(x["trades"],24)
    avg_trade = x["quote_volume"] / x["trades"].replace(0, np.nan)
    x["avgTradeSizeZ24"] = prior_z(avg_trade,24)

    x["takerBuyQuoteRatio"] = x["taker_buy_quote"] / x["quote_volume"].replace(0, np.nan)
    x["takerImbalance"] = 2.0*x["takerBuyQuoteRatio"] - 1.0
    x["takerImbalanceZ24"] = prior_z(x["takerImbalance"],24)
    x["takerImbalanceDelta3"] = x["takerImbalance"] - x["takerImbalance"].shift(3)
    x["takerImbalanceDelta6"] = x["takerImbalance"] - x["takerImbalance"].shift(6)

    x["premiumZ24"] = prior_z(x["premiumClose"],24)
    x["premiumDelta1"] = x["premiumClose"] - x["premiumClose"].shift(1)
    x["premiumDelta3"] = x["premiumClose"] - x["premiumClose"].shift(3)
    abs_p = x["premiumClose"].abs()
    mu = abs_p.shift(1).rolling(24,min_periods=24).mean()
    sd = abs_p.shift(1).rolling(24,min_periods=24).std(ddof=1)
    x["premiumAbsZ24"] = (abs_p-mu)/sd.replace(0,np.nan)
    return x


def funding_state(funding: pd.DataFrame, decision_time: pd.Timestamp) -> dict:
    times = funding["timestamp"].astype("int64").to_numpy()
    pos = int(np.searchsorted(times, decision_time.value, side="right") - 1)
    if pos < 20:
        return {
            "latestFundingRate":math.nan,"fundingMean3":math.nan,"fundingMean9":math.nan,
            "fundingZ20Events":math.nan,"hoursSinceFunding":math.nan
        }
    rates = funding["funding_rate"].to_numpy(float)
    latest = float(rates[pos])
    prev20 = rates[pos-20:pos]
    sd = float(np.std(prev20, ddof=1))
    z = (latest-float(np.mean(prev20)))/sd if math.isfinite(sd) and sd>0 else math.nan
    return {
        "latestFundingRate":latest,
        "fundingMean3":float(np.mean(rates[pos-2:pos+1])),
        "fundingMean9":float(np.mean(rates[pos-8:pos+1])),
        "fundingZ20Events":z,
        "hoursSinceFunding":float((decision_time-pd.Timestamp(funding.iloc[pos]["timestamp"]))/pd.Timedelta(hours=1)),
    }


def split_for_signal(ts: pd.Timestamp) -> str | None:
    if FIT_START <= ts < FIT_END:
        return "FIT"
    if DEV_START <= ts < DEV_END:
        return "DEVELOPMENT"
    return None


def full_horizon_inside(x: pd.DataFrame, i: int, split: str) -> bool:
    entry_i = i+1
    last_i = entry_i + int(LOCK["candidateEngine"]["maxHoldingBars"]) - 1
    if last_i >= len(x):
        return False
    end = FIT_END if split=="FIT" else DEV_END
    last_close = pd.Timestamp(x.iloc[last_i]["timestamp"]) + pd.Timedelta(hours=1)
    return last_close < end


def funding_cashflow_r(funding: pd.DataFrame, entry_time: pd.Timestamp, funding_end: pd.Timestamp, sign: int, entry: float, risk: float) -> float:
    z = funding[(funding.timestamp > entry_time) & (funding.timestamp <= funding_end)]
    if z.empty:
        return 0.0
    return float((-sign * z.funding_rate.astype(float).sum() * entry) / risk)


def simulate_direction(x: pd.DataFrame, funding: pd.DataFrame, i: int, direction: str, friction_bps: float) -> dict:
    sign = 1 if direction=="LONG" else -1
    entry_i = i+1
    entry_time = pd.Timestamp(x.iloc[entry_i]["timestamp"])
    entry = float(x.iloc[entry_i]["open"])
    atr = float(x.iloc[i]["atr14"])
    risk = atr
    stop = entry - sign*risk
    tp1 = entry + sign*0.75*risk
    tp2 = entry + sign*1.50*risk
    tp3 = entry + sign*2.50*risk

    hold = int(LOCK["candidateEngine"]["maxHoldingBars"])
    last_i = entry_i + hold - 1
    exit_price = float(x.iloc[last_i]["close"])
    exit_i = last_i
    exit_type = "TIMEOUT"
    tp1_hit = False
    tp3_potential = False
    mfe_r = 0.0
    mae_r = 0.0

    for j in range(entry_i, last_i+1):
        h = float(x.iloc[j]["high"])
        l = float(x.iloc[j]["low"])
        favorable = (h-entry)/risk if sign==1 else (entry-l)/risk
        adverse = (entry-l)/risk if sign==1 else (h-entry)/risk
        mfe_r = max(mfe_r, favorable)
        mae_r = max(mae_r, adverse)
        if favorable >= 0.75:
            tp1_hit = True
        if favorable >= 2.50:
            tp3_potential = True

        stop_hit = l <= stop if sign==1 else h >= stop
        if stop_hit:
            exit_price = stop
            exit_i = j
            exit_type = "STOP"
            break
        tp_hit = h >= tp2 if sign==1 else l <= tp2
        if tp_hit:
            exit_price = tp2
            exit_i = j
            exit_type = "TP2"
            break

    exit_bar_time = pd.Timestamp(x.iloc[exit_i]["timestamp"])
    if exit_type=="TIMEOUT":
        funding_end = exit_bar_time + pd.Timedelta(hours=1)
    else:
        funding_end = exit_bar_time

    gross_r = sign*(exit_price-entry)/risk
    friction_rate = float(friction_bps)/10000.0
    friction_r = friction_rate*(abs(entry)+abs(exit_price))/risk
    funding_r = funding_cashflow_r(funding,entry_time,funding_end,sign,entry,risk)
    net_r = gross_r - friction_r + funding_r
    return {
        "entryTime":entry_time,
        "entry":entry,"stop":stop,"tp1":tp1,"tp2":tp2,"tp3":tp3,
        "grossR":float(gross_r),"netR":float(net_r),"fundingR":float(funding_r),
        "frictionR":float(friction_r),"exitType":exit_type,
        "mfeR":float(mfe_r),"maeR":float(mae_r),
        "tp1Hit":bool(tp1_hit),"tp3Potential":bool(tp3_potential),
        "goodOpportunity":int(exit_type=="TP2"),
    }


def build_candidates() -> pd.DataFrame:
    rows=[]
    for symbol in LOCK["universe"]:
        c = read_csv(ROOT/"data"/"contract"/f"{symbol}_1h.csv")
        p = read_csv(ROOT/"data"/"premium"/f"{symbol}_1h.csv")
        f = read_csv(ROOT/"data"/"funding"/f"{symbol}.csv")
        x = add_contract_features(c,p)
        group = group_for_symbol(symbol)

        for i in range(50,len(x)-25):
            signal_ts = pd.Timestamp(x.iloc[i]["timestamp"])
            split = split_for_signal(signal_ts)
            if split is None or not full_horizon_inside(x,i,split):
                continue
            decision_time = signal_ts + pd.Timedelta(hours=1)
            if pd.Timestamp(x.iloc[i+1]["timestamp"]) != decision_time:
                continue
            r=x.iloc[i]
            needed = [
                "atr14","ret1","ret3","ret6","ret12","ret24","atrPct","rv24","range24Atr","closePos24",
                "quoteVolumeZ24","tradeCountZ24","avgTradeSizeZ24","takerBuyQuoteRatio","takerImbalance",
                "takerImbalanceZ24","takerImbalanceDelta3","takerImbalanceDelta6",
                "premiumClose","premiumZ24","premiumDelta1","premiumDelta3","premiumAbsZ24"
            ]
            if any(pd.isna(r.get(k,np.nan)) for k in needed):
                continue
            fs=funding_state(f,decision_time)
            if any(not math.isfinite(float(v)) for v in fs.values()):
                continue

            base_by_dir={}
            stress_by_dir={}
            for direction in ["LONG","SHORT"]:
                base_by_dir[direction]=simulate_direction(x,f,i,direction,float(LOCK["costAndFunding"]["oneWayBaseBps"]))
                stress_by_dir[direction]=simulate_direction(x,f,i,direction,float(LOCK["costAndFunding"]["oneWayStressBps"]))

            for direction in ["LONG","SHORT"]:
                sign=1 if direction=="LONG" else -1
                base=base_by_dir[direction]
                stress=stress_by_dir[direction]
                row={
                    "split":split,"signalUtc":signal_ts.isoformat(),"entryUtc":decision_time.isoformat(),
                    "symbol":symbol,"symbolGroup":group,"direction":direction,
                    **{k:float(r[k]) for k in [
                        "ret1","ret3","ret6","ret12","ret24","atrPct","rv24","range24Atr","closePos24",
                        "quoteVolumeZ24","tradeCountZ24","avgTradeSizeZ24","takerBuyQuoteRatio","takerImbalance",
                        "takerImbalanceZ24","takerImbalanceDelta3","takerImbalanceDelta6",
                        "premiumClose","premiumZ24","premiumDelta1","premiumDelta3","premiumAbsZ24"
                    ]},
                    **{k:float(v) for k,v in fs.items()},
                    "signedRet3":sign*float(r["ret3"]),
                    "signedRet12":sign*float(r["ret12"]),
                    "signedTakerImbalance":sign*float(r["takerImbalance"]),
                    "signedPremium":sign*float(r["premiumClose"]),
                    "signedFunding":sign*float(fs["latestFundingRate"]),
                    "goodOpportunity":int(base["goodOpportunity"]),
                    "positiveNet":int(base["netR"]>0),
                    "tailLoss":int(base["netR"]<=-0.75),
                    "netR":float(base["netR"]),
                    "stressNetR":float(stress["netR"]),
                    "grossR":float(base["grossR"]),
                    "fundingR":float(base["fundingR"]),
                    "frictionR":float(base["frictionR"]),
                    "mfeR":float(base["mfeR"]),
                    "maeR":float(base["maeR"]),
                    "exitType":base["exitType"],
                    "tp1Hit":int(base["tp1Hit"]),
                    "tp3Potential":int(base["tp3Potential"]),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def preprocessor():
    num = Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([("num",num,NUMERIC),("cat",cat,CATEGORICAL)],remainder="drop",sparse_threshold=0.0)


def fit_models(fit: pd.DataFrame, dev: pd.DataFrame):
    pre=preprocessor()
    xf=pre.fit_transform(fit[NUMERIC+CATEGORICAL])
    xd=pre.transform(dev[NUMERIC+CATEGORICAL])

    gl=LOCK["models"]["goodLogit"]
    gLog=LogisticRegression(C=float(gl["C"]),class_weight=gl["class_weight"],max_iter=int(gl["max_iter"]),random_state=int(gl["random_state"]),solver="lbfgs")
    pn=LOCK["models"]["positiveNetLogit"]
    pLog=LogisticRegression(C=float(pn["C"]),class_weight=pn["class_weight"],max_iter=int(pn["max_iter"]),random_state=int(pn["random_state"]),solver="lbfgs")
    gh=LOCK["models"]["goodHgb"]
    gH=HistGradientBoostingClassifier(
        loss="log_loss",learning_rate=float(gh["learning_rate"]),max_iter=int(gh["max_iter"]),
        max_depth=int(gh["max_depth"]),min_samples_leaf=int(gh["min_samples_leaf"]),
        l2_regularization=float(gh["l2_regularization"]),random_state=int(gh["random_state"]),early_stopping=False
    )
    th=LOCK["models"]["tailHgb"]
    tH=HistGradientBoostingClassifier(
        loss="log_loss",learning_rate=float(th["learning_rate"]),max_iter=int(th["max_iter"]),
        max_depth=int(th["max_depth"]),min_samples_leaf=int(th["min_samples_leaf"]),
        l2_regularization=float(th["l2_regularization"]),random_state=int(th["random_state"]),early_stopping=False
    )

    for target in ["goodOpportunity","positiveNet","tailLoss"]:
        if fit[target].nunique()!=2:
            raise RuntimeError(f"fit target {target} has only one class")

    gLog.fit(xf,fit.goodOpportunity.astype(int))
    gH.fit(xf,fit.goodOpportunity.astype(int))
    pLog.fit(xf,fit.positiveNet.astype(int))
    tH.fit(xf,fit.tailLoss.astype(int))

    def score(z,x):
        z=z.copy()
        z["pGoodLogit"]=gLog.predict_proba(x)[:,1]
        z["pGoodHgb"]=gH.predict_proba(x)[:,1]
        z["pPositiveNetLogit"]=pLog.predict_proba(x)[:,1]
        z["pTailHgb"]=tH.predict_proba(x)[:,1]
        z["qualityScore"]=(
            0.35*z.pGoodLogit + 0.30*z.pGoodHgb +
            0.20*z.pPositiveNetLogit - 0.15*z.pTailHgb
        )
        return z

    return score(fit,xf),score(dev,xd),{"preprocessor":pre,"goodLogit":gLog,"goodHgb":gH,"positiveNetLogit":pLog,"tailHgb":tH}


def select_cross_section(dev: pd.DataFrame) -> pd.DataFrame:
    z=dev.copy()
    z["selected"]=False
    z["directionMargin"]=np.nan
    for ts,g in z.groupby("entryUtc",sort=True):
        winners=[]
        for symbol,sg in g.groupby("symbol",sort=True):
            if len(sg)!=2:
                continue
            sg=sg.sort_values(["qualityScore","direction"],ascending=[False,True])
            top=sg.iloc[0]; second=sg.iloc[1]
            margin=float(top.qualityScore-second.qualityScore)
            z.loc[sg.index,"directionMargin"]=margin
            if margin + 1e-15 >= float(LOCK["selection"]["perTimestamp"].split(">=")[-1].split(";")[0].strip()) if False else False:
                pass
            if margin >= 0.03:
                winners.append((float(top.qualityScore),str(symbol),str(top.direction),int(top.name)))
        winners=sorted(winners,key=lambda t:(-t[0],t[1],0 if t[2]=="LONG" else 1))
        for _,_,_,idx in winners[:int(LOCK["selection"]["maxTradesPerHour"])]:
            z.at[idx,"selected"]=True
    return z


def profit_factor(a) -> float:
    x=np.asarray(a,float); p=float(x[x>0].sum()); n=float(-x[x<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)


def max_drawdown(a) -> float:
    eq=peak=dd=0.0
    for v in a:
        eq+=float(v); peak=max(peak,eq); dd=max(dd,peak-eq)
    return float(dd)


def auc(y,score) -> float:
    y=pd.Series(y)
    return float(roc_auc_score(y,score)) if y.nunique()==2 else math.nan


def metrics(dev: pd.DataFrame) -> dict:
    sel=dev[dev.selected].copy()
    symbols=sel.groupby("symbol").netR.mean() if len(sel) else pd.Series(dtype=float)
    sym_total=sel.groupby("symbol").netR.sum() if len(sel) else pd.Series(dtype=float)
    pos=sym_total[sym_total>0]
    conc=float(pos.max()/pos.sum()) if len(pos) else 0.0
    s=sel.sort_values("signalUtc").reset_index(drop=True)
    half=len(s)//2
    mon=sel.assign(month=pd.to_datetime(sel.signalUtc,utc=True).dt.strftime("%Y-%m")).groupby("month").netR.sum() if len(sel) else pd.Series(dtype=float)
    good_all=int(dev.goodOpportunity.sum())
    good_sel=int(sel.goodOpportunity.sum()) if len(sel) else 0
    clock_by_time=dev.groupby("entryUtc").netR.mean()
    clock=float(clock_by_time.mean()) if len(clock_by_time) else 0.0
    return {
        "developmentCandidates":int(len(dev)),
        "selectedTrades":int(len(sel)),
        "selectionCoverage":float(len(sel)/max(1,len(dev))),
        "activeSymbols":int(sel.symbol.nunique()) if len(sel) else 0,
        "activeDirections":sorted(sel.direction.unique().tolist()) if len(sel) else [],
        "netExpectancyR":float(sel.netR.mean()) if len(sel) else 0.0,
        "stressNetExpectancyR":float(sel.stressNetR.mean()) if len(sel) else 0.0,
        "profitFactor":profit_factor(sel.netR) if len(sel) else 0.0,
        "hitRate":float((sel.netR>0).mean()) if len(sel) else 0.0,
        "maxDrawdownR":max_drawdown(s.netR.to_numpy(float)) if len(sel) else 0.0,
        "opportunityRecall":float(good_sel/max(1,good_all)),
        "opportunityPrecision":float(good_sel/max(1,len(sel))),
        "positiveSymbolFraction":float((symbols>0).mean()) if len(symbols) else 0.0,
        "positiveMonthFraction":float((mon>0).mean()) if len(mon) else 0.0,
        "firstHalfNetExpectancyR":float(s.iloc[:half].netR.mean()) if half else 0.0,
        "secondHalfNetExpectancyR":float(s.iloc[half:].netR.mean()) if len(s)-half else 0.0,
        "goodLogitAuc":auc(dev.goodOpportunity,dev.pGoodLogit),
        "goodHgbAuc":auc(dev.goodOpportunity,dev.pGoodHgb),
        "positiveNetAuc":auc(dev.positiveNet,dev.pPositiveNetLogit),
        "tailHgbAuc":auc(dev.tailLoss,dev.pTailHgb),
        "clockBaselineExpectancyR":clock,
        "selectedDeltaVsClockBaselineR":float((sel.netR.mean() if len(sel) else 0.0)-clock),
        "singleSymbolPositiveContribution":conc,
        "symbolExpectancyR":{str(k):float(v) for k,v in symbols.items()},
        "exitTypeCounts":{str(k):int(v) for k,v in sel.exitType.value_counts().items()} if len(sel) else {},
    }


def bootstrap(dev: pd.DataFrame) -> dict:
    cfg=LOCK["bootstrap"]; iters=int(cfg["iterations"]); rng=np.random.default_rng(int(cfg["seed"]))
    by_symbol={}
    for sym,g in dev.groupby("symbol"):
        hours=[]
        for ts,h in g.sort_values(["entryUtc","direction"]).groupby("entryUtc",sort=True):
            hours.append({
                "net":h.netR.to_numpy(float),
                "sel":h.selected.to_numpy(bool)
            })
        by_symbol[str(sym)]=hours

    abs_arr=np.zeros(iters); delta_arr=np.zeros(iters)
    block=int(cfg["blockLengthHours"])
    syms=sorted(by_symbol)
    for it in range(iters):
        selected_vals=[]
        all_vals=[]
        for sym in syms:
            hrs=by_symbol[sym]; n=len(hrs)
            if n==0: continue
            sampled=[]
            while len(sampled)<n:
                start=int(rng.integers(0,n))
                take=min(block,n-len(sampled))
                sampled.extend([(start+j)%n for j in range(take)])
            for idx in sampled:
                h=hrs[idx]
                all_vals.extend(h["net"].tolist())
                selected_vals.extend(h["net"][h["sel"]].tolist())
        sm=float(np.mean(selected_vals)) if selected_vals else 0.0
        bm=float(np.mean(all_vals)) if all_vals else 0.0
        abs_arr[it]=sm
        delta_arr[it]=sm-bm
    return {
        "method":cfg["method"],"iterations":iters,"seed":int(cfg["seed"]),"blockLengthHours":block,
        "probabilityPositiveNetExpectancy":float(np.mean(abs_arr>0)),
        "ci95NetExpectancyR":[float(np.quantile(abs_arr,.025)),float(np.quantile(abs_arr,.975))],
        "probabilityPositiveUtilityDeltaVsClockBaseline":float(np.mean(delta_arr>0)),
        "ci95DeltaVsClockBaselineR":[float(np.quantile(delta_arr,.025)),float(np.quantile(delta_arr,.975))],
    }


def checks(fit_rows:int,m:dict,b:dict)->dict:
    g=LOCK["developmentGate"]
    return {
        "minimumFitCandidates":bool(fit_rows>=int(g["minimumFitCandidates"])),
        "minimumDevelopmentCandidates":bool(m["developmentCandidates"]>=int(g["minimumDevelopmentCandidates"])),
        "minimumSelectedTrades":bool(m["selectedTrades"]>=int(g["minimumSelectedTrades"])),
        "minimumActiveSymbols":bool(m["activeSymbols"]>=int(g["minimumActiveSymbols"])),
        "bothDirectionsActive":bool(set(m["activeDirections"])=={"LONG","SHORT"}),
        "netExpectancyRMin":bool(m["netExpectancyR"]>=float(g["netExpectancyRMin"])),
        "stressNetExpectancyRMin":bool(m["stressNetExpectancyR"]>=float(g["stressNetExpectancyRMin"])),
        "profitFactorMin":bool(m["profitFactor"]>=float(g["profitFactorMin"])),
        "opportunityRecallMin":bool(m["opportunityRecall"]>=float(g["opportunityRecallMin"])),
        "opportunityPrecisionMin":bool(m["opportunityPrecision"]>=float(g["opportunityPrecisionMin"])),
        "positiveSymbolFractionMin":bool(m["positiveSymbolFraction"]>=float(g["positiveSymbolFractionMin"])),
        "positiveMonthFractionMin":bool(m["positiveMonthFraction"]>=float(g["positiveMonthFractionMin"])),
        "firstHalfPositive":bool(m["firstHalfNetExpectancyR"]>0),
        "secondHalfPositive":bool(m["secondHalfNetExpectancyR"]>0),
        "goodLogitAucMin":bool(math.isfinite(m["goodLogitAuc"]) and m["goodLogitAuc"]>=float(g["goodLogitAucMin"])),
        "goodHgbAucMin":bool(math.isfinite(m["goodHgbAuc"]) and m["goodHgbAuc"]>=float(g["goodHgbAucMin"])),
        "positiveNetAucMin":bool(math.isfinite(m["positiveNetAuc"]) and m["positiveNetAuc"]>=float(g["positiveNetAucMin"])),
        "tailHgbAucMin":bool(math.isfinite(m["tailHgbAuc"]) and m["tailHgbAuc"]>=float(g["tailHgbAucMin"])),
        "bootstrapProbabilityPositiveNetMin":bool(b["probabilityPositiveNetExpectancy"]>=float(g["bootstrapProbabilityPositiveNetMin"])),
        "bootstrapProbabilityPositiveUtilityDeltaVsClockBaselineMin":bool(b["probabilityPositiveUtilityDeltaVsClockBaseline"]>=float(g["bootstrapProbabilityPositiveUtilityDeltaVsClockBaselineMin"])),
        "singleSymbolPositiveContributionMax":bool(m["singleSymbolPositiveContribution"]<=float(g["singleSymbolPositiveContributionMax"])),
    }


def main():
    freeze=json.loads(FREEZE_PATH.read_text())
    if freeze.get("status")!="PASS_DATA_FREEZE":
        raise RuntimeError(f"MIE1 freeze is not PASS: {freeze.get('status')}")
    lock_sha=hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    if freeze.get("lockSha256")!=lock_sha:
        raise RuntimeError("MIE1 data freeze lock hash mismatch")
    manifest={x["symbol"]:x for x in freeze["symbols"]}
    for sym in LOCK["universe"]:
        if sym not in manifest:
            raise RuntimeError(f"missing manifest symbol {sym}")
        for key,subdir,fn in [
            ("contract","contract",f"{sym}_1h.csv"),
            ("premium","premium",f"{sym}_1h.csv"),
            ("funding","funding",f"{sym}.csv"),
        ]:
            p=ROOT/"data"/subdir/fn
            if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest[sym][key]["sha256"]:
                raise RuntimeError(f"hash mismatch {sym} {key}")

    table=build_candidates()
    if table.empty:
        raise RuntimeError("MIE1 candidate table empty")
    fit=table[table.split=="FIT"].copy().reset_index(drop=True)
    dev=table[table.split=="DEVELOPMENT"].copy().reset_index(drop=True)
    if len(fit)<1000 or len(dev)<1000:
        raise RuntimeError(f"insufficient candidate rows fit={len(fit)} dev={len(dev)}")

    fit_s,dev_s,models=fit_models(fit,dev)
    dev_s=select_cross_section(dev_s)
    m=metrics(dev_s)
    b=bootstrap(dev_s)
    c=checks(len(fit_s),m,b)
    passed=all(c.values())
    status="PASS_MIE1_DEVELOPMENT_REPLICATION_REQUIRED" if passed else "REJECT_MIE1_DEVELOPMENT_GATE"

    model_path=OUT/"MGPT_MIE1_FROZEN_MODEL_20260916.joblib"
    joblib.dump({**models,"lockSha256":lock_sha},model_path)

    summary={
        "schema":"mgpt_mie1_model_summary_v1",
        "fitCandidates":int(len(fit_s)),"developmentCandidates":int(len(dev_s)),
        "numericFeatures":NUMERIC,"categoricalFeatures":CATEGORICAL,
        "selection":LOCK["selection"],"score":LOCK["score"],
        "modelArtifactSha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "scoreSemantics":"Ranking scores only; no probability/calibration claim."
    }
    sp=OUT/"MGPT_MIE1_MODEL_SUMMARY_20260916.json"
    sp.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")

    closeout={
        "schema":"mgpt_mie1_development_closeout_v1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":status,"productionAuthority":False,"r15MutationAllowed":False,
        "freshValidationOpened":False,"sealedHoldoutOpened":False,
        "independentReplicationRequiredBeforeValidation":bool(passed),
        "lockSha256":lock_sha,
        "implementationSpecSha256":hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "dataFreezeSha256":hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest(),
        "fitCandidates":int(len(fit_s)),
        "metrics":m,"bootstrap":b,"checks":c,
        "failedChecks":[k for k,v in c.items() if not v],
        "modelArtifactSha256":summary["modelArtifactSha256"],
        "failureRule":LOCK["failureRule"],
        "nextAction":(
            "Replicate the exact MIE1 microstructure feature/score/selection logic on an independent Binance execution-authority transport before opening May-Jun fresh validation."
            if passed else
            "Register MIE1 rejection. Do not tune MIE1. Keep May-Aug 2026 sealed; any further research must add another genuinely new data source such as historical open interest, liquidation or order-book depth."
        )
    }
    cp=OUT/"MGPT_MIE1_DEVELOPMENT_CLOSEOUT_20260916.json"
    led=OUT/"MGPT_MIE1_DEVELOPMENT_LEDGER_20260916.csv"
    cp.write_text(json.dumps(closeout,indent=2,sort_keys=True)+"\n")
    cols=[
        "signalUtc","entryUtc","symbol","symbolGroup","direction",
        "qualityScore","pGoodLogit","pGoodHgb","pPositiveNetLogit","pTailHgb",
        "directionMargin","selected","goodOpportunity","positiveNet","tailLoss",
        "netR","stressNetR","grossR","fundingR","frictionR","mfeR","maeR","exitType",
        "takerImbalance","premiumClose","latestFundingRate"
    ]
    dev_s[cols].to_csv(led,index=False)

    sums=[]
    for p in [cp,led,model_path,sp]:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/"MGPT_MIE1_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")

    print(json.dumps({
        "status":status,"fitCandidates":len(fit_s),"developmentCandidates":len(dev_s),
        "failedChecks":closeout["failedChecks"],
        "metrics":{k:v for k,v in m.items() if k not in {"symbolExpectancyR","exitTypeCounts"}},
        "bootstrap":b
    },indent=2))


if __name__=="__main__":
    main()
