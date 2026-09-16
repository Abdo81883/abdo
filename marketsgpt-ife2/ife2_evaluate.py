#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "marketsgpt-ite1"))

from ite1_core import CostModel, compute_features, construct_plan, generate_candidates, simulate_trade  # noqa: E402
from ite1_terminal_common import htf_is_neutral, htf_table, htf_trend_state, resample_4h  # noqa: E402

LOCK_PATH = ROOT / "MGPT_IFE2_DISTRIBUTIONAL_OPPORTUNITY_LOCK_20260916.json"
SPEC_PATH = ROOT / "MGPT_IFE2_IMPLEMENTATION_SPEC_V1_20260916.json"
FREEZE_PATH = ROOT / "results" / "MGPT_IFE2_DATA_FREEZE_STATUS_20260916.json"
OUT = ROOT / "results"

LOCK = json.loads(LOCK_PATH.read_text())
SPEC = json.loads(SPEC_PATH.read_text())

FIT_START = pd.Timestamp(LOCK["windows"]["modelFit"]["start"])
FIT_END = pd.Timestamp(LOCK["windows"]["modelFit"]["endExclusive"])
DEV_START = pd.Timestamp(LOCK["windows"]["developmentScreen"]["start"])
DEV_END = pd.Timestamp(LOCK["windows"]["developmentScreen"]["endExclusive"])

INHERITED_NUMERIC = [
    "trend20_50_atr","trend50_200_atr","ema50_slope5_atr","close_ema20_atr",
    "range20_atr","range50_atr","bar_range_atr","body_atr","clv",
    "lower_wick_atr","upper_wick_atr","vol_ratio","rv20",
    "breakout_up_atr","breakout_down_atr","next_open_gap_atr",
    "actual_risk_atr","actual_tp2_r","htf_trend_signed","htf_neutral_flag",
    "vix_z60","spx_trend_atr","dxy_trend_atr","tnx_change5","btc_trend_atr",
]
ADDED_NUMERIC = list(LOCK["featureSet"]["addedNumeric"])
NUMERIC = INHERITED_NUMERIC + ADDED_NUMERIC
BINARY = ["isBreakout","isPullback","isSweep","htfAligned"]
CATEGORICAL = ["marketFamily","timeframe","direction","regime","volState"]


def read_csv(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    for c in ["open","high","low","close","volume"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    if "volume" not in x.columns:
        x["volume"] = np.nan
    return (
        x.dropna(subset=["timestamp","open","high","low","close"])
         .sort_values("timestamp")
         .drop_duplicates("timestamp")
         .reset_index(drop=True)
    )


def market_cost(family: str, stress: bool = False) -> CostModel:
    bps = float(LOCK["costModelOneWayBps"][family]["stress" if stress else "base"])
    return CostModel(one_way_fee_bps=bps, one_way_slippage_bps=0.0)


def _atr(x: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = x["close"].shift(1)
    tr = pd.concat([
        (x["high"] - x["low"]).abs(),
        (x["high"] - pc).abs(),
        (x["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()


def context_table(file_key: str, features: list[str]) -> pd.DataFrame:
    p = ROOT / "data" / "context" / f"{file_key}_1d.csv"
    x = read_csv(p)
    x["atr14"] = _atr(x, 14)
    x["ema20"] = x["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False, min_periods=50).mean()

    if "vix_z60" in features:
        mu = x["close"].rolling(60, min_periods=40).mean()
        sd = x["close"].rolling(60, min_periods=40).std(ddof=1)
        x["vix_z60"] = (x["close"] - mu) / sd.replace(0, np.nan)
    if "spx_trend_atr" in features:
        x["spx_trend_atr"] = (x["ema20"] - x["ema50"]) / x["atr14"].replace(0, np.nan)
    if "dxy_trend_atr" in features:
        x["dxy_trend_atr"] = (x["ema20"] - x["ema50"]) / x["atr14"].replace(0, np.nan)
    if "btc_trend_atr" in features:
        x["btc_trend_atr"] = (x["ema20"] - x["ema50"]) / x["atr14"].replace(0, np.nan)
    if "tnx_change5" in features:
        x["tnx_change5"] = x["close"] / x["close"].shift(5) - 1.0
    if "vix_change5" in features:
        x["vix_change5"] = x["close"] / x["close"].shift(5) - 1.0
    if "spx_return5" in features:
        x["spx_return5"] = x["close"] / x["close"].shift(5) - 1.0
    if "dxy_return5" in features:
        x["dxy_return5"] = x["close"] / x["close"].shift(5) - 1.0
    if "btc_return5" in features:
        x["btc_return5"] = x["close"] / x["close"].shift(5) - 1.0

    x["availableTime"] = x["timestamp"] + pd.Timedelta(days=1)
    return x[["availableTime"] + features].sort_values("availableTime").reset_index(drop=True)


def build_context() -> dict[str, pd.DataFrame]:
    return {
        "VIX": context_table("VIX", ["vix_z60","vix_change5"]),
        "SPX": context_table("GSPC_CONTEXT", ["spx_trend_atr","spx_return5"]),
        "DXY": context_table("DXY", ["dxy_trend_atr","dxy_return5"]),
        "TNX": context_table("TNX", ["tnx_change5"]),
        "BTC": context_table("BTC_CONTEXT", ["btc_trend_atr","btc_return5"]),
    }


def ctx_get(tbl: pd.DataFrame, ts: pd.Timestamp, col: str) -> float:
    arr = tbl["availableTime"].astype("int64").to_numpy()
    pos = int(np.searchsorted(arr, ts.value, side="right") - 1)
    if pos < 0:
        return math.nan
    v = tbl.iloc[pos][col]
    return float(v) if not pd.isna(v) else math.nan


def split_name(ts: pd.Timestamp) -> str | None:
    if FIT_START <= ts < FIT_END:
        return "FIT"
    if DEV_START <= ts < DEV_END:
        return "DEVELOPMENT"
    return None


def split_end(split: str) -> pd.Timestamp:
    return FIT_END if split == "FIT" else DEV_END


def full_horizon_ok(df: pd.DataFrame, signal_i: int, tf: str, end: pd.Timestamp) -> bool:
    wait = int(LOCK["candidateEngine"]["baselineMaxEntryWaitBars"])
    hold = int(LOCK["candidateEngine"]["baselineHoldingBars"][tf])
    worst = signal_i + wait + hold + 1
    return worst < len(df) and pd.Timestamp(df.iloc[worst]["timestamp"]) < end


def grouped_candidates(f: pd.DataFrame):
    grouped = {}
    for c in generate_candidates(f, start_index=200):
        ts = pd.Timestamp(f.iloc[c.signal_index]["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        key = (int(c.signal_index), ts.isoformat(), c.direction)
        grouped.setdefault(key, []).append(c)
    return grouped.values()


def path_targets(f: pd.DataFrame, signal_i: int, entry_i: int, direction: str, atr: float, hold: int, entry: float) -> tuple[bool,float,float]:
    sign = 1.0 if direction == "LONG" else -1.0
    positive = entry + sign * 1.5 * atr
    adverse = entry - sign * 1.0 * atr
    last = min(len(f)-1, entry_i + hold)
    good = False
    resolved = False
    mfe = 0.0
    mae = 0.0
    for j in range(entry_i, last + 1):
        h = float(f.iloc[j]["high"])
        l = float(f.iloc[j]["low"])
        favorable = (h-entry)/atr if sign == 1 else (entry-l)/atr
        adverse_exc = (entry-l)/atr if sign == 1 else (h-entry)/atr
        mfe = max(mfe, favorable)
        mae = max(mae, adverse_exc)
        if not resolved:
            adv_hit = l <= adverse if sign == 1 else h >= adverse
            pos_hit = h >= positive if sign == 1 else l <= positive
            if adv_hit:
                good = False
                resolved = True
            elif pos_hit:
                good = True
                resolved = True
    return bool(good), float(mfe), float(mae)


def volume_z20(f: pd.DataFrame, i: int) -> float:
    if i < 20:
        return math.nan
    cur = f.iloc[i].get("volume", np.nan)
    hist = pd.to_numeric(f.iloc[i-20:i]["volume"], errors="coerce")
    if pd.isna(cur) or hist.notna().sum() < 10:
        return math.nan
    mu = float(hist.mean())
    sd = float(hist.std(ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        return math.nan
    return (float(cur)-mu)/sd


def efficiency20(f: pd.DataFrame, i: int) -> float:
    if i < 20:
        return math.nan
    closes = pd.to_numeric(f.iloc[i-20:i+1]["close"], errors="coerce").to_numpy(float)
    if not np.isfinite(closes).all():
        return math.nan
    denom = float(np.abs(np.diff(closes)).sum())
    return abs(float(closes[-1]-closes[0]))/denom if denom > 0 else math.nan


def feature_row(item, tf, f, htf, candidates, plan, base, stress, context) -> dict | None:
    rep = candidates[0]
    i = rep.signal_index
    ts = pd.Timestamp(f.iloc[i]["timestamp"])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    split = split_name(ts)
    if split is None or not full_horizon_ok(f, i, tf, split_end(split)):
        return None
    if not base.entered or base.entry_index is None or base.entry_price is None or not stress.entered:
        return None

    r = f.iloc[i]
    atr = float(r["atr14"]) if not pd.isna(r["atr14"]) else math.nan
    if not (math.isfinite(atr) and atr > 0) or i < 20:
        return None

    vals = [r.get(k) for k in ["ema20","ema50","ema200","prior_high20","prior_low20","prior_high50","prior_low50","rv20","atr_pct","atr_pct_med60"]]
    if any(pd.isna(v) for v in vals[:7]):
        return None

    e20,e50,e200,ph20,pl20,ph50,pl50 = map(float, vals[:7])
    e50p = float(f.iloc[i-5]["ema50"]) if i >= 5 and not pd.isna(f.iloc[i-5]["ema50"]) else math.nan
    if not math.isfinite(e50p):
        return None

    o,h,l,c = map(float,[r["open"],r["high"],r["low"],r["close"]])
    bar_range = h-l
    if bar_range <= 0:
        return None

    entry = float(base.entry_price)
    entry_i = int(base.entry_index)
    entry_ts = pd.Timestamp(f.iloc[entry_i]["timestamp"])
    entry_ts = entry_ts.tz_localize("UTC") if entry_ts.tzinfo is None else entry_ts.tz_convert("UTC")
    sign = 1.0 if rep.direction=="LONG" else -1.0
    risk = abs(entry-float(plan.stop_loss))
    if not (math.isfinite(risk) and risk>0):
        return None
    actual_tp2_r = sign*(float(plan.tp2)-entry)/risk
    if not math.isfinite(actual_tp2_r):
        return None

    hold = int(LOCK["candidateEngine"]["baselineHoldingBars"][tf])
    good,mfe_atr,mae_atr = path_targets(f,i,entry_i,rep.direction,atr,hold,entry)

    hs = htf_trend_state(htf, entry_ts)
    hs_signed = 1.0 if hs=="LONG" else (-1.0 if hs=="SHORT" else 0.0)
    setup_names = {x.setup_class for x in candidates}
    vol_ratio = float(r["atr_pct"]/r["atr_pct_med60"]) if not pd.isna(r["atr_pct_med60"]) and float(r["atr_pct_med60"])!=0 else math.nan

    def close_lag(n):
        if i < n:
            return math.nan
        v = f.iloc[i-n]["close"]
        return float(v) if not pd.isna(v) else math.nan

    def ratio_lag(col,n):
        if i < n or pd.isna(r[col]) or pd.isna(f.iloc[i-n][col]):
            return math.nan
        den=float(f.iloc[i-n][col])
        return float(r[col])/den-1.0 if den!=0 else math.nan

    range20 = ph20-pl20
    range50 = ph50-pl50
    hour = entry_ts.hour + entry_ts.minute/60.0
    wd = entry_ts.weekday()

    row = {
        "split":split,"signalUtc":ts.isoformat(),"entryUtc":entry_ts.isoformat(),
        "symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,
        "direction":rep.direction,"regime":str(r["regime"]),"volState":str(r["vol_state"]),
        "isBreakout":int("BREAKOUT" in setup_names),
        "isPullback":int("PULLBACK" in setup_names),
        "isSweep":int("SWEEP_RECLAIM" in setup_names),
        "trend20_50_atr":(e20-e50)/atr,
        "trend50_200_atr":(e50-e200)/atr,
        "ema50_slope5_atr":(e50-e50p)/atr,
        "close_ema20_atr":(c-e20)/atr,
        "range20_atr":range20/atr,
        "range50_atr":range50/atr,
        "bar_range_atr":bar_range/atr,
        "body_atr":abs(c-o)/atr,
        "clv":min(1.0,max(0.0,(c-l)/bar_range)),
        "lower_wick_atr":(min(o,c)-l)/atr,
        "upper_wick_atr":(h-max(o,c))/atr,
        "vol_ratio":vol_ratio,
        "rv20":float(r["rv20"]) if not pd.isna(r["rv20"]) else math.nan,
        "breakout_up_atr":(c-ph20)/atr,
        "breakout_down_atr":(pl20-c)/atr,
        "next_open_gap_atr":sign*(entry-c)/atr,
        "actual_risk_atr":risk/atr,
        "actual_tp2_r":actual_tp2_r,
        "htf_trend_signed":hs_signed,
        "htf_neutral_flag":int(htf_is_neutral(htf,entry_ts,0.50)),
        "htfAligned":int(hs==rep.direction),
        "vix_z60":ctx_get(context["VIX"],entry_ts,"vix_z60"),
        "spx_trend_atr":ctx_get(context["SPX"],entry_ts,"spx_trend_atr"),
        "dxy_trend_atr":ctx_get(context["DXY"],entry_ts,"dxy_trend_atr"),
        "tnx_change5":ctx_get(context["TNX"],entry_ts,"tnx_change5"),
        "btc_trend_atr":ctx_get(context["BTC"],entry_ts,"btc_trend_atr"),
        "ret1_atr":(c-close_lag(1))/atr if math.isfinite(close_lag(1)) else math.nan,
        "ret3_atr":(c-close_lag(3))/atr if math.isfinite(close_lag(3)) else math.nan,
        "ret6_atr":(c-close_lag(6))/atr if math.isfinite(close_lag(6)) else math.nan,
        "ret12_atr":(c-close_lag(12))/atr if math.isfinite(close_lag(12)) else math.nan,
        "efficiency20":efficiency20(f,i),
        "close_pos20":(c-pl20)/range20 if range20>0 else math.nan,
        "close_pos50":(c-pl50)/range50 if range50>0 else math.nan,
        "atr_accel10":ratio_lag("atr14",10),
        "rv_accel20":ratio_lag("rv20",20),
        "volume_z20":volume_z20(f,i),
        "hour_sin":math.sin(2*math.pi*hour/24.0),
        "hour_cos":math.cos(2*math.pi*hour/24.0),
        "weekday_sin":math.sin(2*math.pi*wd/7.0),
        "weekday_cos":math.cos(2*math.pi*wd/7.0),
        "distance_ema200_atr":(c-e200)/atr,
        "vix_change5":ctx_get(context["VIX"],entry_ts,"vix_change5"),
        "spx_return5":ctx_get(context["SPX"],entry_ts,"spx_return5"),
        "dxy_return5":ctx_get(context["DXY"],entry_ts,"dxy_return5"),
        "btc_return5":ctx_get(context["BTC"],entry_ts,"btc_return5"),
        "goodOpportunity":int(good),
        "tailLoss":int(float(base.net_r)<=-0.75),
        "mfeAtr":float(mfe_atr),
        "maeAtr":float(mae_atr),
        "netR":float(base.net_r),
        "stressNetR":float(stress.net_r),
        "grossR":float(base.gross_r),
    }
    return row


def build_table() -> tuple[pd.DataFrame,list[dict]]:
    context=build_context()
    rows=[]
    diag=[]
    for item in LOCK["universe"]:
        raw=read_csv(ROOT/"data"/"development"/f"{item['fileKey']}_1h.csv")
        for tf in item["timeframes"]:
            df=raw if tf=="1h" else resample_4h(raw)
            if len(df)<230:
                diag.append({"symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,"status":"INSUFFICIENT_BARS","bars":len(df)})
                continue
            f=compute_features(df)
            htf=htf_table(raw,tf,item["marketFamily"])
            groups=0
            eligible=0
            for candidates in grouped_candidates(f):
                rep=candidates[0]
                ts=pd.Timestamp(f.iloc[rep.signal_index]["timestamp"])
                ts=ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                split=split_name(ts)
                if split is None:
                    continue
                groups+=1
                if not full_horizon_ok(f,rep.signal_index,tf,split_end(split)):
                    continue
                p=construct_plan(
                    f,rep,
                    entry_mode="trigger_close",
                    stop_mode="structure_plus_atr_noise",
                    target_mode="hybrid_structure_volatility_ladder",
                    max_entry_wait_bars=int(LOCK["candidateEngine"]["baselineMaxEntryWaitBars"]),
                    max_holding_bars=int(LOCK["candidateEngine"]["baselineHoldingBars"][tf]),
                )
                if p is None:
                    continue
                base=simulate_trade(f,p,management_mode="fixed_full_exit",cost_model=market_cost(item["marketFamily"],False))
                if not base.entered:
                    continue
                stress=simulate_trade(f,p,management_mode="fixed_full_exit",cost_model=market_cost(item["marketFamily"],True))
                row=feature_row(item,tf,f,htf,candidates,p,base,stress,context)
                if row is not None:
                    rows.append(row); eligible+=1
            diag.append({"symbol":item["symbol"],"marketFamily":item["marketFamily"],"timeframe":tf,"status":"OK","bars":len(df),"candidateGroups":groups,"eligibleRows":eligible})
    return pd.DataFrame(rows),diag


def preprocessor():
    num=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([("num",num,NUMERIC+BINARY),("cat",cat,CATEGORICAL)],remainder="drop",sparse_threshold=0.0)


def score_models(fit: pd.DataFrame, dev: pd.DataFrame):
    pre=preprocessor()
    xfit=pre.fit_transform(fit[NUMERIC+BINARY+CATEGORICAL])
    xdev=pre.transform(dev[NUMERIC+BINARY+CATEGORICAL])

    good_cfg=LOCK["models"]["goodLogit"]
    tail_cfg=LOCK["models"]["tailLogit"]
    good=LogisticRegression(C=float(good_cfg["C"]),class_weight=good_cfg["class_weight"],max_iter=int(good_cfg["max_iter"]),random_state=int(good_cfg["random_state"]))
    tail=LogisticRegression(C=float(tail_cfg["C"]),class_weight=tail_cfg["class_weight"],max_iter=int(tail_cfg["max_iter"]),random_state=int(tail_cfg["random_state"]))
    if fit.goodOpportunity.nunique()!=2 or fit.tailLoss.nunique()!=2:
        raise RuntimeError("IFE2 fit classification target lacks both classes")
    good.fit(xfit,fit.goodOpportunity.astype(int).to_numpy())
    tail.fit(xfit,fit.tailLoss.astype(int).to_numpy())

    mcfg=LOCK["models"]["mfeQ25"]
    acfg=LOCK["models"]["maeQ75"]
    mfe=HistGradientBoostingRegressor(
        loss=mcfg["loss"],quantile=float(mcfg["quantile"]),
        learning_rate=float(mcfg["learning_rate"]),max_iter=int(mcfg["max_iter"]),
        max_depth=int(mcfg["max_depth"]),min_samples_leaf=int(mcfg["min_samples_leaf"]),
        l2_regularization=float(mcfg["l2_regularization"]),random_state=int(mcfg["random_state"])
    )
    mae=HistGradientBoostingRegressor(
        loss=acfg["loss"],quantile=float(acfg["quantile"]),
        learning_rate=float(acfg["learning_rate"]),max_iter=int(acfg["max_iter"]),
        max_depth=int(acfg["max_depth"]),min_samples_leaf=int(acfg["min_samples_leaf"]),
        l2_regularization=float(acfg["l2_regularization"]),random_state=int(acfg["random_state"])
    )
    mfe.fit(xfit,fit.mfeAtr.to_numpy(float))
    mae.fit(xfit,fit.maeAtr.to_numpy(float))

    def attach(z,x):
        z=z.copy()
        z["pGoodScore"]=good.predict_proba(x)[:,1]
        z["pTailLossScore"]=tail.predict_proba(x)[:,1]
        z["predMfeQ25Atr"]=mfe.predict(x)
        z["predMaeQ75Atr"]=mae.predict(x)
        z["qualityScore"]=0.60*(z.pGoodScore-z.pTailLossScore)+0.40*np.tanh((z.predMfeQ25Atr-z.predMaeQ75Atr)/1.5)
        return z

    return attach(fit,xfit),attach(dev,xdev),{"preprocessor":pre,"goodLogit":good,"tailLogit":tail,"mfeQ25":mfe,"maeQ75":mae}


def adaptive_select(fit_scored: pd.DataFrame, dev_scored: pd.DataFrame) -> pd.DataFrame:
    q=float(LOCK["adaptiveSelection"]["quantile"])
    min_hist=int(LOCK["adaptiveSelection"]["minimumHistoryRows"])
    hist=defaultdict(list)
    for fam,g in fit_scored.groupby("marketFamily"):
        hist[str(fam)].extend(g.qualityScore.astype(float).tolist())

    d=dev_scored.copy()
    d["selected"]=False
    d["familyThreshold"]=np.nan
    d["_entryTs"]=pd.to_datetime(d.entryUtc,utc=True)

    for ts,group in d.sort_values(["_entryTs","marketFamily","symbol","timeframe"]).groupby("_entryTs",sort=True):
        idxs=list(group.index)
        decisions=[]
        for idx in idxs:
            r=d.loc[idx]
            fam=str(r.marketFamily)
            history=hist[fam]
            threshold=float(np.quantile(np.asarray(history,float),q)) if len(history)>=min_hist else math.nan
            d.at[idx,"familyThreshold"]=threshold
            hard=(
                float(r.pGoodScore)>=0.50 and
                float(r.pTailLossScore)<=0.50 and
                float(r.predMfeQ25Atr)>=float(r.predMaeQ75Atr) and
                float(r.actual_tp2_r)>=1.25
            )
            decisions.append((idx, bool(math.isfinite(threshold) and float(r.qualityScore)>=threshold and hard)))
        for idx,sel in decisions:
            d.at[idx,"selected"]=sel
        for idx in idxs:
            hist[str(d.at[idx,"marketFamily"])].append(float(d.at[idx,"qualityScore"]))
    return d.drop(columns=["_entryTs"])


def profit_factor(a) -> float:
    x=np.asarray(a,float)
    p=float(x[x>0].sum()); n=float(-x[x<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)


def max_drawdown(a) -> float:
    eq=peak=dd=0.0
    for v in a:
        eq+=float(v); peak=max(peak,eq); dd=max(dd,peak-eq)
    return float(dd)


def metrics(dev: pd.DataFrame) -> dict:
    sel=dev[dev.selected].copy()
    ygood=dev.goodOpportunity.astype(int)
    ytail=dev.tailLoss.astype(int)
    auc_good=float(roc_auc_score(ygood,dev.pGoodScore)) if ygood.nunique()==2 else math.nan
    auc_tail=float(roc_auc_score(ytail,dev.pTailLossScore)) if ytail.nunique()==2 else math.nan
    base_mean=float(dev.netR.mean())
    selected_util=np.where(dev.selected.to_numpy(bool),dev.netR.to_numpy(float),0.0)
    base={
        "developmentEligibleRows":int(len(dev)),
        "selectedTrades":int(len(sel)),
        "selectionCoverage":float(len(sel)/max(1,len(dev))),
        "allEligibleBaselineExpectancyR":base_mean,
        "candidateUtilityDeltaR":float((selected_util-dev.netR.to_numpy(float)).mean()),
        "selectedCandidateUtilityMeanR":float(selected_util.mean()),
        "goodScoreAuc":auc_good,
        "tailLossScoreAuc":auc_tail,
    }
    if sel.empty:
        return {**base,"activeMarketFamilies":0,"activeBlocks":0,"netExpectancyR":0.0,"stressNetExpectancyR":0.0,"profitFactor":0.0,"hitRate":0.0,"maxDrawdownR":0.0,"opportunityRecall":0.0,"opportunityPrecision":0.0,"positiveMarketFamilyFraction":0.0,"positiveBlockFraction":0.0,"firstHalfNetExpectancyR":0.0,"secondHalfNetExpectancyR":0.0,"positiveMonthFraction":0.0,"singleMarketFamilyPositiveContribution":0.0,"familyExpectancyR":{},"blockExpectancyR":{}}
    fam=sel.groupby("marketFamily").netR.mean()
    blk=sel.groupby(["symbol","timeframe"]).netR.mean()
    ft=sel.groupby("marketFamily").netR.sum()
    pos=ft[ft>0]
    conc=float(pos.max()/pos.sum()) if len(pos) else 0.0
    s=sel.sort_values("signalUtc").reset_index(drop=True)
    half=len(s)//2
    mon=sel.assign(month=pd.to_datetime(sel.signalUtc,utc=True).dt.strftime("%Y-%m")).groupby("month").netR.sum()
    good_all=int(dev.goodOpportunity.sum())
    good_sel=int(sel.goodOpportunity.sum())
    return {
        **base,
        "activeMarketFamilies":int(len(fam)),
        "activeBlocks":int(len(blk)),
        "netExpectancyR":float(sel.netR.mean()),
        "stressNetExpectancyR":float(sel.stressNetR.mean()),
        "profitFactor":profit_factor(sel.netR),
        "hitRate":float((sel.netR>0).mean()),
        "maxDrawdownR":max_drawdown(s.netR.to_numpy(float)),
        "opportunityRecall":float(good_sel/max(1,good_all)),
        "opportunityPrecision":float(good_sel/max(1,len(sel))),
        "positiveMarketFamilyFraction":float((fam>0).mean()),
        "positiveBlockFraction":float((blk>0).mean()),
        "firstHalfNetExpectancyR":float(s.iloc[:half].netR.mean()) if half else 0.0,
        "secondHalfNetExpectancyR":float(s.iloc[half:].netR.mean()) if len(s)-half else 0.0,
        "positiveMonthFraction":float((mon>0).mean()) if len(mon) else 0.0,
        "singleMarketFamilyPositiveContribution":conc,
        "familyExpectancyR":{str(k):float(v) for k,v in fam.items()},
        "blockExpectancyR":{f"{k[0]}|{k[1]}":float(v) for k,v in blk.items()},
    }


def circ_idx(n,rng,bl):
    out=[]
    while len(out)<n:
        s=int(rng.integers(0,n)); take=min(bl,n-len(out))
        out.extend(((s+np.arange(take))%n).tolist())
    return np.asarray(out,int)


def bootstrap(dev: pd.DataFrame) -> dict:
    cfg=LOCK["bootstrap"]; iters=int(cfg["iterations"]); bl=int(cfg["blockLengthCandidates"])
    rng=np.random.default_rng(int(cfg["seed"]))
    blocks=sorted(dev[["symbol","timeframe"]].drop_duplicates().itertuples(index=False,name=None))
    packed={}
    for b in blocks:
        z=dev[(dev.symbol==b[0])&(dev.timeframe==b[1])].sort_values("signalUtc").reset_index(drop=True)
        packed[b]={"n":len(z),"sel":z.selected.to_numpy(bool),"net":z.netR.to_numpy(float)}
    abs_s=np.zeros(iters); delta=np.zeros(iters)
    for it in range(iters):
        trsum=0.0; trn=0; dsum=0.0; dn=0
        sampled=[blocks[int(rng.integers(0,len(blocks)))] for _ in blocks]
        for b in sampled:
            p=packed[b]
            if p["n"]<=0: continue
            idx=circ_idx(p["n"],rng,bl)
            sel=p["sel"][idx]; vals=p["net"][idx]
            trsum+=float(vals[sel].sum()); trn+=int(sel.sum())
            dsum+=float((np.where(sel,vals,0.0)-vals).sum()); dn+=len(idx)
        abs_s[it]=trsum/trn if trn else 0.0
        delta[it]=dsum/dn if dn else 0.0
    return {
        "method":cfg["method"],"iterations":iters,"seed":int(cfg["seed"]),"blockLengthCandidates":bl,
        "probabilityPositiveNetExpectancy":float(np.mean(abs_s>0)),
        "ci95NetExpectancyR":[float(np.quantile(abs_s,.025)),float(np.quantile(abs_s,.975))],
        "probabilityPositiveUtilityDeltaVsAllEligible":float(np.mean(delta>0)),
        "ci95UtilityDeltaR":[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))],
    }


def gate_checks(fit_rows:int,m:dict,b:dict)->dict:
    g=LOCK["developmentGate"]
    c={
        "minimumFitRows":fit_rows>=int(g["minimumFitRows"]),
        "minimumDevelopmentEligibleRows":m["developmentEligibleRows"]>=int(g["minimumDevelopmentEligibleRows"]),
        "minimumSelectedTrades":m["selectedTrades"]>=int(g["minimumSelectedTrades"]),
        "minimumActiveMarketFamilies":m["activeMarketFamilies"]>=int(g["minimumActiveMarketFamilies"]),
        "minimumActiveBlocks":m["activeBlocks"]>=int(g["minimumActiveBlocks"]),
        "selectionCoverageMin":m["selectionCoverage"]>=float(g["selectionCoverageMin"]),
        "selectionCoverageMax":m["selectionCoverage"]<=float(g["selectionCoverageMax"]),
        "netExpectancyRMin":m["netExpectancyR"]>=float(g["netExpectancyRMin"]),
        "stressNetExpectancyRMin":m["stressNetExpectancyR"]>=float(g["stressNetExpectancyRMin"]),
        "profitFactorMin":m["profitFactor"]>=float(g["profitFactorMin"]),
        "opportunityRecallMin":m["opportunityRecall"]>=float(g["opportunityRecallMin"]),
        "opportunityPrecisionMin":m["opportunityPrecision"]>=float(g["opportunityPrecisionMin"]),
        "positiveMarketFamilyFractionMin":m["positiveMarketFamilyFraction"]>=float(g["positiveMarketFamilyFractionMin"]),
        "positiveBlockFractionMin":m["positiveBlockFraction"]>=float(g["positiveBlockFractionMin"]),
        "firstHalfPositive":m["firstHalfNetExpectancyR"]>0,
        "secondHalfPositive":m["secondHalfNetExpectancyR"]>0,
        "positiveMonthFractionMin":m["positiveMonthFraction"]>=float(g["positiveMonthFractionMin"]),
        "bootstrapProbabilityPositiveNetMin":b["probabilityPositiveNetExpectancy"]>=float(g["bootstrapProbabilityPositiveNetMin"]),
        "bootstrapProbabilityPositiveUtilityDeltaVsAllEligibleMin":b["probabilityPositiveUtilityDeltaVsAllEligible"]>=float(g["bootstrapProbabilityPositiveUtilityDeltaVsAllEligibleMin"]),
        "candidateUtilityDeltaRMin":m["candidateUtilityDeltaR"]>=float(g["candidateUtilityDeltaRMin"]),
        "goodScoreAucMin":math.isfinite(m["goodScoreAuc"]) and m["goodScoreAuc"]>=float(g["goodScoreAucMin"]),
        "tailLossScoreAucMin":math.isfinite(m["tailLossScoreAuc"]) and m["tailLossScoreAuc"]>=float(g["tailLossScoreAucMin"]),
        "singleMarketFamilyPositiveContributionMax":m["singleMarketFamilyPositiveContribution"]<=float(g["singleMarketFamilyPositiveContributionMax"]),
    }
    return {k:bool(v) for k,v in c.items()}


def main():
    freeze=json.loads(FREEZE_PATH.read_text())
    if freeze.get("status")!="PASS_DATA_FREEZE":
        raise RuntimeError(f"IFE2 freeze not PASS: {freeze.get('status')}")
    lock_sha=hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    if freeze.get("lockSha256")!=lock_sha:
        raise RuntimeError("IFE2 freeze lock hash mismatch")

    mm={x["file"]:x for x in freeze["marketFiles"]}
    cm={x["file"]:x for x in freeze["contextFiles"]}
    for item in LOCK["universe"]:
        fn=f"{item['fileKey']}_1h.csv"; p=ROOT/"data"/"development"/fn
        if fn not in mm or hashlib.sha256(p.read_bytes()).hexdigest()!=mm[fn]["sha256"]:
            raise RuntimeError(f"market hash mismatch {fn}")
    for item in LOCK["contextSeries"]:
        fn=f"{item['fileKey']}_1d.csv"; p=ROOT/"data"/"context"/fn
        if fn not in cm or hashlib.sha256(p.read_bytes()).hexdigest()!=cm[fn]["sha256"]:
            raise RuntimeError(f"context hash mismatch {fn}")

    table,diag=build_table()
    if table.empty: raise RuntimeError("IFE2 candidate table empty")
    fit=table[table.split=="FIT"].copy().reset_index(drop=True)
    dev=table[table.split=="DEVELOPMENT"].copy().reset_index(drop=True)
    if len(fit)<100 or len(dev)<100: raise RuntimeError(f"insufficient rows fit={len(fit)} dev={len(dev)}")

    fit_s,dev_s,models=score_models(fit,dev)
    dev_s=adaptive_select(fit_s,dev_s)
    m=metrics(dev_s)
    b=bootstrap(dev_s)
    checks=gate_checks(len(fit_s),m,b)
    passed=all(checks.values())
    status="PASS_IFE2_DEVELOPMENT_REPLICATION_REQUIRED" if passed else "REJECT_IFE2_DEVELOPMENT_GATE"

    model_path=OUT/"MGPT_IFE2_FROZEN_MODEL_20260916.joblib"
    joblib.dump({**models,"lockSha256":lock_sha},model_path)

    summary={
        "schema":"mgpt_ife2_model_summary_v1",
        "fitRows":int(len(fit_s)),"developmentRows":int(len(dev_s)),
        "numericFeatures":NUMERIC,"binaryFeatures":BINARY,"categoricalFeatures":CATEGORICAL,
        "adaptiveSelection":LOCK["adaptiveSelection"],
        "score":LOCK["score"],
        "modelArtifactSha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "scoreSemantics":"Classifier scores and qualityScore are ranking scores only; no calibrated probability claim."
    }
    sp=OUT/"MGPT_IFE2_MODEL_SUMMARY_20260916.json"
    sp.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")

    closeout={
        "schema":"mgpt_ife2_development_closeout_v1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":status,
        "productionAuthority":False,"r15MutationAllowed":False,
        "freshValidationOpened":False,"sealedHoldoutOpened":False,
        "independentProviderReplicationRequiredBeforeValidation":bool(passed),
        "lockSha256":lock_sha,
        "implementationSpecSha256":hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "dataFreezeSha256":hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest(),
        "fitRows":int(len(fit_s)),
        "metrics":m,"bootstrap":b,"checks":checks,
        "failedChecks":[k for k,v in checks.items() if not v],
        "blockDiagnostics":diag,
        "modelArtifactSha256":summary["modelArtifactSha256"],
        "terminalRuleOnFailure":LOCK["terminalRule"],
        "nextAction":(
            "Replicate exact IFE2 feature/score/selection logic on independent execution-authority data before opening fresh validation."
            if passed else
            "Register IFE2 rejection. Do not tune IFE2 on these outcomes. Keep validation and holdout sealed; any further program must add genuinely new information such as venue microstructure/funding/order-book/event data."
        )
    }
    cp=OUT/"MGPT_IFE2_DEVELOPMENT_CLOSEOUT_20260916.json"
    tp=OUT/"MGPT_IFE2_CANDIDATE_TABLE_20260916.csv"
    dp=OUT/"MGPT_IFE2_DEVELOPMENT_PREDICTIONS_20260916.csv"
    cp.write_text(json.dumps(closeout,indent=2,sort_keys=True)+"\n")
    table.to_csv(tp,index=False)
    dev_s.to_csv(dp,index=False)

    sums=[]
    for p in [cp,tp,dp,model_path,sp]:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/"MGPT_IFE2_SHA256SUMS_20260916.txt").write_text("\n".join(sums)+"\n")

    print(json.dumps({
        "status":status,"fitRows":len(fit_s),"developmentRows":len(dev_s),
        "failedChecks":closeout["failedChecks"],
        "metrics":{k:v for k,v in m.items() if k not in {"familyExpectancyR","blockExpectancyR"}},
        "bootstrap":b
    },indent=2))


if __name__=="__main__":
    main()
