#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "marketsgpt-ite1"))

from ite1_core import (  # noqa: E402
    CostModel,
    audit_reference_opportunity,
    compute_features,
    construct_plan,
    generate_candidates,
    simulate_trade,
)
from ite1_terminal_common import htf_is_neutral, htf_table, htf_trend_state, resample_4h  # noqa: E402

LOCK_PATH = ROOT / "MGPT_IFE1_PROGRAM_LOCK_20260916.json"
SPEC_PATH = ROOT / "MGPT_IFE1_IMPLEMENTATION_SPEC_V1_20260916.json"
FREEZE_PATH = ROOT / "results" / "MGPT_IFE1_DATA_FREEZE_STATUS_20260916.json"
OUT = ROOT / "results"

LOCK = json.loads(LOCK_PATH.read_text())
SPEC = json.loads(SPEC_PATH.read_text())
FIT_START = pd.Timestamp(LOCK["windows"]["modelFit"]["start"])
FIT_END = pd.Timestamp(LOCK["windows"]["modelFit"]["endExclusive"])
DEV_START = pd.Timestamp(LOCK["windows"]["developmentScreen"]["start"])
DEV_END = pd.Timestamp(LOCK["windows"]["developmentScreen"]["endExclusive"])
NUMERIC = list(LOCK["numericFeatures"])
CATEGORICAL = list(LOCK["categoricalFeatures"])
BINARY = list(LOCK["binaryFeatures"])


def read_csv(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    if "volume" not in x.columns:
        x["volume"] = np.nan
    return (
        x.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def market_cost(family: str, stress: bool = False) -> CostModel:
    b = float(LOCK["costModelOneWayBps"][family]["stress" if stress else "base"])
    return CostModel(one_way_fee_bps=b, one_way_slippage_bps=0.0)


def context_atr(x: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = x["close"].shift(1)
    tr = pd.concat(
        [(x["high"] - x["low"]).abs(), (x["high"] - pc).abs(), (x["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def build_context_table(file_key: str, feature_name: str) -> pd.DataFrame:
    p = ROOT / "data" / "context" / f"{file_key}_1d.csv"
    x = read_csv(p)
    x["atr14"] = context_atr(x, 14)
    x["ema20"] = x["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    if feature_name == "vix_z60":
        mu = x["close"].rolling(60, min_periods=40).mean()
        sd = x["close"].rolling(60, min_periods=40).std(ddof=1)
        x[feature_name] = (x["close"] - mu) / sd.replace(0, np.nan)
    elif feature_name in {"spx_trend_atr", "dxy_trend_atr", "btc_trend_atr"}:
        x[feature_name] = (x["ema20"] - x["ema50"]) / x["atr14"].replace(0, np.nan)
    elif feature_name == "tnx_change5":
        x[feature_name] = x["close"] / x["close"].shift(5) - 1.0
    else:
        raise ValueError(feature_name)
    x["availableTime"] = x["timestamp"] + pd.Timedelta(days=1)
    return x[["availableTime", feature_name]].dropna().sort_values("availableTime").reset_index(drop=True)


def context_lookup(table: pd.DataFrame, ts: pd.Timestamp, col: str) -> float:
    arr = table["availableTime"].astype("int64").to_numpy()
    pos = int(np.searchsorted(arr, ts.value, side="right") - 1)
    if pos < 0:
        return math.nan
    v = table.iloc[pos][col]
    return float(v) if not pd.isna(v) else math.nan


def full_split_horizon_ok(df: pd.DataFrame, i: int, tf: str, split_end: pd.Timestamp) -> bool:
    hold = int(LOCK["candidateEngine"]["baselineHoldingBars"][tf])
    wait = int(LOCK["candidateEngine"]["baselineMaxEntryWaitBars"])
    worst = i + wait + hold + 1
    return worst < len(df) and pd.Timestamp(df.iloc[worst]["timestamp"]) < split_end


def candidate_split(ts: pd.Timestamp) -> str | None:
    if FIT_START <= ts < FIT_END:
        return "FIT"
    if DEV_START <= ts < DEV_END:
        return "DEVELOPMENT"
    return None


def group_candidates(f: pd.DataFrame) -> list[tuple[str, list]]:
    grouped = {}
    for c in generate_candidates(f, start_index=200):
        ts = pd.Timestamp(f.iloc[c.signal_index]["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        key = f"{c.signal_index}|{ts.isoformat()}|{c.direction}"
        grouped.setdefault(key, []).append(c)
    return list(grouped.items())


def feature_row(
    *,
    item: dict,
    tf: str,
    f: pd.DataFrame,
    htf: pd.DataFrame,
    candidates: list,
    base,
    stress,
    context: dict[str, pd.DataFrame],
) -> dict | None:
    rep = candidates[0]
    i = rep.signal_index
    r = f.iloc[i]
    ts = pd.Timestamp(r["timestamp"])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    split = candidate_split(ts)
    if split is None:
        return None
    split_end = FIT_END if split == "FIT" else DEV_END
    if not full_split_horizon_ok(f, i, tf, split_end):
        return None
    if not base.entered or base.entry_index is None or base.entry_price is None:
        return None
    if stress.entered is not True:
        return None

    atr = float(r["atr14"]) if not pd.isna(r["atr14"]) else math.nan
    if not (math.isfinite(atr) and atr > 0):
        return None
    if i < 5:
        return None

    plan = construct_plan(
        f,
        rep,
        entry_mode="trigger_close",
        stop_mode="structure_plus_atr_noise",
        target_mode="hybrid_structure_volatility_ladder",
        max_entry_wait_bars=int(LOCK["candidateEngine"]["baselineMaxEntryWaitBars"]),
        max_holding_bars=int(LOCK["candidateEngine"]["baselineHoldingBars"][tf]),
    )
    if plan is None:
        return None

    entry = float(base.entry_price)
    risk = abs(entry - float(plan.stop_loss))
    if not (risk > 0 and math.isfinite(risk)):
        return None
    sign = 1.0 if rep.direction == "LONG" else -1.0
    actual_tp2_r = sign * (float(plan.tp2) - entry) / risk
    if not (math.isfinite(actual_tp2_r) and actual_tp2_r > 0):
        return None

    vals = [r["ema20"], r["ema50"], r["ema200"], f.iloc[i - 5]["ema50"],
            r["prior_high20"], r["prior_low20"], r["prior_high50"], r["prior_low50"]]
    if any(pd.isna(v) for v in vals):
        return None

    o, h, l, c = map(float, [r["open"], r["high"], r["low"], r["close"]])
    bar_range = h - l
    if bar_range <= 0:
        return None
    e20, e50, e200, e50p, ph20, pl20, ph50, pl50 = map(float, vals)
    vol_ratio = float(r["atr_pct"] / r["atr_pct_med60"]) if not pd.isna(r["atr_pct_med60"]) and float(r["atr_pct_med60"]) != 0 else math.nan

    entry_ts = pd.Timestamp(f.iloc[int(base.entry_index)]["timestamp"])
    entry_ts = entry_ts.tz_localize("UTC") if entry_ts.tzinfo is None else entry_ts.tz_convert("UTC")
    hs = htf_trend_state(htf, entry_ts)
    hs_signed = 1.0 if hs == "LONG" else (-1.0 if hs == "SHORT" else 0.0)

    setup_names = {c.setup_class for c in candidates}
    good = audit_reference_opportunity(
        f,
        i,
        rep.direction,
        positive_barrier_r=float(LOCK["opportunityAudit"]["positiveBarrierR"]),
        adverse_barrier_r=float(LOCK["opportunityAudit"]["adverseBarrierR"]),
        horizon_bars=int(LOCK["opportunityAudit"]["horizonBars"][tf]),
    )

    row = {
        "split": split,
        "signalUtc": ts.isoformat(),
        "entryUtc": entry_ts.isoformat(),
        "symbol": item["symbol"],
        "marketFamily": item["marketFamily"],
        "timeframe": tf,
        "direction": rep.direction,
        "regime": str(r["regime"]),
        "volState": str(r["vol_state"]),
        "isBreakout": int("BREAKOUT" in setup_names),
        "isPullback": int("PULLBACK" in setup_names),
        "isSweep": int("SWEEP_RECLAIM" in setup_names),
        "trend20_50_atr": (e20 - e50) / atr,
        "trend50_200_atr": (e50 - e200) / atr,
        "ema50_slope5_atr": (e50 - e50p) / atr,
        "close_ema20_atr": (c - e20) / atr,
        "range20_atr": (ph20 - pl20) / atr,
        "range50_atr": (ph50 - pl50) / atr,
        "bar_range_atr": bar_range / atr,
        "body_atr": abs(c - o) / atr,
        "clv": min(1.0, max(0.0, (c - l) / bar_range)),
        "lower_wick_atr": (min(o, c) - l) / atr,
        "upper_wick_atr": (h - max(o, c)) / atr,
        "vol_ratio": vol_ratio,
        "rv20": float(r["rv20"]) if not pd.isna(r["rv20"]) else math.nan,
        "breakout_up_atr": (c - ph20) / atr,
        "breakout_down_atr": (pl20 - c) / atr,
        "next_open_gap_atr": sign * (entry - c) / atr,
        "actual_risk_atr": risk / atr,
        "actual_tp2_r": actual_tp2_r,
        "htf_trend_signed": hs_signed,
        "htf_neutral_flag": int(htf_is_neutral(htf, entry_ts, 0.50)),
        "htfAligned": int(hs == rep.direction),
        "vix_z60": context_lookup(context["vix_z60"], entry_ts, "vix_z60"),
        "spx_trend_atr": context_lookup(context["spx_trend_atr"], entry_ts, "spx_trend_atr"),
        "dxy_trend_atr": context_lookup(context["dxy_trend_atr"], entry_ts, "dxy_trend_atr"),
        "tnx_change5": context_lookup(context["tnx_change5"], entry_ts, "tnx_change5"),
        "btc_trend_atr": context_lookup(context["btc_trend_atr"], entry_ts, "btc_trend_atr"),
        "netR": float(base.net_r),
        "stressNetR": float(stress.net_r),
        "grossR": float(base.gross_r),
        "mfeR": float(base.mfe_r),
        "maeR": float(base.mae_r),
        "auditGoodOpportunity": bool(good),
    }
    return row


def build_candidate_table() -> tuple[pd.DataFrame, list[dict]]:
    context = {
        "vix_z60": build_context_table("VIX", "vix_z60"),
        "spx_trend_atr": build_context_table("GSPC_CONTEXT", "spx_trend_atr"),
        "dxy_trend_atr": build_context_table("DXY", "dxy_trend_atr"),
        "tnx_change5": build_context_table("TNX", "tnx_change5"),
        "btc_trend_atr": build_context_table("BTC_CONTEXT", "btc_trend_atr"),
    }
    rows = []
    diagnostics = []

    for item in LOCK["universe"]:
        raw = read_csv(ROOT / "data" / "development" / f"{item['fileKey']}_1h.csv")
        for tf in item["timeframes"]:
            df = raw if tf == "1h" else resample_4h(raw)
            if len(df) < 230:
                diagnostics.append({"symbol": item["symbol"], "marketFamily": item["marketFamily"], "timeframe": tf, "status": "INSUFFICIENT_BARS", "bars": len(df)})
                continue
            f = compute_features(df)
            htf = htf_table(raw, tf, item["marketFamily"])
            total_groups = 0
            eligible = 0
            for _, candidates in group_candidates(f):
                rep = candidates[0]
                ts = pd.Timestamp(f.iloc[rep.signal_index]["timestamp"])
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                if candidate_split(ts) is None:
                    continue
                total_groups += 1
                split_end = FIT_END if ts < FIT_END else DEV_END
                if not full_split_horizon_ok(f, rep.signal_index, tf, split_end):
                    continue
                p = construct_plan(
                    f,
                    rep,
                    entry_mode="trigger_close",
                    stop_mode="structure_plus_atr_noise",
                    target_mode="hybrid_structure_volatility_ladder",
                    max_entry_wait_bars=int(LOCK["candidateEngine"]["baselineMaxEntryWaitBars"]),
                    max_holding_bars=int(LOCK["candidateEngine"]["baselineHoldingBars"][tf]),
                )
                if p is None:
                    continue
                base = simulate_trade(
                    f, p, management_mode="fixed_full_exit",
                    cost_model=market_cost(item["marketFamily"], False),
                )
                if not base.entered:
                    continue
                stress = simulate_trade(
                    f, p, management_mode="fixed_full_exit",
                    cost_model=market_cost(item["marketFamily"], True),
                )
                row = feature_row(
                    item=item, tf=tf, f=f, htf=htf, candidates=candidates,
                    base=base, stress=stress, context=context,
                )
                if row is not None:
                    rows.append(row)
                    eligible += 1
            diagnostics.append({
                "symbol": item["symbol"], "marketFamily": item["marketFamily"], "timeframe": tf,
                "status": "OK", "bars": len(df), "candidateGroups": total_groups, "eligibleRows": eligible,
            })
    return pd.DataFrame(rows), diagnostics


def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC + BINARY),
            ("cat", categorical_pipe, CATEGORICAL),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def profit_factor(a) -> float:
    arr = np.asarray(a, dtype=float)
    p = float(arr[arr > 0].sum())
    n = float(-arr[arr < 0].sum())
    return p / n if n > 0 else (999.0 if p > 0 else 0.0)


def max_drawdown(a) -> float:
    eq = peak = dd = 0.0
    for v in a:
        eq += float(v)
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return float(dd)


def circular_indices(n: int, rng, bl: int):
    out = []
    while len(out) < n:
        s = int(rng.integers(0, n))
        take = min(bl, n - len(out))
        out.extend(((s + np.arange(take)) % n).tolist())
    return np.asarray(out, dtype=int)


def bootstrap(dev: pd.DataFrame) -> dict:
    cfg = LOCK["bootstrap"]
    iters = int(cfg["iterations"])
    rng = np.random.default_rng(int(cfg["seed"]))
    bl = int(cfg["blockLengthCandidates"])
    blocks = sorted(dev[["symbol", "timeframe"]].drop_duplicates().itertuples(index=False, name=None))
    packed = {}
    for b in blocks:
        z = dev[(dev.symbol == b[0]) & (dev.timeframe == b[1])].sort_values("signalUtc").reset_index(drop=True)
        packed[b] = {
            "n": len(z),
            "selected": z.selected.to_numpy(bool),
            "net": z.netR.to_numpy(float),
        }
    abs_s = np.zeros(iters)
    delta = np.zeros(iters)
    for it in range(iters):
        ssum = 0.0
        sn = 0
        dsum = 0.0
        dn = 0
        sampled = [blocks[int(rng.integers(0, len(blocks)))] for _ in blocks]
        for b in sampled:
            p = packed[b]
            if p["n"] <= 0:
                continue
            idx = circular_indices(p["n"], rng, bl)
            sel = p["selected"][idx]
            vals = p["net"][idx]
            ssum += float(vals[sel].sum())
            sn += int(sel.sum())
            selected_utility = np.where(sel, vals, 0.0)
            dsum += float((selected_utility - vals).sum())
            dn += len(idx)
        abs_s[it] = ssum / sn if sn else 0.0
        delta[it] = dsum / dn if dn else 0.0
    return {
        "method": cfg["method"],
        "iterations": iters,
        "seed": int(cfg["seed"]),
        "blockLengthCandidates": bl,
        "probabilityPositiveNetExpectancy": float(np.mean(abs_s > 0)),
        "ci95NetExpectancyR": [float(np.quantile(abs_s, .025)), float(np.quantile(abs_s, .975))],
        "probabilityPositiveUtilityDeltaVsAllEligibleBaseline": float(np.mean(delta > 0)),
        "ci95UtilityDeltaR": [float(np.quantile(delta, .025)), float(np.quantile(delta, .975))],
    }


def metrics(dev: pd.DataFrame) -> dict:
    sel = dev[dev.selected].copy()
    eligible = len(dev)
    if sel.empty:
        return {
            "developmentEligibleRows": eligible, "selectedTrades": 0, "selectionCoverage": 0.0,
            "activeMarketFamilies": 0, "activeBlocks": 0, "netExpectancyR": 0.0,
            "stressNetExpectancyR": 0.0, "profitFactor": 0.0, "hitRate": 0.0,
            "maxDrawdownR": 0.0, "referenceOpportunityRecall": 0.0,
            "referenceOpportunityPrecision": 0.0, "positiveMarketFamilyFraction": 0.0,
            "positiveBlockFraction": 0.0, "firstHalfNetExpectancyR": 0.0,
            "secondHalfNetExpectancyR": 0.0, "positiveMonthFraction": 0.0,
            "singleMarketFamilyPositiveContribution": 0.0,
            "candidateUtilityDeltaR": float((-dev.netR).mean()),
            "allEligibleBaselineExpectancyR": float(dev.netR.mean()),
            "selectedCandidateUtilityMeanR": 0.0,
            "pWinScoreAuc": float("nan"),
            "familyExpectancyR": {}, "blockExpectancyR": {},
        }

    net = sel.netR.to_numpy(float)
    fam = sel.groupby("marketFamily").netR.mean()
    blk = sel.groupby(["symbol", "timeframe"]).netR.mean()
    fam_total = sel.groupby("marketFamily").netR.sum()
    pos = fam_total[fam_total > 0]
    concentration = float(pos.max() / pos.sum()) if len(pos) else 0.0

    s = sel.sort_values("signalUtc").reset_index(drop=True)
    half = len(s) // 2
    first = float(s.iloc[:half].netR.mean()) if half else 0.0
    second = float(s.iloc[half:].netR.mean()) if len(s) - half else 0.0
    month_total = sel.assign(month=pd.to_datetime(sel.signalUtc, utc=True).dt.strftime("%Y-%m")).groupby("month").netR.sum()

    good_all = dev.auditGoodOpportunity.astype(bool)
    good_sel = sel.auditGoodOpportunity.astype(bool)

    y = (dev.netR > 0).astype(int)
    auc = float(roc_auc_score(y, dev.pWinScore)) if y.nunique() == 2 else float("nan")
    selected_utility = np.where(dev.selected.to_numpy(bool), dev.netR.to_numpy(float), 0.0)

    return {
        "developmentEligibleRows": eligible,
        "selectedTrades": int(len(sel)),
        "selectionCoverage": float(len(sel) / max(1, eligible)),
        "activeMarketFamilies": int(len(fam)),
        "activeBlocks": int(len(blk)),
        "netExpectancyR": float(net.mean()),
        "stressNetExpectancyR": float(sel.stressNetR.mean()),
        "profitFactor": profit_factor(net),
        "hitRate": float(np.mean(net > 0)),
        "maxDrawdownR": max_drawdown(s.netR.to_numpy(float)),
        "referenceOpportunityRecall": float(good_sel.sum() / max(1, good_all.sum())),
        "referenceOpportunityPrecision": float(good_sel.mean()),
        "positiveMarketFamilyFraction": float((fam > 0).mean()) if len(fam) else 0.0,
        "positiveBlockFraction": float((blk > 0).mean()) if len(blk) else 0.0,
        "firstHalfNetExpectancyR": first,
        "secondHalfNetExpectancyR": second,
        "positiveMonthFraction": float((month_total > 0).mean()) if len(month_total) else 0.0,
        "singleMarketFamilyPositiveContribution": concentration,
        "candidateUtilityDeltaR": float((selected_utility - dev.netR.to_numpy(float)).mean()),
        "allEligibleBaselineExpectancyR": float(dev.netR.mean()),
        "selectedCandidateUtilityMeanR": float(selected_utility.mean()),
        "pWinScoreAuc": auc,
        "familyExpectancyR": {str(k): float(v) for k, v in fam.items()},
        "blockExpectancyR": {f"{k[0]}|{k[1]}": float(v) for k, v in blk.items()},
    }


def checks(fit_rows: int, m: dict, boot: dict) -> dict:
    g = LOCK["developmentGate"]
    auc = m["pWinScoreAuc"]
    c = {
        "minimumFitRows": fit_rows >= int(g["minimumFitRows"]),
        "minimumDevelopmentEligibleRows": m["developmentEligibleRows"] >= int(g["minimumDevelopmentEligibleRows"]),
        "minimumSelectedTrades": m["selectedTrades"] >= int(g["minimumSelectedTrades"]),
        "minimumActiveMarketFamilies": m["activeMarketFamilies"] >= int(g["minimumActiveMarketFamilies"]),
        "minimumActiveBlocks": m["activeBlocks"] >= int(g["minimumActiveBlocks"]),
        "selectionCoverageMin": m["selectionCoverage"] >= float(g["selectionCoverageMin"]),
        "selectionCoverageMax": m["selectionCoverage"] <= float(g["selectionCoverageMax"]),
        "netExpectancyRMin": m["netExpectancyR"] >= float(g["netExpectancyRMin"]),
        "stressNetExpectancyRMin": m["stressNetExpectancyR"] >= float(g["stressNetExpectancyRMin"]),
        "profitFactorMin": m["profitFactor"] >= float(g["profitFactorMin"]),
        "referenceOpportunityRecallMin": m["referenceOpportunityRecall"] >= float(g["referenceOpportunityRecallMin"]),
        "referenceOpportunityPrecisionMin": m["referenceOpportunityPrecision"] >= float(g["referenceOpportunityPrecisionMin"]),
        "positiveMarketFamilyFractionMin": m["positiveMarketFamilyFraction"] >= float(g["positiveMarketFamilyFractionMin"]),
        "positiveBlockFractionMin": m["positiveBlockFraction"] >= float(g["positiveBlockFractionMin"]),
        "firstHalfPositive": m["firstHalfNetExpectancyR"] > 0,
        "secondHalfPositive": m["secondHalfNetExpectancyR"] > 0,
        "positiveMonthFractionMin": m["positiveMonthFraction"] >= float(g["positiveMonthFractionMin"]),
        "bootstrapProbabilityPositiveNetMin": boot["probabilityPositiveNetExpectancy"] >= float(g["bootstrapProbabilityPositiveNetMin"]),
        "bootstrapProbabilityPositiveUtilityDeltaVsAllEligibleBaselineMin": boot["probabilityPositiveUtilityDeltaVsAllEligibleBaseline"] >= float(g["bootstrapProbabilityPositiveUtilityDeltaVsAllEligibleBaselineMin"]),
        "candidateUtilityDeltaRMin": m["candidateUtilityDeltaR"] >= float(g["candidateUtilityDeltaRMin"]),
        "pWinScoreAucMin": math.isfinite(auc) and auc >= float(g["pWinScoreAucMin"]),
        "singleMarketFamilyPositiveContributionMax": m["singleMarketFamilyPositiveContribution"] <= float(g["singleMarketFamilyPositiveContributionMax"]),
    }
    return {k: bool(v) for k, v in c.items()}


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("status") != "PASS_DATA_FREEZE":
        raise RuntimeError(f"IFE1 data freeze is not PASS: {freeze.get('status')}")
    current_lock_sha = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    if freeze.get("lockSha256") != current_lock_sha:
        raise RuntimeError("IFE1 data freeze was not generated from the current lock")

    market_manifest = {x["file"]: x for x in freeze["marketFiles"]}
    context_manifest = {x["file"]: x for x in freeze["contextFiles"]}
    for item in LOCK["universe"]:
        fn = f"{item['fileKey']}_1h.csv"
        p = ROOT / "data" / "development" / fn
        if fn not in market_manifest or hashlib.sha256(p.read_bytes()).hexdigest() != market_manifest[fn]["sha256"]:
            raise RuntimeError(f"market hash mismatch: {fn}")
    for item in LOCK["contextSeries"]:
        fn = f"{item['fileKey']}_1d.csv"
        p = ROOT / "data" / "context" / fn
        if fn not in context_manifest or hashlib.sha256(p.read_bytes()).hexdigest() != context_manifest[fn]["sha256"]:
            raise RuntimeError(f"context hash mismatch: {fn}")

    table, block_diag = build_candidate_table()
    if table.empty:
        raise RuntimeError("IFE1 candidate table empty")
    fit = table[table.split == "FIT"].copy().reset_index(drop=True)
    dev = table[table.split == "DEVELOPMENT"].copy().reset_index(drop=True)
    if len(fit) < 100 or len(dev) < 100:
        raise RuntimeError(f"IFE1 insufficient model rows fit={len(fit)} dev={len(dev)}")

    pre = make_preprocessor()
    xfit = pre.fit_transform(fit[NUMERIC + BINARY + CATEGORICAL])
    xdev = pre.transform(dev[NUMERIC + BINARY + CATEGORICAL])
    yreg = fit.netR.to_numpy(float)
    ycls = (fit.netR > 0).astype(int).to_numpy()
    if len(np.unique(ycls)) < 2:
        raise RuntimeError("IFE1 fit win label has only one class")

    ridge = Ridge(alpha=float(LOCK["model"]["ridge"]["alpha"]))
    ridge.fit(xfit, yreg)
    hcfg = LOCK["model"]["histGradientBoosting"]
    hgb = HistGradientBoostingRegressor(
        learning_rate=float(hcfg["learning_rate"]),
        max_iter=int(hcfg["max_iter"]),
        max_depth=int(hcfg["max_depth"]),
        min_samples_leaf=int(hcfg["min_samples_leaf"]),
        l2_regularization=float(hcfg["l2_regularization"]),
        random_state=int(hcfg["random_state"]),
    )
    hgb.fit(xfit, yreg)
    lcfg = LOCK["model"]["logisticWinScore"]
    logit = LogisticRegression(
        C=float(lcfg["C"]),
        class_weight=lcfg["class_weight"],
        max_iter=int(lcfg["max_iter"]),
        random_state=int(lcfg["random_state"]),
    )
    logit.fit(xfit, ycls)

    dev["ridgePredR"] = ridge.predict(xdev)
    dev["hgbPredR"] = hgb.predict(xdev)
    dev["predictedNetR"] = 0.5 * dev.ridgePredR + 0.5 * dev.hgbPredR
    dev["pWinScore"] = logit.predict_proba(xdev)[:, 1]

    rule = LOCK["selectionRule"]
    dev["selected"] = (
        (dev.predictedNetR >= float(rule["predictedNetRMin"]))
        & (dev.ridgePredR > 0)
        & (dev.hgbPredR > 0)
        & (dev.pWinScore >= float(rule["pWinScoreMin"]))
    )

    m = metrics(dev)
    boot = bootstrap(dev)
    c = checks(len(fit), m, boot)
    passed = all(c.values())
    status = "PASS_IFE1_DEVELOPMENT_REPLICATION_REQUIRED" if passed else "REJECT_IFE1_DEVELOPMENT_GATE"

    model_path = OUT / "MGPT_IFE1_FROZEN_MODEL_20260916.joblib"
    joblib.dump({"preprocessor": pre, "ridge": ridge, "hgb": hgb, "logit": logit, "lockSha256": current_lock_sha}, model_path)

    model_summary = {
        "schema": "mgpt_ife1_model_summary_v1",
        "fitRows": int(len(fit)),
        "developmentRows": int(len(dev)),
        "numericFeatures": NUMERIC,
        "binaryFeatures": BINARY,
        "categoricalFeatures": CATEGORICAL,
        "selectionRule": LOCK["selectionRule"],
        "pWinScoreSemantics": "classifier score only; not calibrated probability",
        "modelArtifactSha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (OUT / "MGPT_IFE1_MODEL_SUMMARY_20260916.json").write_text(json.dumps(model_summary, indent=2, sort_keys=True) + "\n")

    closeout = {
        "schema": "mgpt_ife1_development_closeout_v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "productionAuthority": False,
        "r15MutationAllowed": False,
        "freshValidationOpened": False,
        "sealedHoldoutOpened": False,
        "independentProviderReplicationRequiredBeforeValidation": bool(passed),
        "lockSha256": current_lock_sha,
        "implementationSpecSha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "dataFreezeSha256": hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest(),
        "fitRows": int(len(fit)),
        "metrics": m,
        "bootstrap": boot,
        "checks": c,
        "failedChecks": [k for k, v in c.items() if not v],
        "blockDiagnostics": block_diag,
        "modelArtifactSha256": model_summary["modelArtifactSha256"],
        "nextAction": (
            "Replicate the exact frozen feature table and model decision rule on independent execution-authority data before opening fresh validation."
            if passed
            else "Do not tune features, models or thresholds on this development result. Register the failed feature hypothesis; keep fresh validation and holdout sealed."
        ),
    }

    cp = OUT / "MGPT_IFE1_DEVELOPMENT_CLOSEOUT_20260916.json"
    tp = OUT / "MGPT_IFE1_CANDIDATE_TABLE_20260916.csv"
    dp = OUT / "MGPT_IFE1_DEVELOPMENT_PREDICTIONS_20260916.csv"
    cp.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    table.to_csv(tp, index=False)
    dev.to_csv(dp, index=False)

    sums = []
    for p in [cp, tp, dp, model_path, OUT / "MGPT_IFE1_MODEL_SUMMARY_20260916.json"]:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT / "MGPT_IFE1_SHA256SUMS_20260916.txt").write_text("\n".join(sums) + "\n")

    print(json.dumps({
        "status": status,
        "fitRows": len(fit),
        "failedChecks": closeout["failedChecks"],
        "metrics": {k: v for k, v in m.items() if k not in {"familyExpectancyR", "blockExpectancyR"}},
        "bootstrap": boot,
    }, indent=2))


if __name__ == "__main__":
    main()
