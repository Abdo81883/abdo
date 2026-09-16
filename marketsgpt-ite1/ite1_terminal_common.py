#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ite1_core import CostModel, audit_reference_opportunity, compute_features, construct_plan, generate_candidates, simulate_trade


def read_frozen_csv(root: Path, data_dir: str, item: dict) -> pd.DataFrame:
    p = root / "data" / data_dir / f"{item['fileKey']}_1h.csv"
    if not p.exists():
        raise RuntimeError(f"missing frozen file {p.name}")
    x = pd.read_csv(p)
    req = {"timestamp", "open", "high", "low", "close"}
    if not req.issubset(x.columns):
        raise RuntimeError(f"{p.name}: missing {sorted(req - set(x.columns))}")
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    if "volume" not in x.columns:
        x["volume"] = np.nan
    x = (
        x.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    return x


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    z = df.set_index("timestamp")
    n = z["close"].resample("4h", origin="epoch", label="left", closed="left").count()
    o = z.resample("4h", origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return o[n == 4].dropna(subset=["open", "high", "low", "close"]).reset_index()


def resample_day(df: pd.DataFrame) -> pd.DataFrame:
    z = df.set_index("timestamp")
    n = z["close"].resample("1D", origin="epoch", label="left", closed="left").count()
    o = z.resample("1D", origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return o[n > 0].dropna(subset=["open", "high", "low", "close"]).reset_index()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def context_features(df: pd.DataFrame, duration: pd.Timedelta) -> pd.DataFrame:
    x = df.copy()
    x["ema20"] = x["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    x["ema50_lag5"] = x["ema50"].shift(5)
    x["atr14"] = _atr(x, 14)
    x["end_time"] = x["timestamp"] + duration
    return x


def htf_table(raw: pd.DataFrame, tf: str, family: str) -> pd.DataFrame:
    if tf == "1h" and family not in {"EQUITY", "INDEX"}:
        return context_features(resample_4h(raw), pd.Timedelta(hours=4))
    return context_features(resample_day(raw), pd.Timedelta(days=1))


def last_completed_htf_row(htf: pd.DataFrame, decision_time: pd.Timestamp):
    if htf.empty:
        return None
    ends = htf["end_time"].astype("int64").to_numpy()
    pos = int(np.searchsorted(ends, decision_time.value, side="right") - 1)
    if pos < 0:
        return None
    return htf.iloc[pos]


def htf_trend_state(htf: pd.DataFrame, decision_time: pd.Timestamp) -> str:
    r = last_completed_htf_row(htf, decision_time)
    if r is None:
        return "UNKNOWN"
    vals = [r["ema20"], r["ema50"], r["ema50_lag5"], r["close"]]
    if any(pd.isna(v) for v in vals):
        return "UNKNOWN"
    e20, e50, e50p, c = map(float, vals)
    if e20 > e50 and c > e20 and e50 > e50p:
        return "LONG"
    if e20 < e50 and c < e20 and e50 < e50p:
        return "SHORT"
    return "NEUTRAL"


def htf_is_neutral(htf: pd.DataFrame, decision_time: pd.Timestamp, threshold_atr: float) -> bool:
    r = last_completed_htf_row(htf, decision_time)
    if r is None:
        return False
    vals = [r["ema20"], r["ema50"], r["ema50_lag5"], r["atr14"]]
    if any(pd.isna(v) for v in vals):
        return False
    e20, e50, e50p, atr = map(float, vals)
    return atr > 0 and abs(e20 - e50) <= threshold_atr * atr and abs(e50 - e50p) <= threshold_atr * atr


def cost_model(lock: dict, family: str, stress: bool = False) -> CostModel:
    b = float(lock["costModelOneWayBps"][family]["stress" if stress else "base"])
    return CostModel(one_way_fee_bps=b, one_way_slippage_bps=0.0)


def candidate_key(symbol: str, tf: str, ts: pd.Timestamp, direction: str) -> str:
    return f"{symbol}|{tf}|{ts.isoformat()}|{direction}"


def baseline_reference_rows(
    *,
    lock: dict,
    item: dict,
    tf: str,
    df: pd.DataFrame,
    f: pd.DataFrame,
    dev_start: pd.Timestamp,
    dev_end: pd.Timestamp,
    setup_filter: str,
    audit_positive_r: float,
    audit_horizon: int,
) -> List[dict]:
    out = []
    baseline_hold = 48 if tf == "1h" else 30
    max_wait = 4
    dedup: Dict[str, object] = {}
    for c in generate_candidates(f, start_index=200):
        if c.setup_class != setup_filter:
            continue
        ts = pd.Timestamp(f.iloc[c.signal_index]["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if not (dev_start <= ts < dev_end):
            continue
        worst = c.signal_index + max_wait + baseline_hold + 1
        if worst >= len(df):
            continue
        if pd.Timestamp(df.iloc[worst]["timestamp"]) >= dev_end:
            continue
        k = candidate_key(item["symbol"], tf, ts, c.direction)
        if k not in dedup:
            dedup[k] = c

    for k, c in dedup.items():
        ts = pd.Timestamp(f.iloc[c.signal_index]["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        good = audit_reference_opportunity(
            f,
            c.signal_index,
            c.direction,
            positive_barrier_r=float(audit_positive_r),
            adverse_barrier_r=1.0,
            horizon_bars=int(audit_horizon),
        )
        p = construct_plan(
            f,
            c,
            entry_mode="trigger_close",
            stop_mode="structure_plus_atr_noise",
            target_mode="hybrid_structure_volatility_ladder",
            max_entry_wait_bars=max_wait,
            max_holding_bars=baseline_hold,
        )
        br = None
        if p is not None:
            br = simulate_trade(
                f,
                p,
                management_mode="fixed_full_exit",
                cost_model=cost_model(lock, item["marketFamily"], False),
            )
        out.append(
            {
                "key": k,
                "symbol": item["symbol"],
                "marketFamily": item["marketFamily"],
                "timeframe": tf,
                "signalUtc": ts.isoformat(),
                "direction": c.direction,
                "referenceSetup": c.setup_class,
                "auditGoodOpportunity": bool(good),
                "baselineEntered": bool(br.entered) if br else False,
                "baselineNetR": float(br.net_r) if br and br.entered else 0.0,
            }
        )
    return out


def simulate_simple_trade(
    *,
    df: pd.DataFrame,
    entry_index: int,
    entry: float,
    stop: float,
    tp2: float,
    direction: str,
    hold_bars: int,
    cost: CostModel,
    failed_breakout_level: float | None = None,
) -> dict | None:
    sign = 1 if direction == "LONG" else -1
    if entry_index + hold_bars >= len(df):
        return None
    if (sign == 1 and not stop < entry < tp2) or (sign == -1 and not stop > entry > tp2):
        return None
    risk = abs(entry - stop)
    if not (math.isfinite(risk) and risk > 0):
        return None
    mfe = 0.0
    mae = 0.0
    exit_price = None
    exit_index = None
    exit_type = None
    for j in range(entry_index, entry_index + hold_bars + 1):
        r = df.iloc[j]
        h, l, c = map(float, [r["high"], r["low"], r["close"]])
        mfe = max(mfe, (h - entry) / risk if sign == 1 else (entry - l) / risk)
        mae = max(mae, (entry - l) / risk if sign == 1 else (h - entry) / risk)

        stop_hit = l <= stop if sign == 1 else h >= stop
        if stop_hit:
            exit_price = stop
            exit_index = j
            exit_type = "STOP"
            break

        tp_hit = h >= tp2 if sign == 1 else l <= tp2
        if tp_hit:
            exit_price = tp2
            exit_index = j
            exit_type = "TP2"
            break

        if failed_breakout_level is not None:
            failed = c < failed_breakout_level if sign == 1 else c > failed_breakout_level
            if failed:
                exit_price = c
                exit_index = j
                exit_type = "FAILED_BREAKOUT"
                break

    if exit_price is None:
        exit_index = entry_index + hold_bars
        exit_price = float(df.iloc[exit_index]["close"])
        exit_type = "TIMEOUT"

    gross = sign * (exit_price - entry) / risk
    friction = cost.one_way_total_rate * (abs(entry) + abs(exit_price)) / risk
    return {
        "grossR": float(gross),
        "netR": float(gross - friction),
        "mfeR": float(mfe),
        "maeR": float(mae),
        "exitPrice": float(exit_price),
        "exitIndex": int(exit_index),
        "exitType": exit_type,
    }


def profit_factor(a: Iterable[float]) -> float:
    arr = np.asarray(list(a), dtype=float)
    pos = float(arr[arr > 0].sum())
    neg = float(-arr[arr < 0].sum())
    return pos / neg if neg > 0 else (999.0 if pos > 0 else 0.0)


def max_drawdown_r(a: Iterable[float]) -> float:
    eq = 0.0
    peak = 0.0
    dd = 0.0
    for x in a:
        eq += float(x)
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return float(dd)


def circular_indices(n: int, rng: np.random.Generator, block_len: int) -> np.ndarray:
    out = []
    while len(out) < n:
        s = int(rng.integers(0, n))
        take = min(block_len, n - len(out))
        out.extend(((s + np.arange(take)) % n).tolist())
    return np.asarray(out, dtype=int)


def reference_bootstrap(ref: pd.DataFrame, cfg: dict) -> dict:
    iters = int(cfg["iterations"])
    seed = int(cfg["seed"])
    bl = int(cfg["blockLengthCandidates"])
    rng = np.random.default_rng(seed)
    blocks = sorted(ref[["symbol", "timeframe"]].drop_duplicates().itertuples(index=False, name=None))
    packed = {}
    for b in blocks:
        z = ref[(ref.symbol == b[0]) & (ref.timeframe == b[1])].sort_values("signalUtc").reset_index(drop=True)
        packed[b] = {
            "n": len(z),
            "entered": z.archEntered.to_numpy(bool),
            "arch": z.archNetR.to_numpy(float),
            "base": z.baselineNetR.to_numpy(float),
        }

    abs_s = np.zeros(iters)
    delta = np.zeros(iters)
    for it in range(iters):
        tr_sum = 0.0
        tr_n = 0
        du = 0.0
        u_n = 0
        sampled = [blocks[int(rng.integers(0, len(blocks)))] for _ in blocks]
        for b in sampled:
            p = packed[b]
            if p["n"] <= 0:
                continue
            idx = circular_indices(p["n"], rng, bl)
            ent = p["entered"][idx]
            aa = p["arch"][idx]
            bb = p["base"][idx]
            tr_sum += float(aa[ent].sum())
            tr_n += int(ent.sum())
            du += float((aa - bb).sum())
            u_n += len(idx)
        abs_s[it] = tr_sum / tr_n if tr_n else 0.0
        delta[it] = du / u_n if u_n else 0.0

    return {
        "method": cfg["method"],
        "iterations": iters,
        "seed": seed,
        "blockLengthCandidates": bl,
        "probabilityPositiveNetExpectancy": float(np.mean(abs_s > 0)),
        "ci95NetExpectancyR": [float(np.quantile(abs_s, 0.025)), float(np.quantile(abs_s, 0.975))],
        "probabilityPositiveUtilityDeltaVsBaseline": float(np.mean(delta > 0)),
        "ci95UtilityDeltaVsBaseline": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
    }


def headline_metrics(tr: pd.DataFrame, ref: pd.DataFrame) -> dict:
    if tr.empty:
        raise RuntimeError("no entered architecture trades")
    net = tr.netR.to_numpy(float)
    stress = tr.stressNetR.to_numpy(float)
    good = ref.auditGoodOpportunity.astype(bool)
    captured = good & ref.archEntered.astype(bool)

    fam = tr.groupby("marketFamily").netR.mean()
    blk = tr.groupby(["symbol", "timeframe"]).netR.mean()
    fam_total = tr.groupby("marketFamily").netR.sum()
    pos = fam_total[fam_total > 0]
    concentration = float(pos.max() / pos.sum()) if len(pos) else 0.0

    s = tr.sort_values("signalUtc").reset_index(drop=True)
    half = len(s) // 2
    first = float(s.iloc[:half].netR.mean()) if half else 0.0
    second = float(s.iloc[half:].netR.mean()) if len(s) - half else 0.0

    months = tr.assign(month=pd.to_datetime(tr.signalUtc, utc=True).dt.strftime("%Y-%m")).groupby("month").netR.sum()
    base_enter = ref[ref.baselineEntered.astype(bool)]

    return {
        "referenceCandidateCount": int(len(ref)),
        "referenceGoodOpportunityCount": int(good.sum()),
        "enteredTrades": int(len(tr)),
        "netExpectancyR": float(net.mean()),
        "stressNetExpectancyR": float(stress.mean()),
        "profitFactor": profit_factor(net),
        "hitRate": float(np.mean(net > 0)),
        "meanMfeR": float(tr.mfeR.mean()),
        "meanMaeR": float(tr.maeR.mean()),
        "maxDrawdownR": max_drawdown_r(s.netR.to_numpy(float)),
        "referenceOpportunityRecall": float(captured.sum() / max(1, good.sum())),
        "referenceOpportunityPrecision": float(captured.sum() / max(1, len(tr))),
        "activeMarketFamilies": int(len(fam)),
        "activeBlocks": int(len(blk)),
        "positiveMarketFamilyFraction": float((fam > 0).mean()) if len(fam) else 0.0,
        "positiveBlockFraction": float((blk > 0).mean()) if len(blk) else 0.0,
        "firstHalfNetExpectancyR": first,
        "secondHalfNetExpectancyR": second,
        "positiveMonthFraction": float((months > 0).mean()) if len(months) else 0.0,
        "singleMarketFamilyPositiveContribution": concentration,
        "familyExpectancyR": {str(k): float(v) for k, v in fam.items()},
        "blockExpectancyR": {f"{k[0]}|{k[1]}": float(v) for k, v in blk.items()},
        "exitTypeExpectancyR": {str(k): float(v) for k, v in tr.groupby("exitType").netR.mean().items()},
        "baselineOnFreshUniverse": {
            "enteredTrades": int(len(base_enter)),
            "netExpectancyR": float(base_enter.baselineNetR.mean()) if len(base_enter) else 0.0,
            "candidateUtilityMeanR": float(ref.baselineNetR.mean()),
            "architectureCandidateUtilityMeanR": float(ref.archNetR.mean()),
            "candidateUtilityDeltaR": float((ref.archNetR - ref.baselineNetR).mean()),
        },
    }
