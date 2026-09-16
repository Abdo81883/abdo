from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    one_way_fee_bps: float = 4.0
    one_way_slippage_bps: float = 2.0

    @property
    def one_way_total_rate(self) -> float:
        return (self.one_way_fee_bps + self.one_way_slippage_bps) / 10000.0


@dataclass(frozen=True)
class Candidate:
    signal_index: int
    direction: str
    setup_class: str
    setup_level: float
    regime: str
    vol_state: str


@dataclass(frozen=True)
class TradePlan:
    signal_index: int
    direction: str
    setup_class: str
    regime: str
    vol_state: str
    entry_mode: str
    entry_reference: float
    entry_zone_low: float
    entry_zone_high: float
    stop_mode: str
    stop_loss: float
    target_mode: str
    tp1: float
    tp2: float
    tp3: float
    atr_at_signal: float
    max_entry_wait_bars: int = 4
    max_holding_bars: int = 48

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TradeResult:
    entered: bool
    entry_index: Optional[int]
    exit_index: Optional[int]
    entry_price: Optional[float]
    exit_price: Optional[float]
    gross_r: float
    net_r: float
    mfe_r: float
    mae_r: float
    tp1_hit: bool
    tp2_hit: bool
    tp3_hit: bool
    stopped: bool
    timed_out: bool
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _assert_ohlc(df: pd.DataFrame) -> None:
    req = {"open", "high", "low", "close"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    if len(df) < 5:
        raise ValueError("insufficient bars")


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal decision-time features. No negative shifts are used."""
    _assert_ohlc(df)
    x = df.copy().reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if "volume" not in x.columns:
        x["volume"] = np.nan

    x["atr14"] = _atr(x, 14)
    x["ema20"] = _ema(x["close"], 20)
    x["ema50"] = _ema(x["close"], 50)
    x["ema200"] = _ema(x["close"], 200)

    x["prior_high20"] = x["high"].shift(1).rolling(20, min_periods=20).max()
    x["prior_low20"] = x["low"].shift(1).rolling(20, min_periods=20).min()
    x["prior_high50"] = x["high"].shift(1).rolling(50, min_periods=50).max()
    x["prior_low50"] = x["low"].shift(1).rolling(50, min_periods=50).min()
    x["prior_high120"] = x["high"].shift(1).rolling(120, min_periods=60).max()
    x["prior_low120"] = x["low"].shift(1).rolling(120, min_periods=60).min()

    x["swing_low10"] = x["low"].rolling(10, min_periods=10).min()
    x["swing_high10"] = x["high"].rolling(10, min_periods=10).max()

    logret = np.log(x["close"] / x["close"].shift(1))
    x["rv20"] = logret.rolling(20, min_periods=20).std(ddof=1)
    x["atr_pct"] = x["atr14"] / x["close"].replace(0, np.nan)
    x["atr_pct_med60"] = x["atr_pct"].shift(1).rolling(60, min_periods=30).median()

    trend_up = (x["ema20"] > x["ema50"]) & (x["close"] > x["ema50"])
    trend_down = (x["ema20"] < x["ema50"]) & (x["close"] < x["ema50"])
    x["regime"] = np.where(trend_up, "TREND_UP", np.where(trend_down, "TREND_DOWN", "RANGE"))

    ratio = x["atr_pct"] / x["atr_pct_med60"].replace(0, np.nan)
    x["vol_state"] = np.where(ratio >= 1.50, "EXPANSION", np.where(ratio <= 0.75, "COMPRESSION", "NORMAL"))
    return x


def generate_candidates(features: pd.DataFrame, start_index: int = 200) -> List[Candidate]:
    """Frozen Phase-0 setup family: breakout, pullback, sweep/reclaim."""
    out: List[Candidate] = []
    x = features
    for i in range(max(start_index, 1), len(x) - 1):
        r = x.iloc[i]
        vals = [r.get("atr14"), r.get("ema20"), r.get("ema50"), r.get("prior_high20"), r.get("prior_low20")]
        if any(pd.isna(v) for v in vals):
            continue
        regime = str(r["regime"])
        vol_state = str(r["vol_state"])
        o, h, l, c = map(float, [r["open"], r["high"], r["low"], r["close"]])
        e20 = float(r["ema20"])
        ph = float(r["prior_high20"])
        pl = float(r["prior_low20"])

        if regime == "TREND_UP" and c > ph:
            out.append(Candidate(i, "LONG", "BREAKOUT", ph, regime, vol_state))
        if regime == "TREND_DOWN" and c < pl:
            out.append(Candidate(i, "SHORT", "BREAKOUT", pl, regime, vol_state))

        if regime == "TREND_UP" and l <= e20 <= h and c > e20 and c > o:
            out.append(Candidate(i, "LONG", "PULLBACK", e20, regime, vol_state))
        if regime == "TREND_DOWN" and l <= e20 <= h and c < e20 and c < o:
            out.append(Candidate(i, "SHORT", "PULLBACK", e20, regime, vol_state))

        if l < pl and c > pl:
            out.append(Candidate(i, "LONG", "SWEEP_RECLAIM", pl, regime, vol_state))
        if h > ph and c < ph:
            out.append(Candidate(i, "SHORT", "SWEEP_RECLAIM", ph, regime, vol_state))
    return out


def _historical_structural_targets(x: pd.DataFrame, i: int, direction: str, ref: float, atr: float) -> List[float]:
    """Causal structural objectives using only bars <= signal bar."""
    start = max(0, i - 120)
    hist = x.iloc[start : i + 1]
    if len(hist) < 10:
        return []

    highs = hist["high"].to_numpy(dtype=float)
    lows = hist["low"].to_numpy(dtype=float)
    local_highs: List[float] = []
    local_lows: List[float] = []
    for j in range(2, len(hist) - 2):
        if highs[j] >= max(highs[j - 2 : j]) and highs[j] >= max(highs[j + 1 : j + 3]):
            local_highs.append(float(highs[j]))
        if lows[j] <= min(lows[j - 2 : j]) and lows[j] <= min(lows[j + 1 : j + 3]):
            local_lows.append(float(lows[j]))

    ph = float(x.iloc[i]["prior_high20"])
    pl = float(x.iloc[i]["prior_low20"])
    range_h = max(float(atr), ph - pl) if math.isfinite(ph - pl) else float(atr)

    if direction == "LONG":
        vals = [z for z in local_highs if z > ref + 0.25 * atr]
        vals += [ph + m * range_h for m in (0.50, 1.00, 1.50) if ph + m * range_h > ref]
        return sorted(set(round(float(z), 12) for z in vals))
    vals = [z for z in local_lows if z < ref - 0.25 * atr]
    vals += [pl - m * range_h for m in (0.50, 1.00, 1.50) if pl - m * range_h < ref]
    return sorted(set(round(float(z), 12) for z in vals), reverse=True)


def _pick_three_progressive(vals: Sequence[float], direction: str, ref: float, min_gap: float) -> Optional[Tuple[float, float, float]]:
    chosen: List[float] = []
    for z in vals:
        z = float(z)
        if direction == "LONG" and z <= ref:
            continue
        if direction == "SHORT" and z >= ref:
            continue
        if chosen and abs(z - chosen[-1]) < min_gap:
            continue
        chosen.append(z)
        if len(chosen) == 3:
            break
    if len(chosen) != 3:
        return None
    return chosen[0], chosen[1], chosen[2]


def construct_plan(
    features: pd.DataFrame,
    candidate: Candidate,
    *,
    entry_mode: str,
    stop_mode: str,
    target_mode: str,
    max_entry_wait_bars: int = 4,
    max_holding_bars: int = 48,
) -> Optional[TradePlan]:
    i = candidate.signal_index
    if i < 20 or i >= len(features) - 1:
        return None
    r = features.iloc[i]
    atr = float(r["atr14"])
    if not (math.isfinite(atr) and atr > 0):
        return None

    direction = candidate.direction
    close = float(r["close"])
    setup_level = float(candidate.setup_level)

    if entry_mode == "trigger_close":
        entry_ref = close
        zone_low = zone_high = close
    elif entry_mode in {"retest_limit", "structure_confirmed_retest"}:
        entry_ref = setup_level
        half = 0.10 * atr
        zone_low, zone_high = setup_level - half, setup_level + half
    else:
        raise ValueError(f"unknown entry_mode {entry_mode}")

    if direction == "LONG":
        structural_stop = float(r["swing_low10"])
        if stop_mode == "structure_only":
            stop = structural_stop
        elif stop_mode == "structure_plus_atr_noise":
            stop = structural_stop - 0.25 * atr
        elif stop_mode == "volatility_baseline":
            stop = entry_ref - 1.50 * atr
        else:
            raise ValueError(f"unknown stop_mode {stop_mode}")
        if not stop < entry_ref:
            return None
    else:
        structural_stop = float(r["swing_high10"])
        if stop_mode == "structure_only":
            stop = structural_stop
        elif stop_mode == "structure_plus_atr_noise":
            stop = structural_stop + 0.25 * atr
        elif stop_mode == "volatility_baseline":
            stop = entry_ref + 1.50 * atr
        else:
            raise ValueError(f"unknown stop_mode {stop_mode}")
        if not stop > entry_ref:
            return None

    structural = _historical_structural_targets(features, i, direction, entry_ref, atr)
    vol = [entry_ref + k * atr for k in (1.0, 2.0, 3.0)] if direction == "LONG" else [entry_ref - k * atr for k in (1.0, 2.0, 3.0)]

    if target_mode == "structure_ladder":
        vals = structural
    elif target_mode == "volatility_ladder":
        vals = vol
    elif target_mode == "hybrid_structure_volatility_ladder":
        vals = sorted(set(structural + vol), reverse=(direction == "SHORT"))
    else:
        raise ValueError(f"unknown target_mode {target_mode}")

    targets = _pick_three_progressive(vals, direction, entry_ref, min_gap=0.25 * atr)
    if not targets:
        return None

    return TradePlan(
        signal_index=i,
        direction=direction,
        setup_class=candidate.setup_class,
        regime=candidate.regime,
        vol_state=candidate.vol_state,
        entry_mode=entry_mode,
        entry_reference=float(entry_ref),
        entry_zone_low=float(zone_low),
        entry_zone_high=float(zone_high),
        stop_mode=stop_mode,
        stop_loss=float(stop),
        target_mode=target_mode,
        tp1=float(targets[0]),
        tp2=float(targets[1]),
        tp3=float(targets[2]),
        atr_at_signal=atr,
        max_entry_wait_bars=int(max_entry_wait_bars),
        max_holding_bars=int(max_holding_bars),
    )


def _direction_sign(direction: str) -> int:
    if direction == "LONG":
        return 1
    if direction == "SHORT":
        return -1
    raise ValueError(direction)


def _find_entry(features: pd.DataFrame, plan: TradePlan) -> Tuple[Optional[int], Optional[float]]:
    i = plan.signal_index
    last = min(len(features) - 1, i + plan.max_entry_wait_bars + 1)
    if i + 1 > last:
        return None, None

    if plan.entry_mode == "trigger_close":
        return i + 1, float(features.iloc[i + 1]["open"])

    limit = plan.entry_reference
    if plan.entry_mode == "retest_limit":
        for j in range(i + 1, last + 1):
            r = features.iloc[j]
            if float(r["low"]) <= limit <= float(r["high"]):
                return j, float(limit)
        return None, None

    if plan.entry_mode == "structure_confirmed_retest":
        for j in range(i + 1, last):
            r = features.iloc[j]
            touched = float(r["low"]) <= limit <= float(r["high"])
            confirmed = float(r["close"]) > limit if plan.direction == "LONG" else float(r["close"]) < limit
            if touched and confirmed:
                return j + 1, float(features.iloc[j + 1]["open"])
        return None, None

    raise ValueError(plan.entry_mode)


def _cost_r(entry: float, exit_price: float, risk_abs: float, weight: float, cost: CostModel) -> float:
    rate = cost.one_way_total_rate
    return weight * rate * (abs(entry) + abs(exit_price)) / risk_abs


def simulate_trade(
    features: pd.DataFrame,
    plan: TradePlan,
    *,
    management_mode: str,
    cost_model: CostModel = CostModel(),
) -> TradeResult:
    """Causal OHLC execution simulation with conservative same-bar ordering."""
    entry_i, entry = _find_entry(features, plan)
    if entry_i is None or entry is None:
        return TradeResult(False, None, None, None, None, 0.0, 0.0, 0.0, 0.0, False, False, False, False, False, "NO_ENTRY")

    sign = _direction_sign(plan.direction)
    stop = float(plan.stop_loss)
    if (sign == 1 and not stop < entry) or (sign == -1 and not stop > entry):
        return TradeResult(False, entry_i, None, entry, None, 0.0, 0.0, 0.0, 0.0, False, False, False, False, False, "ENTRY_INVALIDATES_STOP_GEOMETRY")

    risk = abs(entry - stop)
    if not (risk > 0 and math.isfinite(risk)):
        return TradeResult(False, entry_i, None, entry, None, 0.0, 0.0, 0.0, 0.0, False, False, False, False, False, "INVALID_RISK")

    if sign == 1 and not (plan.tp1 > entry and plan.tp2 > plan.tp1 and plan.tp3 > plan.tp2):
        return TradeResult(False, entry_i, None, entry, None, 0.0, 0.0, 0.0, 0.0, False, False, False, False, False, "INVALID_TARGET_GEOMETRY")
    if sign == -1 and not (plan.tp1 < entry and plan.tp2 < plan.tp1 and plan.tp3 < plan.tp2):
        return TradeResult(False, entry_i, None, entry, None, 0.0, 0.0, 0.0, 0.0, False, False, False, False, False, "INVALID_TARGET_GEOMETRY")

    if management_mode not in {"fixed_full_exit", "tp1_partial_then_structure_trail", "tp1_partial_plus_time_stop"}:
        raise ValueError(management_mode)

    max_i = min(len(features) - 1, entry_i + plan.max_holding_bars)
    current_stop = stop
    gross_r = 0.0
    net_r = 0.0
    remaining = 1.0
    tp1_hit = tp2_hit = tp3_hit = False
    stopped = timed_out = False
    exit_i: Optional[int] = None
    exit_price: Optional[float] = None
    mfe_r, mae_r = 0.0, 0.0

    def close_piece(price: float, weight: float) -> None:
        nonlocal gross_r, net_r, remaining, exit_price
        r_mult = sign * (price - entry) / risk
        gross_r += weight * r_mult
        net_r += weight * r_mult - _cost_r(entry, price, risk, weight, cost_model)
        remaining = max(0.0, remaining - weight)
        exit_price = price

    for j in range(entry_i, max_i + 1):
        r = features.iloc[j]
        h, l, c = float(r["high"]), float(r["low"]), float(r["close"])
        favorable = (h - entry) / risk if sign == 1 else (entry - l) / risk
        adverse = (entry - l) / risk if sign == 1 else (h - entry) / risk
        mfe_r = max(mfe_r, favorable)
        mae_r = max(mae_r, adverse)

        stop_touched = l <= current_stop if sign == 1 else h >= current_stop
        if stop_touched and remaining > 1e-12:
            close_piece(current_stop, remaining)
            stopped = True
            exit_i = j
            break

        if management_mode == "fixed_full_exit":
            target = plan.tp2
            hit = h >= target if sign == 1 else l <= target
            if hit:
                tp1_hit = (h >= plan.tp1) if sign == 1 else (l <= plan.tp1)
                tp2_hit = True
                close_piece(target, remaining)
                exit_i = j
                break
            continue

        if not tp1_hit:
            hit1 = h >= plan.tp1 if sign == 1 else l <= plan.tp1
            if hit1:
                tp1_hit = True
                w = min(0.33 if management_mode == "tp1_partial_then_structure_trail" else 0.50, remaining)
                close_piece(plan.tp1, w)
                if management_mode == "tp1_partial_plus_time_stop":
                    current_stop = max(current_stop, entry) if sign == 1 else min(current_stop, entry)

        if management_mode == "tp1_partial_then_structure_trail" and tp1_hit and remaining > 1e-12:
            start = max(entry_i, j - 4)
            if sign == 1:
                structure = float(features.iloc[start : j + 1]["low"].min()) - 0.10 * plan.atr_at_signal
                current_stop = max(current_stop, min(structure, c - 0.10 * plan.atr_at_signal))
            else:
                structure = float(features.iloc[start : j + 1]["high"].max()) + 0.10 * plan.atr_at_signal
                current_stop = min(current_stop, max(structure, c + 0.10 * plan.atr_at_signal))

        if tp1_hit and not tp2_hit and remaining > 1e-12:
            hit2 = h >= plan.tp2 if sign == 1 else l <= plan.tp2
            if hit2:
                tp2_hit = True
                w = min(0.33 if management_mode == "tp1_partial_then_structure_trail" else remaining, remaining)
                close_piece(plan.tp2, w)
                if management_mode == "tp1_partial_plus_time_stop" and remaining <= 1e-12:
                    exit_i = j
                    break

        if management_mode == "tp1_partial_then_structure_trail" and tp2_hit and not tp3_hit and remaining > 1e-12:
            hit3 = h >= plan.tp3 if sign == 1 else l <= plan.tp3
            if hit3:
                tp3_hit = True
                close_piece(plan.tp3, remaining)
                exit_i = j
                break

    if remaining > 1e-12:
        exit_i = max_i
        exit_price = float(features.iloc[max_i]["close"])
        close_piece(exit_price, remaining)
        timed_out = True

    return TradeResult(
        entered=True,
        entry_index=entry_i,
        exit_index=exit_i,
        entry_price=entry,
        exit_price=exit_price,
        gross_r=float(gross_r),
        net_r=float(net_r),
        mfe_r=float(mfe_r),
        mae_r=float(mae_r),
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        tp3_hit=tp3_hit,
        stopped=stopped,
        timed_out=timed_out,
        reason="CLOSED",
    )


def audit_reference_opportunity(
    features: pd.DataFrame,
    signal_index: int,
    direction: str,
    *,
    positive_barrier_r: float = 1.5,
    adverse_barrier_r: float = 1.0,
    horizon_bars: int = 48,
) -> Optional[bool]:
    """Ex-post audit label only; never a model feature."""
    i = signal_index
    if i + 1 >= len(features):
        return None
    atr = float(features.iloc[i]["atr14"])
    if not (math.isfinite(atr) and atr > 0):
        return None
    entry = float(features.iloc[i + 1]["open"])
    sign = _direction_sign(direction)
    adverse = entry - sign * adverse_barrier_r * atr
    positive = entry + sign * positive_barrier_r * atr
    last = min(len(features) - 1, i + horizon_bars)
    for j in range(i + 1, last + 1):
        h, l = float(features.iloc[j]["high"]), float(features.iloc[j]["low"])
        adverse_hit = l <= adverse if sign == 1 else h >= adverse
        positive_hit = h >= positive if sign == 1 else l <= positive
        if adverse_hit:
            return False
        if positive_hit:
            return True
    return False


def summarize_trade_results(results: Iterable[TradeResult]) -> Dict[str, float]:
    entered = [r for r in results if r.entered]
    if not entered:
        return {
            "trades": 0,
            "netExpectancyR": 0.0,
            "profitFactor": 0.0,
            "hitRate": 0.0,
            "meanMfeR": 0.0,
            "meanMaeR": 0.0,
        }
    arr = np.array([r.net_r for r in entered], dtype=float)
    pos = arr[arr > 0].sum()
    neg = -arr[arr < 0].sum()
    pf = float(pos / neg) if neg > 0 else (999.0 if pos > 0 else 0.0)
    return {
        "trades": int(len(entered)),
        "netExpectancyR": float(arr.mean()),
        "profitFactor": pf,
        "hitRate": float(np.mean(arr > 0)),
        "meanMfeR": float(np.mean([r.mfe_r for r in entered])),
        "meanMaeR": float(np.mean([r.mae_r for r in entered])),
    }
