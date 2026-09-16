#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ite1_core import compute_features
from ite1_terminal_common import (
    baseline_reference_rows,
    candidate_key,
    cost_model,
    headline_metrics,
    htf_table,
    htf_trend_state,
    read_frozen_csv,
    reference_bootstrap,
    resample_4h,
    simulate_simple_trade,
)

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "MGPT_ITE1_C1_BREAKOUT_PAYOFF_LOCK_20260916.json"
SPEC_PATH = ROOT / "MGPT_ITE1_C1_IMPLEMENTATION_SPEC_V1_20260916.json"
FREEZE_PATH = ROOT / "results" / "MGPT_ITE1_C1_DATA_FREEZE_STATUS_20260916.json"
OUT = ROOT / "results"

LOCK = json.loads(LOCK_PATH.read_text())
SPEC = json.loads(SPEC_PATH.read_text())
DEV_START = pd.Timestamp(LOCK["windows"]["freshInstrumentDevelopment"]["start"])
DEV_END = pd.Timestamp(LOCK["windows"]["freshInstrumentDevelopment"]["endExclusive"])


def full_horizon_ok(df: pd.DataFrame, entry_i: int, hold: int) -> bool:
    end_i = entry_i + hold
    return end_i < len(df) and pd.Timestamp(df.iloc[end_i]["timestamp"]) < DEV_END


def local_trend(f: pd.DataFrame, i: int) -> str:
    if i < 205:
        return "NEUTRAL"
    r = f.iloc[i]
    vals = [r["ema20"], r["ema50"], r["ema200"], r["close"], f.iloc[i - 5]["ema50"]]
    if any(pd.isna(v) for v in vals):
        return "NEUTRAL"
    e20, e50, e200, c, e50p = map(float, vals)
    if e20 > e50 > e200 and c > e20 and e50 > e50p:
        return "LONG"
    if e20 < e50 < e200 and c < e20 and e50 < e50p:
        return "SHORT"
    return "NEUTRAL"


def choose_targets(direction: str, entry: float, risk: float, atr: float, candidates: list[float]) -> tuple[float, float, float] | None:
    vals = sorted(
        set(round(float(v), 12) for v in candidates if math.isfinite(v) and ((v > entry) if direction == "LONG" else (v < entry))),
        reverse=(direction == "SHORT"),
    )
    floors = [
        float(LOCK["riskGeometry"]["tp1MinR"]),
        float(LOCK["riskGeometry"]["tp2MinR"]),
        float(LOCK["riskGeometry"]["tp3MinR"]),
    ]
    chosen = []
    last = None
    for floor in floors:
        pick = None
        for v in vals:
            rr = (v - entry) / risk if direction == "LONG" else (entry - v) / risk
            if rr + 1e-12 < floor:
                continue
            if last is not None:
                if direction == "LONG" and v <= last + 0.10 * atr:
                    continue
                if direction == "SHORT" and v >= last - 0.10 * atr:
                    continue
            pick = float(v)
            break
        if pick is None:
            return None
        chosen.append(pick)
        last = pick
        vals = [v for v in vals if (v > pick if direction == "LONG" else v < pick)]
    return chosen[0], chosen[1], chosen[2]


def detect_c1(item: dict, tf: str, df: pd.DataFrame, f: pd.DataFrame, htf: pd.DataFrame) -> list[dict]:
    hold = int(LOCK["management"]["holdBars"][tf])
    out = []
    for i in range(205, len(f) - 1):
        ts = pd.Timestamp(f.iloc[i]["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if not (DEV_START <= ts < DEV_END):
            continue
        entry_i = i + 1
        if not full_horizon_ok(df, entry_i, hold):
            continue

        r = f.iloc[i]
        atr = float(r["atr14"]) if not pd.isna(r["atr14"]) else math.nan
        if not (math.isfinite(atr) and atr > 0):
            continue
        if str(r["vol_state"]) != "EXPANSION":
            continue

        ld = local_trend(f, i)
        if ld not in {"LONG", "SHORT"}:
            continue

        decision_time = pd.Timestamp(df.iloc[entry_i]["timestamp"])
        decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        if htf_trend_state(htf, decision_time) != ld:
            continue

        ph = float(r["prior_high50"])
        pl = float(r["prior_low50"])
        if not (math.isfinite(ph) and math.isfinite(pl) and ph > pl):
            continue
        range50 = ph - pl

        o, h, l, c = map(float, [r["open"], r["high"], r["low"], r["close"]])
        bar_range = h - l
        if bar_range <= 0:
            continue
        body = abs(c - o)
        clv = (c - l) / bar_range

        direction = None
        boundary = None
        if ld == "LONG":
            distance = c - ph
            if c > ph and bar_range >= 1.20 * atr and body >= 0.60 * atr and clv >= 0.75 and distance <= 0.75 * atr:
                direction = "LONG"
                boundary = ph
        else:
            distance = pl - c
            if c < pl and bar_range >= 1.20 * atr and body >= 0.60 * atr and clv <= 0.25 and distance <= 0.75 * atr:
                direction = "SHORT"
                boundary = pl
        if direction is None:
            continue

        entry = float(df.iloc[entry_i]["open"])
        if direction == "LONG":
            if not (entry >= boundary and entry <= c + 0.25 * atr):
                continue
            stop = boundary - 0.30 * atr
        else:
            if not (entry <= boundary and entry >= c - 0.25 * atr):
                continue
            stop = boundary + 0.30 * atr

        if (direction == "LONG" and not stop < entry) or (direction == "SHORT" and not stop > entry):
            continue
        risk = abs(entry - stop)
        risk_atr = risk / atr
        lo, hi = map(float, LOCK["riskGeometry"]["riskAtrRange"])
        if risk_atr < lo or risk_atr > hi:
            continue

        if direction == "LONG":
            candidates = [ph + 0.50 * range50, ph + 1.00 * range50, entry + 2.0 * atr, entry + 3.0 * atr, entry + 4.0 * atr]
        else:
            candidates = [pl - 0.50 * range50, pl - 1.00 * range50, entry - 2.0 * atr, entry - 3.0 * atr, entry - 4.0 * atr]

        targets = choose_targets(direction, entry, risk, atr, candidates)
        if targets is None:
            continue
        tp1, tp2, tp3 = targets
        out.append(
            {
                "key": candidate_key(item["symbol"], tf, ts, direction),
                "signalUtc": ts.isoformat(),
                "symbol": item["symbol"],
                "marketFamily": item["marketFamily"],
                "timeframe": tf,
                "direction": direction,
                "setupClass": "TREND_BREAKOUT_EXECUTABLE_PAYOFF",
                "volState": str(r["vol_state"]),
                "entryIndex": entry_i,
                "entry": entry,
                "stop": float(stop),
                "failedBreakoutLevel": float(boundary),
                "tp1": float(tp1),
                "tp2": float(tp2),
                "tp3": float(tp3),
                "tp1R": float(abs(tp1 - entry) / risk),
                "tp2R": float(abs(tp2 - entry) / risk),
                "tp3R": float(abs(tp3 - entry) / risk),
                "riskAtr": float(risk_atr),
                "breakoutDistanceAtr": float(distance / atr),
                "range50Atr": float(range50 / atr),
            }
        )
    return out


def gate_checks(metrics: dict, boot: dict) -> dict:
    g = LOCK["developmentGate"]
    checks = {
        "minimumEnteredTrades": metrics["enteredTrades"] >= int(g["minimumEnteredTrades"]),
        "minimumActiveMarketFamilies": metrics["activeMarketFamilies"] >= int(g["minimumActiveMarketFamilies"]),
        "minimumActiveBlocks": metrics["activeBlocks"] >= int(g["minimumActiveBlocks"]),
        "netExpectancyRMin": metrics["netExpectancyR"] >= float(g["netExpectancyRMin"]),
        "stressNetExpectancyRMin": metrics["stressNetExpectancyR"] >= float(g["stressNetExpectancyRMin"]),
        "profitFactorMin": metrics["profitFactor"] >= float(g["profitFactorMin"]),
        "referenceOpportunityRecallMin": metrics["referenceOpportunityRecall"] >= float(g["referenceOpportunityRecallMin"]),
        "referenceOpportunityPrecisionMin": metrics["referenceOpportunityPrecision"] >= float(g["referenceOpportunityPrecisionMin"]),
        "positiveMarketFamilyFractionMin": metrics["positiveMarketFamilyFraction"] >= float(g["positiveMarketFamilyFractionMin"]),
        "positiveBlockFractionMin": metrics["positiveBlockFraction"] >= float(g["positiveBlockFractionMin"]),
        "firstHalfPositive": metrics["firstHalfNetExpectancyR"] > 0,
        "secondHalfPositive": metrics["secondHalfNetExpectancyR"] > 0,
        "positiveMonthFractionMin": metrics["positiveMonthFraction"] >= float(g["positiveMonthFractionMin"]),
        "bootstrapProbabilityPositiveNetMin": boot["probabilityPositiveNetExpectancy"] >= float(g["bootstrapProbabilityPositiveNetMin"]),
        "bootstrapProbabilityPositiveUtilityDeltaVsBaselineMin": boot["probabilityPositiveUtilityDeltaVsBaseline"] >= float(g["bootstrapProbabilityPositiveUtilityDeltaVsBaselineMin"]),
        "singleMarketFamilyPositiveContributionMax": metrics["singleMarketFamilyPositiveContribution"] <= float(g["singleMarketFamilyPositiveContributionMax"]),
    }
    return {k: bool(v) for k, v in checks.items()}


def zero_metrics(ref: pd.DataFrame) -> dict:
    be = ref.baselineEntered.astype(bool)
    return {
        "referenceCandidateCount": int(len(ref)),
        "referenceGoodOpportunityCount": int(ref.auditGoodOpportunity.astype(bool).sum()),
        "enteredTrades": 0,
        "netExpectancyR": 0.0,
        "stressNetExpectancyR": 0.0,
        "profitFactor": 0.0,
        "hitRate": 0.0,
        "meanMfeR": 0.0,
        "meanMaeR": 0.0,
        "maxDrawdownR": 0.0,
        "referenceOpportunityRecall": 0.0,
        "referenceOpportunityPrecision": 0.0,
        "activeMarketFamilies": 0,
        "activeBlocks": 0,
        "positiveMarketFamilyFraction": 0.0,
        "positiveBlockFraction": 0.0,
        "firstHalfNetExpectancyR": 0.0,
        "secondHalfNetExpectancyR": 0.0,
        "positiveMonthFraction": 0.0,
        "singleMarketFamilyPositiveContribution": 0.0,
        "familyExpectancyR": {},
        "blockExpectancyR": {},
        "exitTypeExpectancyR": {},
        "baselineOnFreshUniverse": {
            "enteredTrades": int(be.sum()),
            "netExpectancyR": float(ref.loc[be, "baselineNetR"].mean()) if be.any() else 0.0,
            "candidateUtilityMeanR": float(ref.baselineNetR.mean()),
            "architectureCandidateUtilityMeanR": 0.0,
            "candidateUtilityDeltaR": float((-ref.baselineNetR).mean()),
        },
    }


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("status") != "PASS_DATA_FREEZE":
        raise RuntimeError(f"C1 freeze is not PASS: {freeze.get('status')}")
    manifest = {x["file"]: x for x in freeze["files"]}
    for item in LOCK["freshUniverse"]:
        fn = f"{item['fileKey']}_1h.csv"
        p = ROOT / "data" / "c1_fresh" / fn
        if fn not in manifest:
            raise RuntimeError(f"{fn} absent from C1 manifest")
        if hashlib.sha256(p.read_bytes()).hexdigest() != manifest[fn]["sha256"]:
            raise RuntimeError(f"{fn} hash mismatch")

    refs = []
    trades = []
    block_diag = []

    for item in LOCK["freshUniverse"]:
        raw = read_frozen_csv(ROOT, "c1_fresh", item)
        for tf in item["timeframes"]:
            df = raw if tf == "1h" else resample_4h(raw)
            if len(df) < 230:
                block_diag.append(
                    {"symbol": item["symbol"], "marketFamily": item["marketFamily"], "timeframe": tf, "status": "INSUFFICIENT_BARS", "bars": len(df)}
                )
                continue
            f = compute_features(df)
            hold = int(LOCK["management"]["holdBars"][tf])
            rr = baseline_reference_rows(
                lock=LOCK,
                item=item,
                tf=tf,
                df=df,
                f=f,
                dev_start=DEV_START,
                dev_end=DEV_END,
                setup_filter="BREAKOUT",
                audit_positive_r=float(LOCK["referenceAudit"]["positiveBarrierR"]),
                audit_horizon=hold,
            )
            refs.extend(rr)
            ref_keys = {x["key"] for x in rr}
            htf = htf_table(raw, tf, item["marketFamily"])
            plans = detect_c1(item, tf, df, f, htf)
            entered = 0
            for p in plans:
                if p["key"] not in ref_keys:
                    raise RuntimeError(f"C1 escaped frozen reference universe: {p['key']}")
                rb = simulate_simple_trade(
                    df=df,
                    entry_index=p["entryIndex"],
                    entry=p["entry"],
                    stop=p["stop"],
                    tp2=p["tp2"],
                    direction=p["direction"],
                    hold_bars=hold,
                    cost=cost_model(LOCK, p["marketFamily"], False),
                    failed_breakout_level=p["failedBreakoutLevel"],
                )
                rs = simulate_simple_trade(
                    df=df,
                    entry_index=p["entryIndex"],
                    entry=p["entry"],
                    stop=p["stop"],
                    tp2=p["tp2"],
                    direction=p["direction"],
                    hold_bars=hold,
                    cost=cost_model(LOCK, p["marketFamily"], True),
                    failed_breakout_level=p["failedBreakoutLevel"],
                )
                if rb is None or rs is None:
                    continue
                entered += 1
                trades.append(
                    {
                        **{k: v for k, v in p.items() if k != "entryIndex"},
                        "grossR": rb["grossR"],
                        "netR": rb["netR"],
                        "stressNetR": rs["netR"],
                        "mfeR": rb["mfeR"],
                        "maeR": rb["maeR"],
                        "exitType": rb["exitType"],
                    }
                )
            block_diag.append(
                {
                    "symbol": item["symbol"],
                    "marketFamily": item["marketFamily"],
                    "timeframe": tf,
                    "status": "OK",
                    "bars": len(df),
                    "referenceCandidates": len(rr),
                    "c1EnteredTrades": entered,
                }
            )

    ref = pd.DataFrame(refs)
    tr = pd.DataFrame(trades)
    if ref.empty:
        raise RuntimeError("C1 reference universe empty")
    if tr.empty:
        tr = pd.DataFrame(columns=["key","signalUtc","symbol","marketFamily","timeframe","direction","setupClass","volState","entry","stop","failedBreakoutLevel","tp1","tp2","tp3","tp1R","tp2R","tp3R","riskAtr","breakoutDistanceAtr","range50Atr","grossR","netR","stressNetR","mfeR","maeR","exitType"])
    if not tr.empty and tr["key"].duplicated().any():
        raise RuntimeError("C1 produced duplicate trade keys")

    amap = tr.set_index("key")["netR"].to_dict() if not tr.empty else {}
    ref["archEntered"] = ref["key"].isin(amap)
    ref["archNetR"] = ref["key"].map(amap).fillna(0.0).astype(float)

    metrics = zero_metrics(ref) if tr.empty else headline_metrics(tr, ref)
    boot = reference_bootstrap(ref, LOCK["bootstrap"])
    checks = gate_checks(metrics, boot)
    passed = all(checks.values())
    status = "PASS_C1_FRESH_INSTRUMENT_GATE_REPLICATION_REQUIRED" if passed else "REJECT_C1_FRESH_INSTRUMENT_GATE"

    closeout = {
        "schema": "mgpt_ite1_c1_development_closeout_v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "productionAuthority": False,
        "r15MutationAllowed": False,
        "timeValidationOpened": False,
        "sealedHoldoutOpened": False,
        "providerReplicationRequiredBeforeNextPhase": bool(passed),
        "lockSha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "implementationSpecSha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "dataFreezeSha256": hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest(),
        "metrics": metrics,
        "bootstrap": boot,
        "checks": checks,
        "failedChecks": [k for k, v in checks.items() if not v],
        "blockDiagnostics": block_diag,
        "terminalBudgetRuleIfB1AlsoFails": "Do not launch another price-pattern architecture in ITE1. Freeze this family and move any future work to a new data/feature program with fresh evidence and governance.",
        "nextAction": (
            "Replicate exact C1 on independent execution-authority data before any time validation."
            if passed
            else "Register C1 rejection. If B1 is also rejected, activate the prelocked terminal price-pattern stop rule; do not rescue C1 and do not open time validation/holdout."
        ),
    }

    cp = OUT / "MGPT_ITE1_C1_DEVELOPMENT_CLOSEOUT_20260916.json"
    tp = OUT / "MGPT_ITE1_C1_TRADE_LEDGER_20260916.csv"
    rp = OUT / "MGPT_ITE1_C1_REFERENCE_LEDGER_20260916.csv"
    cp.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    tr.to_csv(tp, index=False)
    ref.to_csv(rp, index=False)
    sums = []
    for p in [cp, tp, rp]:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT / "MGPT_ITE1_C1_SHA256SUMS_20260916.txt").write_text("\n".join(sums) + "\n")

    print(
        json.dumps(
            {
                "status": status,
                "failedChecks": closeout["failedChecks"],
                "metrics": {k: v for k, v in metrics.items() if k not in {"familyExpectancyR", "blockExpectancyR", "exitTypeExpectancyR"}},
                "bootstrap": boot,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
