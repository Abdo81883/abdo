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
    htf_is_neutral,
    htf_table,
    read_frozen_csv,
    reference_bootstrap,
    resample_4h,
    simulate_simple_trade,
)

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "MGPT_ITE1_B1_RANGE_REVERSION_LOCK_20260916.json"
SPEC_PATH = ROOT / "MGPT_ITE1_B1_IMPLEMENTATION_SPEC_V1_20260916.json"
FREEZE_PATH = ROOT / "results" / "MGPT_ITE1_B1_DATA_FREEZE_STATUS_20260916.json"
OUT = ROOT / "results"

LOCK = json.loads(LOCK_PATH.read_text())
SPEC = json.loads(SPEC_PATH.read_text())
DEV_START = pd.Timestamp(LOCK["windows"]["freshInstrumentDevelopment"]["start"])
DEV_END = pd.Timestamp(LOCK["windows"]["freshInstrumentDevelopment"]["endExclusive"])


def full_horizon_ok(df: pd.DataFrame, entry_i: int, hold: int) -> bool:
    end_i = entry_i + hold
    return end_i < len(df) and pd.Timestamp(df.iloc[end_i]["timestamp"]) < DEV_END


def local_neutral(f: pd.DataFrame, i: int) -> bool:
    if i < 5:
        return False
    r = f.iloc[i]
    vals = [r["ema20"], r["ema50"], f.iloc[i - 5]["ema50"], r["atr14"]]
    if any(pd.isna(v) for v in vals):
        return False
    e20, e50, e50p, atr = map(float, vals)
    return atr > 0 and abs(e20 - e50) <= 0.35 * atr and abs(e50 - e50p) <= 0.35 * atr


def detect_b1(item: dict, tf: str, df: pd.DataFrame, f: pd.DataFrame, htf: pd.DataFrame) -> list[dict]:
    hold = int(LOCK["management"]["holdBars"][tf])
    out = []
    for i in range(200, len(f) - 1):
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
        if str(r["vol_state"]) not in {"COMPRESSION", "NORMAL"}:
            continue
        if not local_neutral(f, i):
            continue

        decision_time = pd.Timestamp(df.iloc[entry_i]["timestamp"])
        decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        if not htf_is_neutral(htf, decision_time, 0.50):
            continue

        ph = float(r["prior_high20"])
        pl = float(r["prior_low20"])
        if not (math.isfinite(ph) and math.isfinite(pl) and ph > pl):
            continue
        range_w = ph - pl
        ratio = range_w / atr
        if ratio < 3.0 or ratio > 8.0:
            continue

        o, h, l, c = map(float, [r["open"], r["high"], r["low"], r["close"]])
        bar_range = h - l
        if bar_range <= 0:
            continue
        clv = (c - l) / bar_range
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        midpoint = pl + 0.50 * range_w
        uq = pl + 0.75 * range_w
        lq = pl + 0.25 * range_w

        setups = []
        if l < pl - 0.10 * atr and c > pl and clv >= 0.65 and lower_wick >= 0.35 * atr:
            setups.append(("LONG", l - 0.20 * atr, [midpoint, uq, ph]))
        if h > ph + 0.10 * atr and c < ph and clv <= 0.35 and upper_wick >= 0.35 * atr:
            setups.append(("SHORT", h + 0.20 * atr, [midpoint, lq, pl]))

        for direction, stop, targets in setups:
            entry = float(df.iloc[entry_i]["open"])
            if direction == "LONG":
                if not (entry > stop and entry <= midpoint):
                    continue
                if not (entry < targets[0] < targets[1] < targets[2]):
                    continue
            else:
                if not (entry < stop and entry >= midpoint):
                    continue
                if not (entry > targets[0] > targets[1] > targets[2]):
                    continue

            risk = abs(entry - stop)
            risk_atr = risk / atr
            lo, hi = map(float, LOCK["riskGeometry"]["riskAtrRange"])
            if risk_atr < lo or risk_atr > hi:
                continue

            floors = [
                float(LOCK["riskGeometry"]["tp1MinR"]),
                float(LOCK["riskGeometry"]["tp2MinR"]),
                float(LOCK["riskGeometry"]["tp3MinR"]),
            ]
            rs = [
                (t - entry) / risk if direction == "LONG" else (entry - t) / risk
                for t in targets
            ]
            if any(rs[j] + 1e-12 < floors[j] for j in range(3)):
                continue
            if abs(targets[1] - targets[0]) < 0.10 * atr or abs(targets[2] - targets[1]) < 0.10 * atr:
                continue

            out.append(
                {
                    "key": candidate_key(item["symbol"], tf, ts, direction),
                    "signalUtc": ts.isoformat(),
                    "symbol": item["symbol"],
                    "marketFamily": item["marketFamily"],
                    "timeframe": tf,
                    "direction": direction,
                    "setupClass": "RANGE_REVERSION_LIQUIDITY_RECLAIM",
                    "volState": str(r["vol_state"]),
                    "entryIndex": entry_i,
                    "entry": entry,
                    "stop": float(stop),
                    "tp1": float(targets[0]),
                    "tp2": float(targets[1]),
                    "tp3": float(targets[2]),
                    "tp1R": float(rs[0]),
                    "tp2R": float(rs[1]),
                    "tp3R": float(rs[2]),
                    "riskAtr": float(risk_atr),
                    "rangeAtr": float(ratio),
                    "atr": float(atr),
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


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("status") != "PASS_DATA_FREEZE":
        raise RuntimeError(f"B1 freeze is not PASS: {freeze.get('status')}")
    manifest = {x["file"]: x for x in freeze["files"]}
    for item in LOCK["freshUniverse"]:
        fn = f"{item['fileKey']}_1h.csv"
        p = ROOT / "data" / "b1_fresh" / fn
        if fn not in manifest:
            raise RuntimeError(f"{fn} absent from B1 manifest")
        if hashlib.sha256(p.read_bytes()).hexdigest() != manifest[fn]["sha256"]:
            raise RuntimeError(f"{fn} hash mismatch")

    refs = []
    trades = []
    block_diag = []

    for item in LOCK["freshUniverse"]:
        raw = read_frozen_csv(ROOT, "b1_fresh", item)
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
                setup_filter="SWEEP_RECLAIM",
                audit_positive_r=float(LOCK["referenceAudit"]["positiveBarrierR"]),
                audit_horizon=hold,
            )
            refs.extend(rr)
            ref_keys = {x["key"] for x in rr}
            htf = htf_table(raw, tf, item["marketFamily"])
            plans = detect_b1(item, tf, df, f, htf)
            entered = 0
            for p in plans:
                if p["key"] not in ref_keys:
                    raise RuntimeError(f"B1 escaped frozen reference universe: {p['key']}")
                rb = simulate_simple_trade(
                    df=df,
                    entry_index=p["entryIndex"],
                    entry=p["entry"],
                    stop=p["stop"],
                    tp2=p["tp2"],
                    direction=p["direction"],
                    hold_bars=hold,
                    cost=cost_model(LOCK, p["marketFamily"], False),
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
                )
                if rb is None or rs is None:
                    continue
                entered += 1
                trades.append(
                    {
                        **{k: v for k, v in p.items() if k not in {"entryIndex", "atr"}},
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
                    "b1EnteredTrades": entered,
                }
            )

    ref = pd.DataFrame(refs)
    tr = pd.DataFrame(trades)
    if ref.empty:
        raise RuntimeError("B1 reference universe empty")
    if tr.empty:
        # This is a valid scientific outcome. Create zero-trade closeout rather than crash.
        tr = pd.DataFrame(columns=["key","signalUtc","symbol","marketFamily","timeframe","direction","setupClass","volState","entry","stop","tp1","tp2","tp3","tp1R","tp2R","tp3R","riskAtr","rangeAtr","grossR","netR","stressNetR","mfeR","maeR","exitType"])

    if not tr.empty and tr["key"].duplicated().any():
        raise RuntimeError("B1 produced duplicate trade keys")

    amap = tr.set_index("key")["netR"].to_dict() if not tr.empty else {}
    ref["archEntered"] = ref["key"].isin(amap)
    ref["archNetR"] = ref["key"].map(amap).fillna(0.0).astype(float)

    if tr.empty:
        metrics = {
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
                "enteredTrades": int(ref.baselineEntered.astype(bool).sum()),
                "netExpectancyR": float(ref.loc[ref.baselineEntered.astype(bool), "baselineNetR"].mean()) if ref.baselineEntered.astype(bool).any() else 0.0,
                "candidateUtilityMeanR": float(ref.baselineNetR.mean()),
                "architectureCandidateUtilityMeanR": 0.0,
                "candidateUtilityDeltaR": float((-ref.baselineNetR).mean()),
            },
        }
    else:
        metrics = headline_metrics(tr, ref)

    boot = reference_bootstrap(ref, LOCK["bootstrap"])
    checks = gate_checks(metrics, boot)
    passed = all(checks.values())
    status = "PASS_B1_FRESH_INSTRUMENT_GATE_REPLICATION_REQUIRED" if passed else "REJECT_B1_FRESH_INSTRUMENT_GATE"

    closeout = {
        "schema": "mgpt_ite1_b1_development_closeout_v1",
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
        "nextAction": (
            "Replicate exact B1 on independent execution-authority data before any time validation."
            if passed
            else "Register B1 rejection. Do not tune B1 or open time validation/holdout. C1 remains the only other prelocked price-pattern architecture in the finite ITE1 budget."
        ),
    }

    cp = OUT / "MGPT_ITE1_B1_DEVELOPMENT_CLOSEOUT_20260916.json"
    tp = OUT / "MGPT_ITE1_B1_TRADE_LEDGER_20260916.csv"
    rp = OUT / "MGPT_ITE1_B1_REFERENCE_LEDGER_20260916.csv"
    cp.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    tr.to_csv(tp, index=False)
    ref.to_csv(rp, index=False)
    sums = []
    for p in [cp, tp, rp]:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT / "MGPT_ITE1_B1_SHA256SUMS_20260916.txt").write_text("\n".join(sums) + "\n")

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
