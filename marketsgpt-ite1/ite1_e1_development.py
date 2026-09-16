#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from ite1_core import (
    CostModel,
    audit_reference_opportunity,
    compute_features,
    construct_plan,
    generate_candidates,
    simulate_trade,
    summarize_trade_results,
)

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "MGPT_ITE1_E1_DEVELOPMENT_LOCK_20260916.json"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

LOCK = json.loads(LOCK_PATH.read_text())
DEV_START = pd.Timestamp(LOCK["windows"]["development"]["start"])
DEV_END = pd.Timestamp(LOCK["windows"]["development"]["endExclusive"])
WARMUP_START = pd.Timestamp(LOCK["windows"]["warmupStart"])

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 MarketsGPT-ITE1/1.0",
        "Accept": "application/json,text/plain,*/*",
    }
)

ENTRY_MODES = list(LOCK["entryChallengers"])
TIMEFRAMES = list(LOCK["timeframes"]["evaluated"])


def utc_ts(s: str) -> int:
    return int(pd.Timestamp(s).timestamp())


def _fetch_yahoo_hourly(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    encoded = quote(symbol, safe="")
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": "1h",
        "includePrePost": "false",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    last = None
    payload = None
    source = None
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{encoded}"
        for k in range(5):
            try:
                r = SESSION.get(url, params=params, timeout=45)
                if r.status_code == 200:
                    payload = r.json()
                    source = r.url
                    break
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last = repr(e)
            time.sleep(min(8.0, 0.7 * (2**k)))
        if payload is not None:
            break
    if payload is None:
        raise RuntimeError(f"Yahoo fetch failed for {symbol}: {last}")

    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"Yahoo error for {symbol}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"Yahoo empty result for {symbol}")

    ts = result.get("timestamp") or []
    q = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    n = min(len(ts), *(len(q.get(k) or []) for k in ("open", "high", "low", "close")))
    rows = []
    for i in range(n):
        vals = [q.get(k, [None] * n)[i] for k in ("open", "high", "low", "close")]
        if any(v is None for v in vals):
            continue
        o, h, l, c = map(float, vals)
        if not all(math.isfinite(z) and z > 0 for z in (o, h, l, c)):
            continue
        volarr = q.get("volume") or []
        v = float(volarr[i]) if i < len(volarr) and volarr[i] is not None else float("nan")
        rows.append(
            {
                "timestamp": pd.to_datetime(int(ts[i]), unit="s", utc=True),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )
    if not rows:
        raise RuntimeError(f"Yahoo returned no valid bars for {symbol}")
    df = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)

    canonical = df.to_csv(index=False, float_format="%.12g").encode()
    diag = {
        "symbol": symbol,
        "provider": "YahooFinanceChart",
        "sourceUrl": source,
        "bars1h": int(len(df)),
        "firstUtc": df["timestamp"].iloc[0].isoformat() if len(df) else None,
        "lastUtc": df["timestamp"].iloc[-1].isoformat() if len(df) else None,
        "sha256CanonicalCsv": hashlib.sha256(canonical).hexdigest(),
        "duplicateTimestampsAfterNormalization": 0,
    }
    return df, diag


def _resample_4h_exact(df: pd.DataFrame) -> pd.DataFrame:
    """UTC-anchored 4h groups; only groups with exactly 4 source bars survive."""
    z = df.copy().set_index("timestamp")
    count = z["close"].resample("4h", origin="epoch", label="left", closed="left").count()
    out = z.resample("4h", origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    out = out[count == 4].dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def _market_cost(family: str, stress: bool = False) -> CostModel:
    spec = LOCK["costModelOneWayBps"][family]
    bps = float(spec["stress" if stress else "base"])
    return CostModel(one_way_fee_bps=bps, one_way_slippage_bps=0.0)


def _candidate_id(symbol: str, tf: str, signal_time: pd.Timestamp, c) -> str:
    return f"{symbol}|{tf}|{signal_time.isoformat()}|{c.direction}|{c.setup_class}|{c.signal_index}"


def _max_hold(tf: str) -> int:
    return int(LOCK["frozenComponents"]["maximumHoldingBarsByTimeframe"][tf])


def _evaluate_block(symbol: str, family: str, tf: str, raw: pd.DataFrame) -> tuple[list[dict], dict]:
    df = raw if tf == "1h" else _resample_4h_exact(raw)
    if len(df) < 230:
        return [], {"symbol": symbol, "marketFamily": family, "timeframe": tf, "status": "INSUFFICIENT_BARS", "bars": int(len(df))}

    f = compute_features(df)
    f.insert(0, "timestamp", df["timestamp"].values)
    candidates = generate_candidates(f, start_index=200)
    rows = []
    max_hold = _max_hold(tf)
    audit_horizon = int(LOCK["opportunityAudit"].get("maximumHorizonBars", max_hold)) if "maximumHorizonBars" in LOCK["opportunityAudit"] else max_hold
    base_cost = _market_cost(family, False)
    stress_cost = _market_cost(family, True)

    dev_candidates = 0
    for c in candidates:
        sig_time = pd.Timestamp(f.iloc[c.signal_index]["timestamp"]).tz_localize("UTC") if pd.Timestamp(f.iloc[c.signal_index]["timestamp"]).tzinfo is None else pd.Timestamp(f.iloc[c.signal_index]["timestamp"]).tz_convert("UTC")
        if not (DEV_START <= sig_time < DEV_END):
            continue
        # Preserve a full causal evaluation horizon inside development only.
        needed = max(max_hold, audit_horizon) + int(LOCK["frozenComponents"]["maximumEntryWaitBars"]) + 1
        if c.signal_index + needed >= len(f):
            continue

        dev_candidates += 1
        cid = _candidate_id(symbol, tf, sig_time, c)
        opp = audit_reference_opportunity(
            f,
            c.signal_index,
            c.direction,
            positive_barrier_r=float(LOCK["opportunityAudit"]["positiveBarrierR"]),
            adverse_barrier_r=float(LOCK["opportunityAudit"]["adverseBarrierR"]),
            horizon_bars=audit_horizon,
        )
        for mode in ENTRY_MODES:
            p = construct_plan(
                f,
                c,
                entry_mode=mode,
                stop_mode=LOCK["frozenComponents"]["stop"],
                target_mode=LOCK["frozenComponents"]["target"],
                max_entry_wait_bars=int(LOCK["frozenComponents"]["maximumEntryWaitBars"]),
                max_holding_bars=max_hold,
            )
            if p is None:
                rows.append(
                    {
                        "candidateId": cid,
                        "signalUtc": sig_time.isoformat(),
                        "symbol": symbol,
                        "marketFamily": family,
                        "timeframe": tf,
                        "direction": c.direction,
                        "setupClass": c.setup_class,
                        "entryMode": mode,
                        "planValid": False,
                        "entered": False,
                        "netR": 0.0,
                        "stressNetR": 0.0,
                        "grossR": 0.0,
                        "mfeR": 0.0,
                        "maeR": 0.0,
                        "auditGoodOpportunity": bool(opp),
                        "reason": "NO_VALID_PLAN",
                    }
                )
                continue
            rb = simulate_trade(f, p, management_mode=LOCK["frozenComponents"]["management"], cost_model=base_cost)
            rs = simulate_trade(f, p, management_mode=LOCK["frozenComponents"]["management"], cost_model=stress_cost)
            rows.append(
                {
                    "candidateId": cid,
                    "signalUtc": sig_time.isoformat(),
                    "symbol": symbol,
                    "marketFamily": family,
                    "timeframe": tf,
                    "direction": c.direction,
                    "setupClass": c.setup_class,
                    "entryMode": mode,
                    "planValid": True,
                    "entered": bool(rb.entered),
                    "netR": float(rb.net_r if rb.entered else 0.0),
                    "stressNetR": float(rs.net_r if rs.entered else 0.0),
                    "grossR": float(rb.gross_r if rb.entered else 0.0),
                    "mfeR": float(rb.mfe_r if rb.entered else 0.0),
                    "maeR": float(rb.mae_r if rb.entered else 0.0),
                    "auditGoodOpportunity": bool(opp),
                    "reason": rb.reason,
                }
            )

    diag = {
        "symbol": symbol,
        "marketFamily": family,
        "timeframe": tf,
        "status": "OK",
        "bars": int(len(df)),
        "developmentCandidates": int(dev_candidates),
    }
    return rows, diag


def _pf(a: np.ndarray) -> float:
    p = float(a[a > 0].sum())
    n = float(-a[a < 0].sum())
    return p / n if n > 0 else (999.0 if p > 0 else 0.0)


def _mode_metrics(df: pd.DataFrame, mode: str) -> dict:
    z = df[df["entryMode"] == mode].copy()
    ent = z[z["entered"]]
    net = ent["netR"].to_numpy(float)
    stress = ent["stressNetR"].to_numpy(float)
    good = z["auditGoodOpportunity"].astype(bool)
    entered = z["entered"].astype(bool)
    good_n = int(good.sum())
    captured = int((good & entered).sum())
    precision = captured / max(1, int(entered.sum()))
    recall = captured / max(1, good_n)

    fam = ent.groupby("marketFamily")["netR"].mean()
    blocks = ent.groupby(["symbol", "timeframe"])["netR"].mean()
    active_families = int(len(fam))
    active_blocks = int(len(blocks))
    pos_fam_frac = float((fam > 0).mean()) if len(fam) else 0.0
    pos_block_frac = float((blocks > 0).mean()) if len(blocks) else 0.0

    return {
        "candidateRows": int(len(z)),
        "validPlans": int(z["planValid"].sum()),
        "enteredTrades": int(len(ent)),
        "netExpectancyR": float(net.mean()) if len(net) else 0.0,
        "stressNetExpectancyR": float(stress.mean()) if len(stress) else 0.0,
        "netProfitFactor": _pf(net) if len(net) else 0.0,
        "hitRate": float((net > 0).mean()) if len(net) else 0.0,
        "meanMfeR": float(ent["mfeR"].mean()) if len(ent) else 0.0,
        "meanMaeR": float(ent["maeR"].mean()) if len(ent) else 0.0,
        "opportunityCount": good_n,
        "opportunityCaptured": captured,
        "opportunityRecall": float(recall),
        "precision": float(precision),
        "activeMarketFamilies": active_families,
        "activeInstrumentTimeframeBlocks": active_blocks,
        "positiveMarketFamilyFraction": pos_fam_frac,
        "positiveBlockFraction": pos_block_frac,
        "familyExpectancyR": {str(k): float(v) for k, v in fam.items()},
        "blockExpectancyR": {f"{k[0]}|{k[1]}": float(v) for k, v in blocks.items()},
    }


def _circular_indices(n: int, target: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    if n <= 0:
        return np.array([], dtype=int)
    while len(out) < target:
        st = int(rng.integers(0, n))
        take = min(block_len, target - len(out))
        out.extend(((st + np.arange(take)) % n).tolist())
    return np.asarray(out, dtype=int)


def _bootstrap(df: pd.DataFrame, iterations: int, seed: int, block_len: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    blocks = sorted(df[["symbol", "timeframe"]].drop_duplicates().itertuples(index=False, name=None))
    by = {}
    for b in blocks:
        z = df[(df["symbol"] == b[0]) & (df["timeframe"] == b[1])].copy().sort_values("signalUtc")
        by[b] = z.reset_index(drop=True)

    abs_samples = {m: np.zeros(iterations, float) for m in ENTRY_MODES}
    delta_samples = {m: np.zeros(iterations, float) for m in ENTRY_MODES if m != "trigger_close"}

    for it in range(iterations):
        sampled_blocks = [blocks[int(rng.integers(0, len(blocks)))] for _ in range(len(blocks))]
        mode_trade_r = {m: [] for m in ENTRY_MODES}
        mode_utility = {m: [] for m in ENTRY_MODES}

        for b in sampled_blocks:
            z = by[b]
            ids = z["candidateId"].drop_duplicates().tolist()
            if not ids:
                continue
            # Preserve candidate temporal clusters. Each candidate contributes one row per entry mode.
            cand_order = (
                z[["candidateId", "signalUtc"]]
                .drop_duplicates()
                .sort_values("signalUtc")["candidateId"]
                .tolist()
            )
            idx = _circular_indices(len(cand_order), len(cand_order), block_len, rng)
            sampled_ids = [cand_order[k] for k in idx]
            for cid in sampled_ids:
                cz = z[z["candidateId"] == cid]
                for m in ENTRY_MODES:
                    row = cz[cz["entryMode"] == m]
                    if row.empty:
                        mode_utility[m].append(0.0)
                        continue
                    rr = row.iloc[0]
                    util = float(rr["netR"]) if bool(rr["entered"]) else 0.0
                    mode_utility[m].append(util)
                    if bool(rr["entered"]):
                        mode_trade_r[m].append(float(rr["netR"]))

        for m in ENTRY_MODES:
            vals = np.asarray(mode_trade_r[m], float)
            abs_samples[m][it] = float(vals.mean()) if len(vals) else 0.0
        base_util = float(np.mean(mode_utility["trigger_close"])) if mode_utility["trigger_close"] else 0.0
        for m in delta_samples:
            u = float(np.mean(mode_utility[m])) if mode_utility[m] else 0.0
            delta_samples[m][it] = u - base_util

    out = {"method": "hierarchical instrument-timeframe bootstrap with circular candidate blocks", "blockLengthCandidates": block_len, "iterations": iterations, "seed": seed, "modes": {}}
    for m in ENTRY_MODES:
        arr = abs_samples[m]
        item = {
            "probabilityPositiveNetExpectancy": float(np.mean(arr > 0)),
            "ci95LowNetExpectancyR": float(np.quantile(arr, 0.025)),
            "ci95HighNetExpectancyR": float(np.quantile(arr, 0.975)),
        }
        if m != "trigger_close":
            d = delta_samples[m]
            item.update(
                {
                    "probabilityPositiveCandidateUtilityDeltaVsBaseline": float(np.mean(d > 0)),
                    "ci95LowCandidateUtilityDeltaVsBaseline": float(np.quantile(d, 0.025)),
                    "ci95HighCandidateUtilityDeltaVsBaseline": float(np.quantile(d, 0.975)),
                }
            )
        out["modes"][m] = item
    return out


def _checks(metrics: dict, bootstrap: dict, baseline_metrics: dict, mode: str) -> dict:
    g = LOCK["developmentDecisionRule"]
    b = bootstrap["modes"][mode]
    c = {
        "minimumTotalEnteredTrades": metrics["enteredTrades"] >= int(g["minimumTotalEnteredTrades"]),
        "minimumActiveMarketFamilies": metrics["activeMarketFamilies"] >= int(g["minimumActiveMarketFamilies"]),
        "minimumActiveInstrumentTimeframeBlocks": metrics["activeInstrumentTimeframeBlocks"] >= int(g["minimumActiveInstrumentTimeframeBlocks"]),
        "minimumNetExpectancyR": metrics["netExpectancyR"] >= float(g["minimumNetExpectancyR"]),
        "minimumStressNetExpectancyR": metrics["stressNetExpectancyR"] >= float(g["minimumStressNetExpectancyR"]),
        "minimumProfitFactor": metrics["netProfitFactor"] >= float(g["minimumProfitFactor"]),
        "minimumOpportunityRecall": metrics["opportunityRecall"] >= float(g["minimumOpportunityRecall"]),
        "minimumPrecision": metrics["precision"] >= float(g["minimumPrecision"]),
        "minimumPositiveMarketFamilyFraction": metrics["positiveMarketFamilyFraction"] >= float(g["minimumPositiveMarketFamilyFraction"]),
        "minimumPositiveBlockFraction": metrics["positiveBlockFraction"] >= float(g["minimumPositiveBlockFraction"]),
        "bootstrapProbabilityPositiveExpectancyMin": b["probabilityPositiveNetExpectancy"] >= float(g["bootstrapProbabilityPositiveExpectancyMin"]),
    }
    if mode != "trigger_close":
        c["challengerBootstrapProbabilityDeltaVsBaselinePositiveMin"] = b["probabilityPositiveCandidateUtilityDeltaVsBaseline"] >= float(g["challengerBootstrapProbabilityDeltaVsBaselinePositiveMin"])
        base_recall = float(baseline_metrics["opportunityRecall"])
        min_allowed = base_recall * (1.0 - float(g["maximumRelativeRecallDegradationVsBaseline"]))
        c["maximumRelativeRecallDegradationVsBaseline"] = metrics["opportunityRecall"] >= min_allowed
    return {k: bool(v) for k, v in c.items()}


def _select(metrics_all: dict, checks_all: dict) -> tuple[str | None, str]:
    passing = [m for m in ENTRY_MODES if all(checks_all[m].values())]
    if not passing:
        return None, "NO_ENTRY_MODE_CLEARED_ABSOLUTE_AND_ROBUSTNESS_GATES"
    challengers = [m for m in passing if m != "trigger_close"]
    if challengers:
        # Lock-defined tiebreak order: stress expectancy, recall, precision, lower MAE, then simplicity.
        simplicity = {"trigger_close": 0, "retest_limit": 1, "structure_confirmed_retest": 2}
        winner = max(
            challengers,
            key=lambda m: (
                metrics_all[m]["stressNetExpectancyR"],
                metrics_all[m]["opportunityRecall"],
                metrics_all[m]["precision"],
                -metrics_all[m]["meanMaeR"],
                -simplicity[m],
            ),
        )
        return winner, "CHALLENGER_CLEARED_ABSOLUTE_AND_INCREMENTAL_GATES"
    if "trigger_close" in passing:
        return "trigger_close", "BASELINE_CLEARED_ABSOLUTE_GATE_NO_CHALLENGER_PROVED_INCREMENTAL_IMPROVEMENT"
    return None, "NO_ELIGIBLE_WINNER"


def main() -> None:
    transport = []
    block_diag = []
    all_rows = []
    raw_by_symbol = {}

    # DEVELOPMENT ONLY: period2 is the predeclared development end. Validation/holdout are never requested.
    for item in LOCK["universe"]:
        symbol = item["symbol"]
        family = item["marketFamily"]
        try:
            raw, diag = _fetch_yahoo_hourly(symbol, WARMUP_START, DEV_END)
            raw_by_symbol[symbol] = (family, raw)
            transport.append({**diag, "marketFamily": family, "status": "OK"})
        except Exception as e:
            transport.append({"symbol": symbol, "marketFamily": family, "status": "FAILED", "error": str(e)})
            continue

    for symbol, (family, raw) in raw_by_symbol.items():
        for tf in TIMEFRAMES:
            rows, diag = _evaluate_block(symbol, family, tf, raw)
            all_rows.extend(rows)
            block_diag.append(diag)

    if not all_rows:
        raise RuntimeError("E1 produced no evaluation rows")

    df = pd.DataFrame(all_rows)
    metrics_all = {m: _mode_metrics(df, m) for m in ENTRY_MODES}
    boot = _bootstrap(
        df,
        iterations=int(LOCK["bootstrap"]["iterations"]),
        seed=int(LOCK["bootstrap"]["seed"]),
        block_len=20,
    )
    baseline = metrics_all["trigger_close"]
    checks_all = {m: _checks(metrics_all[m], boot, baseline, m) for m in ENTRY_MODES}
    winner, reason = _select(metrics_all, checks_all)

    result_status = "PASS_E1_FREEZE_ENTRY_WINNER" if winner is not None else "REJECT_E1_COMPONENT_GATE"
    closeout = {
        "schema": "mgpt_ite1_e1_development_closeout_v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "program": "MGPT-ITE1",
        "trial": "ITE1-E1",
        "status": result_status,
        "selectedEntryMode": winner,
        "selectionReason": reason,
        "productionAuthority": False,
        "r15MutationAllowed": False,
        "freshValidationOpened": False,
        "sealedHoldoutOpened": False,
        "lockSha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "dataBoundary": {
            "requestedWarmupStart": WARMUP_START.isoformat(),
            "requestedDevelopmentEndExclusive": DEV_END.isoformat(),
            "freshValidationDataRequested": False,
            "holdoutDataRequested": False,
        },
        "transport": transport,
        "blockDiagnostics": block_diag,
        "metricsByEntryMode": metrics_all,
        "bootstrap": boot,
        "checksByEntryMode": checks_all,
        "failedChecksByEntryMode": {m: [k for k, v in checks_all[m].items() if not v] for m in ENTRY_MODES},
        "evaluatorDisclosure": {
            "portfolioConcurrencyModeled": False,
            "componentPurpose": "Entry-mechanic screen only; overlapping candidate trades are allowed for component attribution.",
            "opportunityLabelsUsedAsFeatures": False,
            "bootstrapDeviation": "Uses hierarchical instrument-timeframe resampling with circular blocks of candidate events; this is the predeclared conservative approximation because candidate clocks differ across instruments.",
            "futuresAuthority": "Continuous futures proxy results are non-promotable until contract-aware provider replication.",
        },
        "nextAction": (
            "Freeze the selected entry mode, perform provider-replication gate, then authorize ITE1-S1 only if replication preserves the E1 decision."
            if winner is not None
            else
            "Do not open ITE1-S1/T1/M1. Reassess the frozen candidate/setup architecture under a new predeclared ITE1 amendment; do not rescue E1 thresholds or inspect validation/holdout."
        ),
    }

    close_path = OUT / "MGPT_ITE1_E1_DEVELOPMENT_CLOSEOUT_20260916.json"
    ledger_path = OUT / "MGPT_ITE1_E1_DEVELOPMENT_LEDGER_20260916.csv"
    trans_path = OUT / "MGPT_ITE1_E1_TRANSPORT_DIAGNOSTICS_20260916.json"
    sums_path = OUT / "MGPT_ITE1_E1_SHA256SUMS_20260916.txt"

    close_path.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    df.to_csv(ledger_path, index=False)
    trans_path.write_text(json.dumps({"schema":"mgpt_ite1_e1_transport_v1","rows":transport,"blocks":block_diag}, indent=2, sort_keys=True) + "\n")
    sums = []
    for p in [close_path, ledger_path, trans_path]:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    sums_path.write_text("\n".join(sums) + "\n")

    print(json.dumps({
        "status": result_status,
        "selectedEntryMode": winner,
        "selectionReason": reason,
        "failedChecksByEntryMode": closeout["failedChecksByEntryMode"],
        "metricsByEntryMode": {
            m: {k: v for k, v in metrics_all[m].items() if k not in {"familyExpectancyR","blockExpectancyR"}}
            for m in ENTRY_MODES
        },
        "transportFailures": [x for x in transport if x["status"] != "OK"],
    }, indent=2))


if __name__ == "__main__":
    main()
