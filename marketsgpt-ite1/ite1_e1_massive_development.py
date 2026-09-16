#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ite1_core import (
    CostModel,
    audit_reference_opportunity,
    compute_features,
    construct_plan,
    generate_candidates,
    simulate_trade,
)

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "MGPT_ITE1_E1_MASSIVE_DEVELOPMENT_LOCK_V2_20260916.json"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOCK = json.loads(LOCK_PATH.read_text())

DEV_START = pd.Timestamp(LOCK["windows"]["development"]["start"])
DEV_END = pd.Timestamp(LOCK["windows"]["development"]["endExclusive"])
WARMUP_START = pd.Timestamp(LOCK["windows"]["warmupStart"])
ENTRY_MODES = list(LOCK["entryChallengers"])
TIMEFRAMES = list(LOCK["timeframes"]["evaluated"])


def _safe_read(symbol: str) -> tuple[pd.DataFrame, dict]:
    fn = LOCK["dataFiles"]["mappings"][symbol]
    p = ROOT / "data" / "e1_v2" / fn
    if not p.exists():
        raise RuntimeError(f"missing staged file: {fn}")
    raw = p.read_bytes()
    txt = raw.decode("utf-8", "replace")
    if "Warning [RATE_LIMIT]" in txt or "ERROR" in txt[:200].upper():
        raise RuntimeError(f"provider warning/error payload in {fn}")
    df = pd.read_csv(p)
    needed = {"o","h","l","c","t"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"{fn}: missing columns {sorted(needed-set(df.columns))}")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["t"], unit="ms", utc=True),
        "open": pd.to_numeric(df["o"], errors="coerce"),
        "high": pd.to_numeric(df["h"], errors="coerce"),
        "low": pd.to_numeric(df["l"], errors="coerce"),
        "close": pd.to_numeric(df["c"], errors="coerce"),
        "volume": pd.to_numeric(df["v"], errors="coerce") if "v" in df.columns else np.nan,
    })
    out = out.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp")
    out = out.drop_duplicates("timestamp").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"{fn}: no valid bars")

    first = out["timestamp"].iloc[0]
    last = out["timestamp"].iloc[-1]
    coverage_ok = first <= pd.Timestamp("2024-11-02T00:00:00Z") and last >= pd.Timestamp("2025-06-29T00:00:00Z")
    diag = {
        "symbol": symbol,
        "file": fn,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": int(len(out)),
        "firstUtc": first.isoformat(),
        "lastUtc": last.isoformat(),
        "coverageOk": bool(coverage_ok),
    }
    if not coverage_ok:
        raise RuntimeError(f"{fn}: incomplete frozen coverage {first.isoformat()} -> {last.isoformat()}")
    return out, diag


def _resample_4h_exact(df: pd.DataFrame) -> pd.DataFrame:
    z = df.set_index("timestamp").copy()
    count = z["close"].resample("4h", origin="epoch", label="left", closed="left").count()
    out = z.resample("4h", origin="epoch", label="left", closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    )
    out = out[count == 4].dropna(subset=["open","high","low","close"]).reset_index()
    return out


def _cost(family: str, stress: bool) -> CostModel:
    spec = LOCK["costModelOneWayBps"][family]
    bps = float(spec["stress" if stress else "base"])
    return CostModel(one_way_fee_bps=bps, one_way_slippage_bps=0.0)


def _eval_block(symbol: str, family: str, tf: str, raw: pd.DataFrame) -> tuple[list[dict], dict]:
    df = raw if tf == "1h" else _resample_4h_exact(raw)
    if len(df) < 230:
        return [], {"symbol":symbol,"marketFamily":family,"timeframe":tf,"status":"INSUFFICIENT_BARS","bars":int(len(df))}
    f = compute_features(df)
    if "timestamp" not in f.columns:
        f.insert(0, "timestamp", df["timestamp"].values)

    candidates = generate_candidates(f, start_index=200)
    hold = int(LOCK["frozenComponents"]["maximumHoldingBarsByTimeframe"][tf])
    wait = int(LOCK["frozenComponents"]["maximumEntryWaitBars"])
    base_cost = _cost(family, False)
    stress_cost = _cost(family, True)
    rows = []
    ndev = 0

    for c in candidates:
        ts = pd.Timestamp(f.iloc[c.signal_index]["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if not (DEV_START <= ts < DEV_END):
            continue
        if c.signal_index + hold + wait + 1 >= len(f):
            continue
        ndev += 1
        cid = f"{symbol}|{tf}|{ts.isoformat()}|{c.direction}|{c.setup_class}|{c.signal_index}"
        opp = audit_reference_opportunity(
            f, c.signal_index, c.direction,
            positive_barrier_r=float(LOCK["opportunityAudit"]["positiveBarrierR"]),
            adverse_barrier_r=float(LOCK["opportunityAudit"]["adverseBarrierR"]),
            horizon_bars=hold,
        )
        for mode in ENTRY_MODES:
            p = construct_plan(
                f, c,
                entry_mode=mode,
                stop_mode=LOCK["frozenComponents"]["stop"],
                target_mode=LOCK["frozenComponents"]["target"],
                max_entry_wait_bars=wait,
                max_holding_bars=hold,
            )
            if p is None:
                rows.append({
                    "candidateId":cid,"signalUtc":ts.isoformat(),"symbol":symbol,"marketFamily":family,
                    "timeframe":tf,"direction":c.direction,"setupClass":c.setup_class,"entryMode":mode,
                    "planValid":False,"entered":False,"netR":0.0,"stressNetR":0.0,"grossR":0.0,
                    "mfeR":0.0,"maeR":0.0,"auditGoodOpportunity":bool(opp),"reason":"NO_VALID_PLAN"
                })
                continue
            rb = simulate_trade(f, p, management_mode=LOCK["frozenComponents"]["management"], cost_model=base_cost)
            rs = simulate_trade(f, p, management_mode=LOCK["frozenComponents"]["management"], cost_model=stress_cost)
            rows.append({
                "candidateId":cid,"signalUtc":ts.isoformat(),"symbol":symbol,"marketFamily":family,
                "timeframe":tf,"direction":c.direction,"setupClass":c.setup_class,"entryMode":mode,
                "planValid":True,"entered":bool(rb.entered),
                "netR":float(rb.net_r if rb.entered else 0.0),
                "stressNetR":float(rs.net_r if rs.entered else 0.0),
                "grossR":float(rb.gross_r if rb.entered else 0.0),
                "mfeR":float(rb.mfe_r if rb.entered else 0.0),
                "maeR":float(rb.mae_r if rb.entered else 0.0),
                "auditGoodOpportunity":bool(opp),"reason":rb.reason
            })
    return rows, {"symbol":symbol,"marketFamily":family,"timeframe":tf,"status":"OK","bars":int(len(df)),"developmentCandidates":ndev}


def _pf(a: np.ndarray) -> float:
    p = float(a[a>0].sum()); n = float(-a[a<0].sum())
    return p/n if n>0 else (999.0 if p>0 else 0.0)


def _metrics(df: pd.DataFrame, mode: str) -> dict:
    z = df[df["entryMode"] == mode].copy()
    ent = z[z["entered"]].copy()
    net = ent["netR"].to_numpy(float)
    stress = ent["stressNetR"].to_numpy(float)
    good = z["auditGoodOpportunity"].astype(bool)
    entered = z["entered"].astype(bool)
    captured = int((good & entered).sum())
    opp = int(good.sum())
    fam = ent.groupby("marketFamily")["netR"].mean()
    blk = ent.groupby(["symbol","timeframe"])["netR"].mean()
    return {
        "candidateRows":int(len(z)),
        "validPlans":int(z["planValid"].sum()),
        "enteredTrades":int(len(ent)),
        "netExpectancyR":float(net.mean()) if len(net) else 0.0,
        "stressNetExpectancyR":float(stress.mean()) if len(stress) else 0.0,
        "netProfitFactor":_pf(net) if len(net) else 0.0,
        "hitRate":float(np.mean(net>0)) if len(net) else 0.0,
        "meanMfeR":float(ent["mfeR"].mean()) if len(ent) else 0.0,
        "meanMaeR":float(ent["maeR"].mean()) if len(ent) else 0.0,
        "opportunityCount":opp,
        "opportunityCaptured":captured,
        "opportunityRecall":float(captured/max(1,opp)),
        "precision":float(captured/max(1,int(entered.sum()))),
        "activeMarketFamilies":int(len(fam)),
        "activeInstrumentTimeframeBlocks":int(len(blk)),
        "positiveMarketFamilyFraction":float((fam>0).mean()) if len(fam) else 0.0,
        "positiveBlockFraction":float((blk>0).mean()) if len(blk) else 0.0,
        "familyExpectancyR":{str(k):float(v) for k,v in fam.items()},
        "blockExpectancyR":{f"{k[0]}|{k[1]}":float(v) for k,v in blk.items()},
    }


def _circular_indices(n: int, rng: np.random.Generator, block_len: int) -> np.ndarray:
    out = []
    while len(out) < n:
        st = int(rng.integers(0,n))
        take = min(block_len, n-len(out))
        out.extend(((st + np.arange(take)) % n).tolist())
    return np.asarray(out, dtype=int)


def _bootstrap(df: pd.DataFrame) -> dict:
    cfg = LOCK["bootstrap"]; iters=int(cfg["iterations"]); seed=int(cfg["seed"]); bl=int(cfg["blockLengthCandidates"])
    rng = np.random.default_rng(seed)
    blocks = sorted(df[["symbol","timeframe"]].drop_duplicates().itertuples(index=False,name=None))
    by = {}
    for b in blocks:
        by[b] = df[(df["symbol"]==b[0]) & (df["timeframe"]==b[1])].sort_values("signalUtc").reset_index(drop=True)

    abs_s = {m:np.zeros(iters) for m in ENTRY_MODES}
    delta_s = {m:np.zeros(iters) for m in ENTRY_MODES if m!="trigger_close"}

    for it in range(iters):
        sampled = [blocks[int(rng.integers(0,len(blocks)))] for _ in range(len(blocks))]
        trade_r = {m:[] for m in ENTRY_MODES}
        util = {m:[] for m in ENTRY_MODES}
        for b in sampled:
            z=by[b]
            ids=z[["candidateId","signalUtc"]].drop_duplicates().sort_values("signalUtc")["candidateId"].tolist()
            if not ids: continue
            for k in _circular_indices(len(ids),rng,bl):
                cz=z[z["candidateId"]==ids[k]]
                for m in ENTRY_MODES:
                    r=cz[cz["entryMode"]==m]
                    if r.empty:
                        util[m].append(0.0); continue
                    rr=r.iloc[0]
                    u=float(rr["netR"]) if bool(rr["entered"]) else 0.0
                    util[m].append(u)
                    if bool(rr["entered"]): trade_r[m].append(float(rr["netR"]))
        for m in ENTRY_MODES:
            a=np.asarray(trade_r[m],float)
            abs_s[m][it]=float(a.mean()) if len(a) else 0.0
        base=float(np.mean(util["trigger_close"])) if util["trigger_close"] else 0.0
        for m in delta_s:
            u=float(np.mean(util[m])) if util[m] else 0.0
            delta_s[m][it]=u-base

    out={"method":cfg["method"],"iterations":iters,"seed":seed,"blockLengthCandidates":bl,"modes":{}}
    for m in ENTRY_MODES:
        a=abs_s[m]
        d={
            "probabilityPositiveNetExpectancy":float(np.mean(a>0)),
            "ci95LowNetExpectancyR":float(np.quantile(a,.025)),
            "ci95HighNetExpectancyR":float(np.quantile(a,.975)),
        }
        if m!="trigger_close":
            q=delta_s[m]
            d.update({
                "probabilityPositiveCandidateUtilityDeltaVsBaseline":float(np.mean(q>0)),
                "ci95LowCandidateUtilityDeltaVsBaseline":float(np.quantile(q,.025)),
                "ci95HighCandidateUtilityDeltaVsBaseline":float(np.quantile(q,.975)),
            })
        out["modes"][m]=d
    return out


def _checks(m: dict, b: dict, baseline: dict, mode: str) -> dict:
    g=LOCK["developmentDecisionRule"]; bb=b["modes"][mode]
    c={
        "minimumTotalEnteredTrades":m["enteredTrades"]>=int(g["minimumTotalEnteredTrades"]),
        "minimumActiveMarketFamilies":m["activeMarketFamilies"]>=int(g["minimumActiveMarketFamilies"]),
        "minimumActiveInstrumentTimeframeBlocks":m["activeInstrumentTimeframeBlocks"]>=int(g["minimumActiveInstrumentTimeframeBlocks"]),
        "minimumNetExpectancyR":m["netExpectancyR"]>=float(g["minimumNetExpectancyR"]),
        "minimumStressNetExpectancyR":m["stressNetExpectancyR"]>=float(g["minimumStressNetExpectancyR"]),
        "minimumProfitFactor":m["netProfitFactor"]>=float(g["minimumProfitFactor"]),
        "minimumOpportunityRecall":m["opportunityRecall"]>=float(g["minimumOpportunityRecall"]),
        "minimumPrecision":m["precision"]>=float(g["minimumPrecision"]),
        "minimumPositiveMarketFamilyFraction":m["positiveMarketFamilyFraction"]>=float(g["minimumPositiveMarketFamilyFraction"]),
        "minimumPositiveBlockFraction":m["positiveBlockFraction"]>=float(g["minimumPositiveBlockFraction"]),
        "bootstrapProbabilityPositiveExpectancyMin":bb["probabilityPositiveNetExpectancy"]>=float(g["bootstrapProbabilityPositiveExpectancyMin"]),
    }
    if mode!="trigger_close":
        c["challengerBootstrapProbabilityDeltaVsBaselinePositiveMin"]=bb["probabilityPositiveCandidateUtilityDeltaVsBaseline"]>=float(g["challengerBootstrapProbabilityDeltaVsBaselinePositiveMin"])
        c["maximumRelativeRecallDegradationVsBaseline"]=m["opportunityRecall"]>=baseline["opportunityRecall"]*(1.0-float(g["maximumRelativeRecallDegradationVsBaseline"]))
    return {k:bool(v) for k,v in c.items()}


def _select(mm: dict, cc: dict) -> tuple[str|None,str]:
    passing=[m for m in ENTRY_MODES if all(cc[m].values())]
    if not passing: return None,"NO_ENTRY_MODE_CLEARED_ABSOLUTE_AND_ROBUSTNESS_GATES"
    challengers=[m for m in passing if m!="trigger_close"]
    if challengers:
        simp={"trigger_close":0,"retest_limit":1,"structure_confirmed_retest":2}
        w=max(challengers,key=lambda m:(mm[m]["stressNetExpectancyR"],mm[m]["opportunityRecall"],mm[m]["precision"],-mm[m]["meanMaeR"],-simp[m]))
        return w,"CHALLENGER_CLEARED_ABSOLUTE_AND_INCREMENTAL_GATES"
    return ("trigger_close","BASELINE_CLEARED_ABSOLUTE_GATE_NO_CHALLENGER_PROVED_INCREMENTAL_IMPROVEMENT") if "trigger_close" in passing else (None,"NO_ELIGIBLE_WINNER")


def _write_freeze_failure(errors: list[dict], diagnostics: list[dict]) -> None:
    x={
        "schema":"mgpt_ite1_e1_data_freeze_closeout_v1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":"BLOCKED_DATA_FREEZE_INCOMPLETE",
        "performanceOutcomesComputed":False,
        "freshValidationOpened":False,
        "sealedHoldoutOpened":False,
        "errors":errors,
        "diagnostics":diagnostics,
        "nextAction":"Complete the predeclared Massive development extracts and checksums. Do not alter E1 strategy rules or inspect validation/holdout."
    }
    (OUT/"MGPT_ITE1_E1_DATA_FREEZE_STATUS_20260916.json").write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
    print(json.dumps(x,indent=2))


def main() -> None:
    data={}; transport=[]; errors=[]
    for item in LOCK["universe"]:
        s=item["symbol"]
        try:
            df,diag=_safe_read(s); data[s]=(item["marketFamily"],df); transport.append(diag)
        except Exception as e:
            errors.append({"symbol":s,"error":str(e)})
    if errors:
        _write_freeze_failure(errors,transport)
        raise SystemExit(3)

    rows=[]; blocks=[]
    for s,(fam,df) in data.items():
        for tf in TIMEFRAMES:
            rr,dd=_eval_block(s,fam,tf,df); rows.extend(rr); blocks.append(dd)
    if not rows:
        raise RuntimeError("no E1 evaluation rows after complete data freeze")

    ledger=pd.DataFrame(rows)
    mm={m:_metrics(ledger,m) for m in ENTRY_MODES}
    boot=_bootstrap(ledger)
    baseline=mm["trigger_close"]
    cc={m:_checks(mm[m],boot,baseline,m) for m in ENTRY_MODES}
    winner,reason=_select(mm,cc)
    status="PASS_E1_FREEZE_ENTRY_WINNER" if winner else "REJECT_E1_COMPONENT_GATE"

    close={
        "schema":"mgpt_ite1_e1_massive_development_closeout_v2",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":status,"selectedEntryMode":winner,"selectionReason":reason,
        "productionAuthority":False,"r15MutationAllowed":False,
        "freshValidationOpened":False,"sealedHoldoutOpened":False,
        "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "transport":transport,"blockDiagnostics":blocks,
        "metricsByEntryMode":mm,"bootstrap":boot,"checksByEntryMode":cc,
        "failedChecksByEntryMode":{m:[k for k,v in cc[m].items() if not v] for m in ENTRY_MODES},
        "nextAction":(
            "Freeze winner and open provider/execution replication gate before ITE1-S1."
            if winner else
            "Do not open ITE1-S1/T1/M1. Amend candidate/setup architecture under a new prelock; do not rescue E1 thresholds or open validation/holdout."
        )
    }
    close_p=OUT/"MGPT_ITE1_E1_MASSIVE_DEVELOPMENT_CLOSEOUT_20260916.json"
    ledger_p=OUT/"MGPT_ITE1_E1_MASSIVE_DEVELOPMENT_LEDGER_20260916.csv"
    sums_p=OUT/"MGPT_ITE1_E1_MASSIVE_SHA256SUMS_20260916.txt"
    close_p.write_text(json.dumps(close,indent=2,sort_keys=True)+"\n")
    ledger.to_csv(ledger_p,index=False)
    sums=[]
    for p in [close_p,ledger_p]:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    for d in transport:
        sums.append(f"{d['sha256']}  data/e1_v2/{d['file']}")
    sums_p.write_text("\n".join(sums)+"\n")
    print(json.dumps({"status":status,"selectedEntryMode":winner,"selectionReason":reason,"failedChecksByEntryMode":close["failedChecksByEntryMode"]},indent=2))


if __name__=="__main__":
    main()
