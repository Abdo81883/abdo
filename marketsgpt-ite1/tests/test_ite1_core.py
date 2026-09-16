import math
import numpy as np
import pandas as pd

from ite1_core import (
    CostModel,
    TradePlan,
    audit_reference_opportunity,
    compute_features,
    construct_plan,
    generate_candidates,
    simulate_trade,
)


def synthetic_trend(n=320, seed=7):
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(0.08 + rng.normal(0, 0.35, n))
    op = np.r_[base[0], base[:-1]] + rng.normal(0, 0.08, n)
    cl = base
    hi = np.maximum(op, cl) + rng.uniform(0.10, 0.45, n)
    lo = np.minimum(op, cl) - rng.uniform(0.10, 0.45, n)
    return pd.DataFrame({"open":op,"high":hi,"low":lo,"close":cl,"volume":1000.0})


def test_features_prior_levels_do_not_use_current_bar():
    df = synthetic_trend()
    f = compute_features(df)
    i = 250
    assert math.isclose(f.loc[i, "prior_high20"], df.loc[i-20:i-1, "high"].max())
    assert math.isclose(f.loc[i, "prior_low20"], df.loc[i-20:i-1, "low"].min())


def test_future_mutation_does_not_change_signal_features():
    df = synthetic_trend()
    f1 = compute_features(df)
    cut = 250
    mutated = df.copy()
    mutated.loc[cut+1:, ["open","high","low","close"]] *= 10.0
    f2 = compute_features(mutated)
    cols = ["atr14","ema20","ema50","prior_high20","prior_low20","regime","vol_state"]
    for c in cols:
        a, b = f1.loc[cut, c], f2.loc[cut, c]
        if isinstance(a, str):
            assert a == b
        else:
            assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def test_plan_geometry_and_progressive_targets():
    f = compute_features(synthetic_trend())
    cands = generate_candidates(f)
    assert cands
    p = None
    for c in cands:
        p = construct_plan(
            f, c,
            entry_mode="retest_limit",
            stop_mode="structure_plus_atr_noise",
            target_mode="hybrid_structure_volatility_ladder",
        )
        if p:
            break
    assert p is not None
    if p.direction == "LONG":
        assert p.stop_loss < p.entry_reference < p.tp1 < p.tp2 < p.tp3
    else:
        assert p.stop_loss > p.entry_reference > p.tp1 > p.tp2 > p.tp3


def test_same_bar_stop_wins_over_target():
    n = 230
    px = np.linspace(90, 100, n)
    df = pd.DataFrame({"open":px,"high":px+0.2,"low":px-0.2,"close":px})
    f = compute_features(df)
    i = 210
    entry = float(f.loc[i+1,"open"])
    plan = TradePlan(
        i,"LONG","TEST","TREND_UP","NORMAL","trigger_close",
        entry,entry,entry,"volatility_baseline",entry-1.0,
        "volatility_ladder",entry+1.0,entry+2.0,entry+3.0,1.0,4,5
    )
    f.loc[i+1,"low"] = entry-1.2
    f.loc[i+1,"high"] = entry+2.2
    r = simulate_trade(f, plan, management_mode="fixed_full_exit", cost_model=CostModel(0,0))
    assert r.stopped is True
    assert r.net_r == -1.0


def test_reference_opportunity_adverse_wins_same_bar():
    df = synthetic_trend()
    f = compute_features(df)
    i = 250
    atr = float(f.loc[i,"atr14"])
    e = float(f.loc[i+1,"open"])
    f.loc[i+1,"low"] = e - 1.1*atr
    f.loc[i+1,"high"] = e + 2.0*atr
    assert audit_reference_opportunity(
        f, i, "LONG",
        positive_barrier_r=1.5,
        adverse_barrier_r=1.0,
        horizon_bars=10
    ) is False


def test_costs_reduce_r():
    df = synthetic_trend()
    f = compute_features(df)
    i = 250
    entry = float(f.loc[i+1,"open"])
    plan = TradePlan(
        i,"LONG","TEST","TREND_UP","NORMAL","trigger_close",
        entry,entry,entry,"volatility_baseline",entry-2.0,
        "volatility_ladder",entry+1.0,entry+2.0,entry+3.0,1.0,4,5
    )
    for j in range(i+1, min(i+6,len(f))):
        f.loc[j,"low"] = max(float(f.loc[j,"low"]), entry-1.0)
        f.loc[j,"high"] = entry+2.2
    a = simulate_trade(f, plan, management_mode="fixed_full_exit", cost_model=CostModel(0,0))
    b = simulate_trade(f, plan, management_mode="fixed_full_exit", cost_model=CostModel(4,2))
    assert b.net_r < a.net_r
