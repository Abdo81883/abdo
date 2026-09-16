# MGPT-ITE1 Senior Council Baseline Audit — R15 trade construction

## Scope

This audit treats `RC4S3H5R4R15` as frozen. It does **not** reopen the prior alpha campaign and does not authorize any Worker-byte mutation. The purpose is to identify what R15 already does well and what must change before MarketsGPT can credibly claim an institutional-quality trade-construction and opportunity-selection engine.

## Strong foundations already present in R15

R15 already contains material institutional plumbing that should be preserved rather than rewritten blindly:

- Canonical instrument/market-family resolution and provider-governor logic.
- Fail-closed handling for missing, conflicting or non-executable evidence.
- Canonical plan geometry checks for direction, stop placement and progressive targets.
- Net-RR/cost-aware decision rails and explicit market-profile policies.
- Multiple decision states rather than unconditional BUY/SELL.
- Entry-zone, stop, TP and invalidation mirroring across plan surfaces.
- Repair/advisory surfaces and execution checklists.
- Provenance, snapshot coherence and data-quality controls.
- Stable product/live/package gates already closed for the frozen Worker.

These are valuable product assets. ITE1 should use R15 as the truth/data/governance substrate.

## Critical limitations for the new objective

### 1. Current institutional ranking is heuristic, not calibrated

The R15 `institutionalRank` combines quality, execution, repairability and context with fixed weights. Missing quality may default to a middle score. This is useful for presentation, but it is not evidence that an 80 score has a higher realized success probability than a 65 score.

**ITE1 requirement:** retain heuristic ranks only as diagnostics. The trade decision must ultimately be backed by out-of-sample conditional performance and, if a probability is displayed, proper calibration.

### 2. Market-profile RR and stop thresholds are hand-authored policy rails

R15 has a substantial policy matrix by market family/mode/timeframe. That is better than one threshold for every market, but the matrix is still a ruleset rather than a demonstrated optimum.

**ITE1 requirement:** treat current rails as baseline challengers. Test whether structural/volatility-derived entry/stop/target rules improve net expectancy and recall without unacceptable drawdown.

### 3. Repair logic can improve geometry without proving economic edge

The current repair layer can suggest entry repricing, stop normalization or TP extension to recover executable RR. Geometric repair is not equivalent to increasing expected return.

**ITE1 requirement:** a proposed repair is eligible only if the repair class itself shows positive OOS incremental expectancy versus leaving the setup as WAIT.

### 4. There are symbol/timeframe-specific surgical optimizations in R15

Examples in the frozen code include an EURUSD 15m stop-floor adjustment and an NVDA 4h TP2/RR strengthening rule. R15 contains other hot-route/symbol-timeframe guards for reliability and provider constraints.

Operational guards are acceptable when they describe data/resource truth. **Outcome-driven trading geometry patches are not acceptable as a future design pattern.**

**ITE1 requirement:** no post-result symbol/timeframe tuning. Trading policies must be frozen at market-family/regime/setup level before validation, except for exogenous contract mechanics.

### 5. Target construction is not yet a validated TP ladder

R15 can enforce target progression and RR thresholds, but a target may still be shaped by threshold geometry. A professional TP ladder should distinguish:

- TP1: tactical liquidity/structure objective.
- TP2: primary expected move.
- TP3: extension only when the regime supports continuation.

**ITE1 requirement:** targets must have causal provenance from structure/volatility/liquidity and must be evaluated individually by hit probability, time-to-hit, expected R contribution and tail behavior.

### 6. Stop construction must become an explicit invalidation/noise model

A fixed percentage floor can protect against unrealistically tight stops, but the best stop is market- and regime-dependent.

**ITE1 requirement:** separate:
1. structural invalidation;
2. volatility/noise buffer;
3. execution buffer;
4. maximum-risk cap.

A stop may be rejected as too wide; it must not be moved merely to make RR look better.

### 7. Missed opportunities are not a first-class measured failure mode

The current system can be conservative and correctly return WAIT, but we do not yet have a rigorous measure of how much realizable edge is left on the table.

**ITE1 requirement:** every evaluation set must include an opportunity ledger with false-positive and false-negative analysis, including MFE/MAE and captured-forward-MFE fraction.

### 8. Trade lifecycle after entry needs a formal engine

The current report can advise on execution and cancellation, but institutional reliance requires a causal lifecycle:

`candidate -> trigger -> fill -> open risk -> TP1/partial -> stop movement rule -> TP2 -> runner/TP3 -> time stop -> failure exit`.

Every state transition must be deterministic and backtestable.

### 9. Cost and slippage need state-dependent stress

R15 already includes friction-aware rails. ITE1 should extend this to market/session/regime-dependent execution assumptions, with explicit adverse stress rather than one static cost number.

### 10. Reliability and profitability must remain separate gates

R15 proved engineering/product stability. It did not prove a profitable trade engine. ITE1 must preserve that distinction all the way to release.

## Senior Council architectural decision

Do **not** start by adding another alpha factor or by rewriting the Worker.

Start with a parallel research engine that consumes clean OHLC/execution context and builds complete trades under frozen market-family playbooks. Compare it against R15's current plan geometry and WAIT/TRADE behaviour. Only a statistically and economically superior frozen engine can become an R16 candidate.

## Initial research families

ITE1 begins with a deliberately small hypothesis budget:

1. **Entry family** — direct trigger vs retest/limit vs structure-confirmed entry.
2. **Stop family** — pure structure vs structure+ATR/noise buffer vs volatility-only baseline.
3. **Target family** — structure ladder vs volatility ladder vs hybrid structure+volatility ladder.
4. **Management family** — fixed exit vs TP1 partial + structure-managed remainder vs time-stop overlay.
5. **Selection/meta-label family** — abstain/execute decision based only on causal decision-time features.

No family may be multiplied into dozens of parameter combinations after seeing outcomes. All tested variants go into the ITE1 trial ledger.

## Definition of success

Success is not “more green trades.” A promoted engine must show:

- positive **net** expectancy after realistic costs;
- robustness under adverse cost/slippage stress;
- controlled drawdown and tail loss;
- broad contribution across multiple market families;
- materially useful opportunity recall without collapsing precision;
- stability under reasonable parameter perturbation;
- fresh OOS and one-shot holdout success;
- forward-shadow confirmation before production execution claims.

Until those conditions are met, ITE1 remains research/shadow mode and R15 remains the final production-stable baseline.
