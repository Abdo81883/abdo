# MarketsGPT Final Senior Council Closure V2 — 2026-09-16

## Final verdict

**CLOSED_PRODUCT_STABLE_ALPHA_NOT_PROVEN_AFTER_EXTENDED_PROGRAM**

The operational Worker remains **RC4S3H5R4R15 byte-for-byte unchanged**.

- Version: `31.6.982-r1-65d7a5h1-final-closure-rc4s3h5r4r15`
- Worker SHA-256: `8552bb8c3889f059d08234077c3ef1b6a10c74a01213fae576cd0a0b9a62b6c3`
- Product/live gate: PASS.
- Automatic execution: disabled.
- Broker order submission: disabled.
- Profitability claim: not authorized.

## Extended research completed after the original closure

The project was deliberately reopened only as a **separate research extension** while R15 remained frozen. We tested progressively richer decision architectures rather than silently tuning failed results:

1. **ITE1** — market-specific trade construction and price-pattern architecture: E1/A1/B1/C1 all failed their frozen development gates.
2. **IFE1** — causal selective meta-decision engine: rejected after selecting zero trades on the development screen.
3. **IFE2** — distributional opportunity/tail-risk engine: rejected; tail-loss discrimination existed, but no executable positive edge was established.
4. **MIE1** — new Binance perpetual microstructure/funding information: rejected with negative expectancy and PF below 1.
5. **MIE2** — new Binance open-interest/positioning pairwise direction engine: this was the terminal family.

## The strongest result: MIE2

MIE2 did uncover **non-trivial directional information**:

- Pair-direction accuracy: **56.90%**
- LONG-better logistic AUC: **0.5795**
- Selected improvement versus pair mean: **+0.1195R**
- Bootstrap P(relative delta > 0): **1.000**

However, the production-relevant absolute results did not satisfy the predeclared gate:

- Base net expectancy: **+0.00369R** per selected trade
- Stress net expectancy: **-0.06535R**
- Profit factor: **1.0060**
- Bootstrap P(net expectancy > 0): **0.5532**, versus locked minimum **0.95**
- Positive symbols: **50%**
- Positive months: **25%**
- Second-half expectancy: **-0.02384R**

This is useful research evidence, but it is not robust profitability evidence.

## Worker integration decision

**No alpha module is integrated.**

The release contract required development PASS, independent/provider parity where relevant, fresh validation, one-shot sealed holdout, then full R15 regression and live-gate integration. MIE2 failed at the first stage. Its May–August 2026 evidence was never opened.

Changing Worker bytes now merely to satisfy an integration request would convert a rejected research artifact into production logic and invalidate the governance protocol. The correct production decision is therefore a **no-change release**.

## Terminal research stop

The finite research budget is exhausted. No additional derived-price/positioning family is authorized inside this closure campaign.

A future research effort may be opened only as a **new project** with materially different information—such as historical order-book depth, liquidation events or event-level execution data—with new prelocks and untouched evidence.

## Final delivery

The final deliverable is the existing byte-identical R15 Worker plus this closure extension. MarketsGPT remains a stable, disciplined market-analysis and decision-support product. It has **not** been scientifically shown to select consistently profitable trades after realistic costs and therefore must not be represented as a validated-profit or autonomous execution engine.
