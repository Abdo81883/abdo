# MGPT-ITE1 Data Authority Matrix — 2026-09-16

## Purpose

This matrix separates **research signal truth**, **execution truth**, and **provider transport truth**. A market may be analyzable without every field being executable. ITE1 must fail closed rather than silently replace missing evidence with a proxy that changes the economic meaning.

| Market family | Development price authority | Validation/execution additions required | Key market-specific requirements |
|---|---|---|---|
| Crypto spot | Massive aggregate bars; Binance official archives for independent replication where symbol mapping is valid | exchange-specific fees/spread/slippage and venue mapping | 24x7 clock, venue differences, liquidity, leader/relative-strength context |
| Crypto perpetuals | Binance official USD-M archives | funding cashflows, contract rules, fees, OI/liquidation data only when point-in-time causal | 24x7, leverage, funding, liquidation risk |
| FX | Massive quote-derived aggregates | bid/ask spread sampling and session-aware execution model | 24x5 sessions, quote rather than trade semantics, rollover/event risk |
| Spot metals | Massive XAUUSD/XAGUSD quote aggregates | spread/session execution model; macro context only if causally timestamped | London/New York overlap, USD/rates context, volatility expansion |
| U.S. equities | Massive adjusted stock aggregates | corporate actions, earnings/event calendar, NBBO/spread or conservative slippage | regular session, gaps, sector/index context, event risk |
| Cash indices | Massive index values when entitled | map signal to an executable future/ETF/CFD before any fill claim | index value is not an executable print |
| Futures | Massive contract-specific bars + contract metadata | causal front-contract selection, roll schedule, point-in-time specifications, spread/slippage | no opaque continuous-contract shortcut; roll-aware back-adjustment |
| Energy/other commodities | Massive futures authority after futures roll protocol is frozen | same as futures plus product event/calendar context | contract-specific liquidity and seasonality |

## Hard rules

1. Instrument != alias != proxy. A proxy may contribute context but cannot silently become the executable instrument.
2. Index values are not fills.
3. Forex/spot-metal quote aggregates are not exchange trade prints.
4. Futures research does not open until the point-in-time roll protocol is frozen.
5. Provider gaps are explicit `DATA_GAP` or block rejection.
6. New provider authority, market mechanics, or trading math reopens the relevant Senior Council gate.
