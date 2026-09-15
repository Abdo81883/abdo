#!/usr/bin/env python3
from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "ARP23_OID1_LOCK_20260915.json"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOCK = json.loads(LOCK_PATH.read_text())
SYMS = LOCK["cohort"]["symbols"]
BASE = "https://data.binance.vision/data/futures/um"
DEV_START = datetime.fromisoformat(LOCK["windows"]["development"]["start"].replace("Z", "+00:00"))
DEV_END = datetime.fromisoformat(LOCK["windows"]["development"]["endExclusive"].replace("Z", "+00:00"))
WARMUP_START = datetime.fromisoformat(LOCK["windows"]["warmupStart"].replace("Z", "+00:00"))
HOUR_MS = 3600_000
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"MarketsGPT-ARP23-OID1/1.0"})


def dt_ms(x: datetime) -> int:
    return int(x.timestamp() * 1000)


def date_range(a: date, b_exclusive: date):
    d = a
    while d < b_exclusive:
        yield d
        d += timedelta(days=1)


def month_range(a: datetime, b: datetime):
    y, m = a.year, a.month
    out = []
    while (y, m) <= (b.year, b.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def get_bytes(url: str, tries: int = 5):
    last = None
    for k in range(tries):
        try:
            r = SESSION.get(url, timeout=45)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(8, 0.75 * (2 ** k)))
    raise RuntimeError(f"download failed {url}: {last}")


def verified_zip(url: str):
    blob = get_bytes(url)
    if blob is None:
        return None, None
    chk = get_bytes(url + ".CHECKSUM")
    if chk is None:
        raise RuntimeError(f"missing checksum: {url}")
    expected = chk.decode("utf-8", "replace").strip().split()[0].lower()
    actual = hashlib.sha256(blob).hexdigest()
    if expected != actual:
        raise RuntimeError(f"checksum mismatch: {url}: {expected} != {actual}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise RuntimeError(f"unexpected zip members: {url}: {names}")
        raw = z.read(names[0]).decode("utf-8-sig", "replace")
    return raw, actual


def parse_time(s: str) -> int:
    s = str(s).strip()
    if not s:
        raise ValueError("empty timestamp")
    try:
        n = int(float(s))
        if n > 10**14:
            n //= 1000
        if n > 10**11:
            return n
    except Exception:
        pass
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return dt_ms(d.astimezone(timezone.utc))


def download_metric(sym: str, d: date):
    stamp = d.isoformat()
    name = f"{sym}-metrics-{stamp}.zip"
    url = f"{BASE}/daily/metrics/{sym}/{name}"
    raw, sha = verified_zip(url)
    if raw is None:
        return {"kind":"metrics","symbol":sym,"period":stamp,"status":"404","sha256":None,"rows":[]}
    rows = []
    for rec in csv.reader(io.StringIO(raw)):
        if not rec:
            continue
        if str(rec[0]).strip().lower() == "create_time":
            continue
        if len(rec) < 3:
            continue
        try:
            t = parse_time(rec[0])
            oi = float(rec[2])
        except Exception:
            continue
        if oi > 0 and math.isfinite(oi):
            rows.append((t, oi))
    return {"kind":"metrics","symbol":sym,"period":stamp,"status":"ok","sha256":sha,"rows":rows}


def download_kline_month(sym: str, ym: str):
    name = f"{sym}-1h-{ym}.zip"
    url = f"{BASE}/monthly/klines/{sym}/1h/{name}"
    raw, sha = verified_zip(url)
    if raw is None:
        return {"kind":"klines","symbol":sym,"period":ym,"status":"404","sha256":None,"rows":[]}
    rows = []
    for rec in csv.reader(io.StringIO(raw)):
        if not rec:
            continue
        try:
            t = int(float(rec[0]))
        except Exception:
            continue
        if t > 10**14:
            t //= 1000
        if len(rec) < 2:
            continue
        try:
            op = float(rec[1])
        except Exception:
            continue
        if op > 0 and math.isfinite(op):
            rows.append((t, op))
    return {"kind":"klines","symbol":sym,"period":ym,"status":"ok","sha256":sha,"rows":rows}


def transport():
    metric_start = (WARMUP_START - timedelta(days=1)).date()
    metric_end = (DEV_END + timedelta(days=1)).date()
    months = month_range(WARMUP_START - timedelta(days=1), DEV_END + timedelta(days=1))
    jobs = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for sym in SYMS:
            for d in date_range(metric_start, metric_end):
                jobs.append(ex.submit(download_metric, sym, d))
            for ym in months:
                jobs.append(ex.submit(download_kline_month, sym, ym))
        got = [f.result() for f in as_completed(jobs)]

    metrics = {s:{} for s in SYMS}
    prices = {s:{} for s in SYMS}
    dup_m = {s:0 for s in SYMS}
    dup_p = {s:0 for s in SYMS}
    manifest = []
    for x in got:
        manifest.append({k:v for k,v in x.items() if k != "rows"})
        if x["kind"] == "metrics":
            for t, oi in x["rows"]:
                if t in metrics[x["symbol"]]:
                    dup_m[x["symbol"]] += 1
                metrics[x["symbol"]][t] = oi
        else:
            for t, op in x["rows"]:
                if t in prices[x["symbol"]]:
                    dup_p[x["symbol"]] += 1
                prices[x["symbol"]][t] = op

    metric_series = {}
    diagnostics = []
    for sym in SYMS:
        mts = sorted(metrics[sym])
        metric_series[sym] = (mts, [metrics[sym][t] for t in mts])
        pts = sorted(prices[sym])
        missing_metric_days = sum(1 for x in manifest if x["kind"]=="metrics" and x["symbol"]==sym and x["status"]=="404")
        missing_kline_months = sum(1 for x in manifest if x["kind"]=="klines" and x["symbol"]==sym and x["status"]=="404")
        diagnostics.append({
            "symbol":sym,
            "metricSnapshots":len(mts),"hourlyPriceBars":len(pts),
            "firstMetricUtc":datetime.fromtimestamp(mts[0]/1000,timezone.utc).isoformat() if mts else None,
            "lastMetricUtc":datetime.fromtimestamp(mts[-1]/1000,timezone.utc).isoformat() if mts else None,
            "firstPriceUtc":datetime.fromtimestamp(pts[0]/1000,timezone.utc).isoformat() if pts else None,
            "lastPriceUtc":datetime.fromtimestamp(pts[-1]/1000,timezone.utc).isoformat() if pts else None,
            "duplicateMetricTimestamps":dup_m[sym],"duplicateHourlyPriceTimestamps":dup_p[sym],
            "missingMetricDailyArchives":missing_metric_days,"missingKlineMonthlyArchives":missing_kline_months
        })
    return metric_series, prices, manifest, diagnostics


def sampled_oi(metric_series, sym: str, boundary_ms: int):
    ts, vals = metric_series[sym]
    i = bisect.bisect_left(ts, boundary_ms) - 1
    if i < 0:
        return None
    age = boundary_ms - ts[i]
    if age <= 0 or age > 10 * 60 * 1000:
        return None
    return vals[i]


def average_ranks(values):
    """Ascending average ranks, equivalent to pandas rank(method='average')."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    k = 0
    while k < n:
        j = k + 1
        while j < n and values[order[j]] == values[order[k]]:
            j += 1
        avg = ((k + 1) + j) / 2.0
        for z in range(k, j):
            ranks[order[z]] = avg
        k = j
    return np.asarray(ranks, float)


def weights_from_scores(items):
    # items: [(symbol_index, score), ...]
    scores = [x[1] for x in items]
    r = average_ranks(scores)
    denom = float(np.abs(r).sum())
    if denom <= 0:
        return np.zeros(len(SYMS), float)
    w_local = r / denom
    w_local = w_local - float(np.mean(w_local))
    w = np.zeros(len(SYMS), float)
    for (si, _), wi in zip(items, w_local):
        w[si] = wi
    return w


def pf(a):
    pos = float(np.sum(a[a > 0]))
    neg = float(-np.sum(a[a < 0]))
    return pos / neg if neg > 0 else (999.0 if pos > 0 else 0.0)


def max_dd(a):
    e, pk, dd = 1.0, 1.0, 0.0
    for r in a:
        e = max(1e-12, e * (1.0 + float(r)))
        pk = max(pk, e)
        dd = max(dd, 1.0 - e/pk)
    return dd


def bootstrap(a, seed, iterations=10000, block=24):
    a = np.asarray(a, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    out = np.empty(iterations)
    for k in range(iterations):
        vals = []
        while len(vals) < n:
            st = int(rng.integers(0, n))
            take = min(block, n - len(vals))
            vals.extend(a[(st + np.arange(take)) % n])
        out[k] = float(np.mean(vals))
    out.sort()
    return {
        "probPositive":float(np.mean(out > 0)),
        "ci95Low":float(out[int(.025*(iterations-1))]),
        "ci95High":float(out[int(.975*(iterations-1))])
    }


def stats(a, times, seed):
    a = np.asarray(a, float)
    m = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    weeks = {}
    for r, t in zip(a, times):
        d = datetime.fromtimestamp(t/1000, timezone.utc)
        iso = d.isocalendar()
        key = f"{iso.year}-W{iso.week:02d}"
        weeks[key] = weeks.get(key, 0.0) + float(r)
    return {
        "n":len(a),"meanHourlyReturn":m,"annualizedMean":m*8760,
        "annualizedSharpe":float(m/sd*math.sqrt(8760)) if sd>0 else 0.0,
        "profitFactor":pf(a),"maxDrawdown":max_dd(a),
        "positiveWeekFraction":sum(v>0 for v in weeks.values())/max(1,len(weeks)),
        "weekSums":weeks,"bootstrap":bootstrap(a,seed)
    }


def main():
    metric_series, prices, manifest, diagnostics = transport()
    start = dt_ms(DEV_START)
    end = dt_ms(DEV_END)
    expected = int((end-start)//HOUR_MS)
    prev = np.zeros(len(SYMS), float)
    gross, turnover, times = [], [], []
    eligible_counts = []
    active_hours = 0
    contrib = np.zeros(len(SYMS), float)

    for h in range(expected):
        t = start + h*HOUR_MS
        prev_t = t - HOUR_MS
        next_t = t + HOUR_MS
        items = []
        future_ret_by_si = {}
        for si, sym in enumerate(SYMS):
            o_prev = prices[sym].get(prev_t)
            o_t = prices[sym].get(t)
            o_next = prices[sym].get(next_t)
            if o_prev is None or o_t is None or o_next is None:
                continue
            oi_prev = sampled_oi(metric_series, sym, prev_t)
            oi_t = sampled_oi(metric_series, sym, t)
            if oi_prev is None or oi_t is None or oi_prev <= 0:
                continue
            completed_ret = o_t/o_prev - 1.0
            oi_delta = oi_t/oi_prev - 1.0
            score = completed_ret - oi_delta
            fut = o_next/o_t - 1.0
            if not (math.isfinite(score) and math.isfinite(fut)):
                continue
            items.append((si, score))
            future_ret_by_si[si] = fut

        eligible_counts.append(len(items))
        desired = np.zeros(len(SYMS), float)
        if len(items) >= LOCK["cohort"]["minimumEligiblePerDecision"]:
            active_hours += 1
            desired = weights_from_scores(items)

        r = 0.0
        for si, fut in future_ret_by_si.items():
            c = desired[si] * fut
            r += c
            contrib[si] += c
        to = float(np.abs(desired-prev).sum())
        gross.append(r); turnover.append(to); times.append(t)
        prev = desired

    if turnover:
        turnover[-1] += float(np.abs(prev).sum())
    gross=np.asarray(gross,float); turnover=np.asarray(turnover,float)
    c=LOCK["costs"]
    net=gross-turnover*c["baseOneWayRate"]
    n15=gross-turnover*c["stress1_5xOneWayRate"]
    n2=gross-turnover*c["stress2xOneWayRate"]

    gs=stats(gross,times,23100); ns=stats(net,times,23101); s15=stats(n15,times,23111); s2=stats(n2,times,23112)
    coverage=active_hours/expected if expected else 0.0
    avg_eligible=float(np.mean(eligible_counts)) if eligible_counts else 0.0
    pos=contrib[contrib>0]
    dominance=float(pos.max()/pos.sum()) if len(pos) and pos.sum()>0 else 0.0
    g=LOCK["developmentGate"]
    gate={
        "minimumEvaluatedHours":len(net)>=g["minimumEvaluatedHours"],
        "minimumCoverageFraction":coverage>=g["minimumCoverageFraction"],
        "netMeanHourlyReturnMin":ns["meanHourlyReturn"]>=g["netMeanHourlyReturnMin"],
        "netAnnualizedSharpeMin":ns["annualizedSharpe"]>=g["netAnnualizedSharpeMin"],
        "netProfitFactorMin":ns["profitFactor"]>=g["netProfitFactorMin"],
        "stress1_5xMeanHourlyReturnMin":s15["meanHourlyReturn"]>=g["stress1_5xMeanHourlyReturnMin"],
        "stress2xMeanHourlyReturnMin":s2["meanHourlyReturn"]>=g["stress2xMeanHourlyReturnMin"],
        "positiveWeekFractionMin":ns["positiveWeekFraction"]>=g["positiveWeekFractionMin"],
        "bootstrapProbabilityPositiveMin":ns["bootstrap"]["probPositive"]>=g["bootstrapProbabilityPositiveMin"],
        "bootstrapCi95LowMin":ns["bootstrap"]["ci95Low"]>=g["bootstrapCi95LowMin"],
        "maxDrawdownMax":ns["maxDrawdown"]<=g["maxDrawdownMax"],
        "averageEligibleSymbolsMin":avg_eligible>=g["averageEligibleSymbolsMin"],
        "singleSymbolPositiveContributionMax":dominance<=g["singleSymbolPositiveContributionMax"]
    }
    gate={k:bool(v) for k,v in gate.items()}
    failed=[k for k,v in gate.items() if not v]
    closeout={
        "schema":"mgpt_arp23_oid1_development_closeout_v1","generation":"ARP23-OID1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS_DEVELOPMENT_GATE" if not failed else "REJECT_DEVELOPMENT_GATE",
        "productionAuthority":False,"r15MutationAllowed":False,
        "validationAuthorized":not failed,"sealedHoldoutOpened":False,
        "lockSha256":hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "development":{
            "expectedHours":expected,"evaluatedHours":len(net),"activeCoverageHours":active_hours,
            "coverageFraction":coverage,"averageEligibleSymbols":avg_eligible,
            "meanTurnover":float(np.mean(turnover)),"singleSymbolPositiveContribution":dominance,
            "gross":gs,"baseCost":ns,"stress1_5x":s15,"stress2x":s2,
            "symbolGrossContributions":{s:float(contrib[i]) for i,s in enumerate(SYMS)}
        },
        "gate":gate,"failedChecks":failed,
        "nextAction":"Open unchanged fresh validation only." if not failed else "Close ARP23-OID1; validation and holdout remain sealed."
    }
    transport_out={
        "schema":"mgpt_arp23_oid1_transport_diagnostics_v1","generation":"ARP23-OID1",
        "generatedAtUtc":datetime.now(timezone.utc).isoformat(),"checksumRequired":True,
        "manifestCount":len(manifest),"symbols":diagnostics
    }
    cp=OUT/'ARP23_OID1_DEVELOPMENT_CLOSEOUT_20260915.json'
    tp=OUT/'ARP23_OID1_TRANSPORT_DIAGNOSTICS_20260915.json'
    cp.write_text(json.dumps(closeout,indent=2)+'\n'); tp.write_text(json.dumps(transport_out,indent=2)+'\n')
    sums=[]
    for p in sorted(OUT.glob('ARP23_OID1_*_20260915.json')):
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/'ARP23_OID1_SHA256SUMS_20260915.txt').write_text('\n'.join(sums)+'\n')
    print(json.dumps({"status":closeout["status"],"failedChecks":failed,"development":closeout["development"]},indent=2))

if __name__=='__main__':
    main()