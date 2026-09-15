#!/usr/bin/env python3
from __future__ import annotations
import json, time, hashlib
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT/"ARP27_TVOL1_TURNOVER_VOLATILITY_LOCK_20260915.json").read_text())
OUT = ROOT/"results"; OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://api.coingecko.com/api/v3"
S = requests.Session(); S.headers.update({"User-Agent":"MarketsGPT-ARP27-TVOL1-Probe/1.0"})

def get_json(path, params=None, tries=8):
    last=None
    for k in range(tries):
        try:
            r=S.get(BASE+path, params=params, timeout=45)
            if r.status_code==200: return r.json()
            last={"status":r.status_code,"body":r.text[:300]}
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2**k)); continue
            raise RuntimeError(f"CoinGecko {last}")
        except Exception as e:
            last=repr(e)
            if k==tries-1: raise
            time.sleep(min(60, 2**k))
    raise RuntimeError(last)

def base_symbol(binance_symbol):
    assert binance_symbol.endswith("USDT")
    return binance_symbol[:-4].lower()

symbols=LOCK["cohort"]["symbols"]
bases=[base_symbol(x) for x in symbols]
all_rows=[]
for i in range(0,len(bases),40):
    batch=bases[i:i+40]
    params={
        "vs_currency":"usd",
        "symbols":",".join(batch),
        "include_tokens":"all",
        "order":"market_cap_desc",
        "per_page":250,
        "page":1,
        "sparkline":"false"
    }
    rows=get_json("/coins/markets",params)
    all_rows.extend(rows)
    time.sleep(4)

by={}
for r in all_rows:
    by.setdefault(str(r.get("symbol","")).lower(),[]).append(r)

mapping={}
unresolved=[]
ambiguous=[]
for bs in symbols:
    b=base_symbol(bs)
    cand=by.get(b,[])
    cand=sorted(cand,key=lambda r:(
        -(float(r.get("market_cap") or 0)),
        int(r.get("market_cap_rank") or 10**9),
        str(r.get("id",""))
    ))
    chosen=cand[0] if cand else None
    mapping[bs]={
        "baseSymbol":b,
        "chosen":None if not chosen else {
            "id":chosen.get("id"),"name":chosen.get("name"),"symbol":chosen.get("symbol"),
            "marketCap":chosen.get("market_cap"),"marketCapRank":chosen.get("market_cap_rank")
        },
        "candidates":[{
            "id":r.get("id"),"name":r.get("name"),"marketCap":r.get("market_cap"),
            "marketCapRank":r.get("market_cap_rank")
        } for r in cand[:8]]
    }
    if not chosen: unresolved.append(bs)
    elif len(cand)>1:
        top=float(cand[0].get("market_cap") or 0); sec=float(cand[1].get("market_cap") or 0)
        if top<=0 or (sec>0 and top/sec<3): ambiguous.append(bs)

chosen_ids=[v["chosen"]["id"] for v in mapping.values() if v["chosen"]]
# Probe representative large/mid/small mapped IDs only; this is transport metadata, not strategy PnL.
probe_ids=[]
for preferred in ("BTCUSDT","ETHUSDT","DOGEUSDT","AAVEUSDT","SANDUSDT"):
    x=mapping.get(preferred,{}).get("chosen")
    if x and x["id"] not in probe_ids: probe_ids.append(x["id"])
probe_ids=probe_ids[:5]
history_probe={}
for cid in probe_ids:
    d=get_json(f"/coins/{cid}/market_chart",{"vs_currency":"usd","days":"max","interval":"daily"})
    caps=d.get("market_caps") or []; vols=d.get("total_volumes") or []
    history_probe[cid]={
        "marketCapPoints":len(caps),"volumePoints":len(vols),
        "firstMarketCapUtc": datetime.fromtimestamp(caps[0][0]/1000,timezone.utc).isoformat() if caps else None,
        "lastMarketCapUtc": datetime.fromtimestamp(caps[-1][0]/1000,timezone.utc).isoformat() if caps else None,
        "firstVolumeUtc": datetime.fromtimestamp(vols[0][0]/1000,timezone.utc).isoformat() if vols else None,
        "lastVolumeUtc": datetime.fromtimestamp(vols[-1][0]/1000,timezone.utc).isoformat() if vols else None,
        "containsDevelopmentStart": bool(caps and caps[0][0] <= int(datetime(2024,7,1,tzinfo=timezone.utc).timestamp()*1000))
    }
    time.sleep(4)

mapped=sum(1 for v in mapping.values() if v["chosen"])
probe_ok=bool(history_probe) and all(v["containsDevelopmentStart"] and v["marketCapPoints"]>365 and v["volumePoints"]>365 for v in history_probe.values())
audit={
    "schema":"mgpt_arp27_tvol1_coingecko_keyless_probe_v1",
    "generation":"ARP27-TVOL1",
    "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
    "status":"PASS_PROVIDER_TRANSPORT_PROBE" if mapped>=110 and probe_ok else "FAIL_PROVIDER_TRANSPORT_PROBE_NO_ALPHA_JUDGMENT",
    "outcomesOpened":False,
    "alphaCalculated":False,
    "provider":{
        "name":"CoinGecko Keyless Public API",
        "baseUrl":BASE,
        "identityEndpoint":"/coins/markets",
        "historicalEndpoint":"/coins/{id}/market_chart?days=max&interval=daily",
        "fields":["market_caps","total_volumes"]
    },
    "mappingCoverage":{"mapped":mapped,"total":len(symbols),"fraction":mapped/len(symbols),"unresolved":unresolved,"ambiguous":ambiguous},
    "mapping":mapping,
    "historyProbe":history_probe,
    "decision":"Provider transport may be frozen before outcomes." if mapped>=110 and probe_ok else "Keep ARP27 at DATA_BOUNDARY; do not calculate returns."
}
p=OUT/"ARP27_TVOL1_COINGECKO_PROBE_20260915.json"
p.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
(OUT/"ARP27_TVOL1_COINGECKO_PROBE_20260915.sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
print(json.dumps({"status":audit["status"],"mappingCoverage":audit["mappingCoverage"],"historyProbe":history_probe},indent=2))
