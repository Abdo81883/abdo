#!/usr/bin/env python3
from __future__ import annotations
import io,json,math,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np, requests, openpyxl

ROOT=Path(__file__).resolve().parent
LOCK_PATH=ROOT/"ARP29_CTREND_EXT1_AUTHOR_POSTPUBLICATION_LOCK_20260915.json"
OUT=ROOT/"results";OUT.mkdir(parents=True,exist_ok=True)
LOCK=json.loads(LOCK_PATH.read_text())
SID=LOCK["source"]["publicUpdatedWeeklySheetId"]
URL=f"https://docs.google.com/spreadsheets/d/{SID}/export?format=xlsx"
r=requests.get(URL,timeout=60,headers={"User-Agent":"MarketsGPT-ARP29-EXT1/1.0"})
r.raise_for_status()
blob=r.content
wb=openpyxl.load_workbook(io.BytesIO(blob),read_only=True,data_only=True)
ws=wb["Data"]
rows=list(ws.iter_rows(values_only=True))
hdr=[str(x) if x is not None else "" for x in rows[0]]
idx={h:i for i,h in enumerate(hdr)}
need=["Time","CMKT","CSMB","CMOM","CTREND","CTREND_Ens"]
for x in need:
    if x not in idx: raise RuntimeError(f"missing column {x}")
data=[]
for rr in rows[1:]:
    t=rr[idx["Time"]]
    if t is None: continue
    if isinstance(t,str):
        t=datetime.fromisoformat(t.replace("Z","+00:00"))
    elif not isinstance(t,datetime):
        continue
    if t.tzinfo is None:t=t.replace(tzinfo=timezone.utc)
    row={"Time":t.astimezone(timezone.utc)}
    ok=True
    for c in need[1:]:
        try:v=float(rr[idx[c]])
        except Exception:ok=False;break
        if not math.isfinite(v):ok=False;break
        row[c]=v
    if ok:data.append(row)
data.sort(key=lambda x:x["Time"])

def pf(a):
    a=np.asarray(a,float);p=float(a[a>0].sum());n=float(-a[a<0].sum());return p/n if n>0 else (999.0 if p>0 else 0.0)
def maxdd(a):
    e=1.;pk=1.;d=0.
    for x in a:
        e=max(1e-12,e*(1+float(x)));pk=max(pk,e);d=max(d,1-e/pk)
    return d
def boot(a,seed=29101,it=10000,block=4):
    a=np.asarray(a,float);n=len(a);rng=np.random.default_rng(seed);z=np.empty(it)
    for k in range(it):
        vals=[]
        while len(vals)<n:
            st=int(rng.integers(0,n));take=min(block,n-len(vals));vals.extend(a[(st+np.arange(take))%n])
        z[k]=float(np.mean(vals))
    z.sort();return {"probPositive":float(np.mean(z>0)),"ci95Low":float(z[int(.025*(it-1))]),"ci95High":float(z[int(.975*(it-1))])}
def stats(rs,dates,seed=29101):
    a=np.asarray(rs,float);m=float(a.mean()) if len(a) else 0.;sd=float(a.std(ddof=1)) if len(a)>1 else 0.;months={}
    for x,d in zip(a,dates):months[d.strftime("%Y-%m")]=months.get(d.strftime("%Y-%m"),0.)+float(x)
    return {"n":len(a),"meanWeeklyReturn":m,"annualizedMean":m*52,"annualizedSharpe":float(m/sd*math.sqrt(52)) if sd>0 else 0.,"profitFactor":pf(a),"maxDrawdown":maxdd(a),"positiveMonthFraction":sum(v>0 for v in months.values())/max(1,len(months)),"monthSums":months,"bootstrap":boot(a,seed)}
def slice_stats(start,end,col,seed):
    s=datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e=datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    sub=[x for x in data if s<=x["Time"]<e]
    return stats([x[col] for x in sub],[x["Time"] for x in sub],seed),sub
windows=LOCK["windows"]
res={"schema":"mgpt_arp29_ctrend_ext1_author_postpublication_result_v1","generation":"ARP29-CTREND-EXT1","generatedAtUtc":datetime.now(timezone.utc).isoformat(),"sourceSha256":hashlib.sha256(blob).hexdigest(),"sourceRows":len(data),"sourceFirst":data[0]["Time"].isoformat() if data else None,"sourceLast":data[-1]["Time"].isoformat() if data else None,"windows":{}}
seed=29101
for wi,(name,w) in enumerate(windows.items()):
    p,_=slice_stats(w["start"],w["endExclusive"],"CTREND",seed+wi)
    q,_=slice_stats(w["start"],w["endExclusive"],"CTREND_Ens",seed+100+wi)
    res["windows"][name]={"authority":w["evidenceAuthority"],"CTREND":p,"CTREND_Ens":q}
clean=res["windows"]["cleanPostPublication"]["CTREND"];g=LOCK["primaryExternalGate"]
gate={
    "minimumWeeks":clean["n"]>=g["minimumWeeks"],
    "meanWeeklyReturnMin":clean["meanWeeklyReturn"]>=g["meanWeeklyReturnMin"],
    "annualizedSharpeMin":clean["annualizedSharpe"]>=g["annualizedSharpeMin"],
    "profitFactorMin":clean["profitFactor"]>=g["profitFactorMin"],
    "positiveMonthFractionMin":clean["positiveMonthFraction"]>=g["positiveMonthFractionMin"],
    "bootstrapProbabilityPositiveMin":clean["bootstrap"]["probPositive"]>=g["bootstrapProbabilityPositiveMin"],
    "bootstrapCi95LowMin":clean["bootstrap"]["ci95Low"]>=g["bootstrapCi95LowMin"],
    "maxDrawdownMax":clean["maxDrawdown"]<=g["maxDrawdownMax"]
}
gate={k:bool(v) for k,v in gate.items()};res["gate"]={"pass":all(gate.values()),"checks":gate,"failed":[k for k,v in gate.items() if not v]}
res["decision"]="AUTHOR_POSTPUBLICATION_BENCHMARK_PASS_AUTHORIZE_PRELOCKED_BINANCE_ADAPTATION" if res["gate"]["pass"] else "AUTHOR_POSTPUBLICATION_BENCHMARK_FAIL_DO_NOT_BUILD_ADAPTATION"
p=OUT/"ARP29_CTREND_EXT1_AUTHOR_POSTPUBLICATION_RESULT_20260915.json";p.write_text(json.dumps(res,indent=2)+"\n")
(OUT/"ARP29_CTREND_EXT1_AUTHOR_POSTPUBLICATION_RESULT_20260915.sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
print(json.dumps({"decision":res["decision"],"gate":res["gate"],"cleanPostPublication":res["windows"]["cleanPostPublication"]},indent=2))
