#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,io,json,math,time,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import requests

ROOT=Path(__file__).resolve().parent
LOCK_PATH=ROOT/'ARP28_ILLIQ1_AMIHUD_WEEKLY_LOCK_20260915.json'
OUT=ROOT/'results';OUT.mkdir(parents=True,exist_ok=True)
LOCK=json.loads(LOCK_PATH.read_text())
SYMS=LOCK['universe']['symbols']
BASE='https://data.binance.vision/data/futures/um/monthly/klines'
WARM=datetime.fromisoformat(LOCK['windows']['warmupStart'].replace('Z','+00:00'))
DEV_START=datetime.fromisoformat(LOCK['windows']['development']['start'].replace('Z','+00:00'))
DEV_END=datetime.fromisoformat(LOCK['windows']['development']['endExclusive'].replace('Z','+00:00'))
DAY=86400000
SESSION=requests.Session();SESSION.headers.update({'User-Agent':'MarketsGPT-ARP28-ILLIQ1/1.0'})

def ms(x):return int(x.timestamp()*1000)
def month_keys(a,b):
 y,m=a.year,a.month;out=[]
 while (y,m)<=(b.year,b.month):
  out.append(f'{y:04d}-{m:02d}');m+=1
  if m==13:y+=1;m=1
 return out
MONTHS=month_keys(WARM,DEV_END+timedelta(days=7))

def get_bytes(url,tries=5):
 last=None
 for k in range(tries):
  try:
   r=SESSION.get(url,timeout=45)
   if r.status_code==200:return r.content
   if r.status_code==404:return None
   last=f'HTTP {r.status_code}: {r.text[:120]}'
  except Exception as e:last=repr(e)
  time.sleep(min(8,.75*(2**k)))
 raise RuntimeError(f'download failed {url}: {last}')

def download_month(sym,ym):
 name=f'{sym}-1d-{ym}.zip';url=f'{BASE}/{sym}/1d/{name}';blob=get_bytes(url)
 if blob is None:return {'symbol':sym,'ym':ym,'status':'404','rows':[]}
 chk=get_bytes(url+'.CHECKSUM')
 if chk is None:raise RuntimeError(f'missing CHECKSUM {url}')
 exp=chk.decode('utf-8','replace').strip().split()[0].lower();got=hashlib.sha256(blob).hexdigest()
 if exp!=got:raise RuntimeError(f'checksum mismatch {name}')
 with zipfile.ZipFile(io.BytesIO(blob)) as z:
  ns=[n for n in z.namelist() if not n.endswith('/')]
  if len(ns)!=1:raise RuntimeError(f'unexpected members {name}: {ns}')
  raw=z.read(ns[0]).decode('utf-8-sig','replace')
 rows=[]
 for rec in csv.reader(io.StringIO(raw)):
  if not rec:continue
  try:t=int(float(rec[0]))
  except Exception:continue
  if t>10**14:t//=1000
  if len(rec)<8:continue
  try:o=float(rec[1]);c=float(rec[4]);qv=float(rec[7])
  except Exception:continue
  if min(o,c)>0 and qv>0 and all(map(math.isfinite,[o,c,qv])):rows.append((t,o,c,qv))
 return {'symbol':sym,'ym':ym,'status':'ok','sha256':got,'rows':rows}

def transport():
 jobs=[]
 with ThreadPoolExecutor(max_workers=32) as ex:
  for s in SYMS:
   for ym in MONTHS:jobs.append(ex.submit(download_month,s,ym))
  got=[f.result() for f in as_completed(jobs)]
 data={s:{} for s in SYMS};manifest=[];dup={s:0 for s in SYMS}
 for x in got:
  manifest.append({k:v for k,v in x.items() if k!='rows'})
  for row in x['rows']:
   if row[0] in data[x['symbol']]:dup[x['symbol']]+=1
   data[x['symbol']][row[0]]=row[1:]
 diag=[]
 for s in SYMS:
  ts=sorted(data[s]);gaps=sum(1 for a,b in zip(ts,ts[1:]) if b-a!=DAY)
  diag.append({'symbol':s,'rows':len(ts),'firstUtc':datetime.fromtimestamp(ts[0]/1000,timezone.utc).isoformat() if ts else None,'lastUtc':datetime.fromtimestamp(ts[-1]/1000,timezone.utc).isoformat() if ts else None,'duplicateRows':dup[s],'postFirstGapCount':gaps})
 return data,manifest,diag

def pf(a):
 a=np.asarray(a,float);p=float(a[a>0].sum());n=float(-a[a<0].sum());return p/n if n>0 else (999.0 if p>0 else 0.0)
def maxdd(a):
 e=1.;pk=1.;d=0.
 for r in a:e=max(1e-12,e*(1+float(r)));pk=max(pk,e);d=max(d,1-e/pk)
 return d
def boot(a,seed,it=10000,block=4):
 a=np.asarray(a,float);n=len(a)
 if not n:return {'probPositive':0.,'ci95Low':0.,'ci95High':0.}
 rng=np.random.default_rng(seed);z=np.empty(it)
 for k in range(it):
  vals=[]
  while len(vals)<n:
   st=int(rng.integers(0,n));take=min(block,n-len(vals));vals.extend(a[(st+np.arange(take))%n])
  z[k]=float(np.mean(vals))
 z.sort();return {'probPositive':float(np.mean(z>0)),'ci95Low':float(z[int(.025*(it-1))]),'ci95High':float(z[int(.975*(it-1))])}
def stats(a,dates,seed):
 a=np.asarray(a,float);m=float(a.mean()) if len(a) else 0.;sd=float(a.std(ddof=1)) if len(a)>1 else 0.;months={}
 for r,d in zip(a,dates):months[d[:7]]=months.get(d[:7],0.)+float(r)
 return {'n':len(a),'meanWeeklyReturn':m,'annualizedMean':m*52,'annualizedSharpe':float(m/sd*math.sqrt(52)) if sd>0 else 0.,'profitFactor':pf(a),'maxDrawdown':maxdd(a),'positiveMonthFraction':sum(v>0 for v in months.values())/max(1,len(months)),'monthSums':months,'bootstrap':boot(a,seed)}
def monday_grid(a,b):
 out=[];t=a
 while t<b:out.append(t);t+=timedelta(days=7)
 return out

def main():
 data,manifest,diag=transport();grid=monday_grid(DEV_START,DEV_END)
 prev=np.zeros(len(SYMS));gross=[];tos=[];dates=[];elig_counts=[];active=0;contrib=np.zeros(len(SYMS));selection={s:{'long':0,'short':0} for s in SYMS}
 for dt in grid:
  t=ms(dt);nxt=t+7*DAY;cand=[]
  for i,s in enumerate(SYMS):
   d=data[s];r0=d.get(t);r1=d.get(nxt)
   if r0 is None or r1 is None:continue
   ill=[];qv=[]
   ok=True
   for k in range(1,31):
    cur=d.get(t-k*DAY);prevday=d.get(t-(k+1)*DAY)
    if cur is None or prevday is None:ok=False;break
    dret=cur[1]/prevday[1]-1.
    if not math.isfinite(dret) or cur[2]<=0:ok=False;break
    ill.append(abs(dret)/cur[2]);qv.append(cur[2])
   if not ok:continue
   avgq=float(np.mean(qv))
   if avgq<LOCK['universe']['liquidityFloorPrior30dMeanQuoteVolumeUsd']:continue
   sig=float(np.mean(ill));ret=r1[0]/r0[0]-1.
   if math.isfinite(sig) and math.isfinite(ret):cand.append((i,s,sig,avgq,ret))
  elig_counts.append(len(cand));desired=np.zeros(len(SYMS))
  if len(cand)>=LOCK['universe']['minimumEligiblePerRebalance']:
   active+=1
   ranked=sorted(cand,key=lambda x:(-x[2],x[1]));q=len(ranked)//4
   longs=ranked[:q];shorts=ranked[-q:]
   for i,s,_,_,_ in longs:desired[i]+=.5/q;selection[s]['long']+=1
   for i,s,_,_,_ in shorts:desired[i]-=.5/q;selection[s]['short']+=1
  retmap={i:r for i,_,_,_,r in cand};g=0.;by=np.zeros(len(SYMS))
  for i,w in enumerate(desired):
   if w and i in retmap:
    z=w*retmap[i];g+=z;by[i]=z
  contrib+=by;gross.append(g);tos.append(float(np.abs(desired-prev).sum()));dates.append(dt.date().isoformat());prev=desired
 if tos:tos[-1]+=float(np.abs(prev).sum())
 gross=np.asarray(gross);tos=np.asarray(tos)
 c=LOCK['costs'];net=gross-tos*c['baseOneWayRate'];n2=gross-tos*c['stress2xOneWayRate'];n4=gross-tos*c['stress4xOneWayRate']
 gs=stats(gross,dates,28100);ns=stats(net,dates,28101);s2=stats(n2,dates,28102);s4=stats(n4,dates,28103)
 pos=contrib[contrib>0];dom=float(pos.max()/pos.sum()) if len(pos) and pos.sum()>0 else 0.
 cov=active/max(1,len(grid));avg=float(np.mean(elig_counts)) if elig_counts else 0.;half=len(net)//2;fh=float(net[:half].sum());sh=float(net[half:].sum())
 gate_cfg=LOCK['developmentGate']
 gate={
  'minimumEvaluatedWeeks':len(net)>=gate_cfg['minimumEvaluatedWeeks'],
  'minimumCoverageFraction':cov>=gate_cfg['minimumCoverageFraction'],
  'averageEligibleSymbolsMin':avg>=gate_cfg['averageEligibleSymbolsMin'],
  'netMeanWeeklyReturnMin':ns['meanWeeklyReturn']>=gate_cfg['netMeanWeeklyReturnMin'],
  'netAnnualizedSharpeMin':ns['annualizedSharpe']>=gate_cfg['netAnnualizedSharpeMin'],
  'netProfitFactorMin':ns['profitFactor']>=gate_cfg['netProfitFactorMin'],
  'stress2xMeanWeeklyReturnMin':s2['meanWeeklyReturn']>=gate_cfg['stress2xMeanWeeklyReturnMin'],
  'stress4xMeanWeeklyReturnMin':s4['meanWeeklyReturn']>=gate_cfg['stress4xMeanWeeklyReturnMin'],
  'positiveMonthFractionMin':ns['positiveMonthFraction']>=gate_cfg['positiveMonthFractionMin'],
  'firstHalfNetReturnPositive':fh>0,
  'secondHalfNetReturnPositive':sh>0,
  'bootstrapProbabilityPositiveMin':ns['bootstrap']['probPositive']>=gate_cfg['bootstrapProbabilityPositiveMin'],
  'bootstrapCi95LowMin':ns['bootstrap']['ci95Low']>=gate_cfg['bootstrapCi95LowMin'],
  'maxDrawdownMax':ns['maxDrawdown']<=gate_cfg['maxDrawdownMax'],
  'singleSymbolPositiveContributionMax':dom<=gate_cfg['singleSymbolPositiveContributionMax']
 }
 gate={k:bool(v) for k,v in gate.items()};failed=[k for k,v in gate.items() if not v]
 close={'schema':'mgpt_arp28_illiq1_development_closeout_v1','generation':'ARP28-ILLIQ1','generatedAtUtc':datetime.now(timezone.utc).isoformat(),'status':'PASS_DEVELOPMENT_GATE' if not failed else 'REJECT_DEVELOPMENT_GATE','productionAuthority':False,'r15MutationAllowed':False,'validationAuthorized':not failed,'sealedHoldoutOpened':False,'lockSha256':hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),'development':{'expectedWeeks':len(grid),'evaluatedWeeks':len(net),'activeCoverageWeeks':active,'coverageFraction':cov,'averageEligibleSymbols':avg,'meanTurnover':float(tos.mean()) if len(tos) else 0.,'singleSymbolPositiveContribution':dom,'firstHalfNetReturn':fh,'secondHalfNetReturn':sh,'gross':gs,'baseCost':ns,'stress2x':s2,'stress4x':s4,'symbolGrossContributions':{s:float(contrib[i]) for i,s in enumerate(SYMS)},'selectionCounts':selection},'gate':gate,'failedChecks':failed,'nextAction':'Open unchanged fresh validation only.' if not failed else 'Close ARP28-ILLIQ1; validation and holdout remain sealed.'}
 trans={'schema':'mgpt_arp28_illiq1_transport_diagnostics_v1','generation':'ARP28-ILLIQ1','generatedAtUtc':datetime.now(timezone.utc).isoformat(),'checksumRequired':True,'manifestCount':len(manifest),'symbols':diag}
 cp=OUT/'ARP28_ILLIQ1_DEVELOPMENT_CLOSEOUT_20260915.json';tp=OUT/'ARP28_ILLIQ1_TRANSPORT_DIAGNOSTICS_20260915.json'
 cp.write_text(json.dumps(close,indent=2)+'\n');tp.write_text(json.dumps(trans,indent=2)+'\n')
 sums=[]
 for p in sorted(OUT.glob('ARP28_ILLIQ1_*_20260915.json')):sums.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}')
 (OUT/'ARP28_ILLIQ1_SHA256SUMS_20260915.txt').write_text('\n'.join(sums)+'\n')
 print(json.dumps({'status':close['status'],'failedChecks':failed,'development':{k:v for k,v in close['development'].items() if k not in ('symbolGrossContributions','selectionCounts')}},indent=2))

if __name__=='__main__':main()
