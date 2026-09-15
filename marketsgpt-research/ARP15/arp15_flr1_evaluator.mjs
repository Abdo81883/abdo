export const VERSION='arp15-flr1-v1';
const H=3600000, YEAR_EVENTS=1095;
export const COST={base:.0006,stress1_5:.0009,stress2:.0012};
const num=x=>{const n=Number(x);return Number.isFinite(n)?n:null};
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:0;
function sd(a){if(a.length<2)return 0;const m=mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/(a.length-1));}
function pf(a){let p=0,n=0;for(const x of a){if(x>0)p+=x;else if(x<0)n-=x;}return n?p/n:(p?999:0);}
function maxDD(a){let e=1,pk=1,d=0;for(const r of a){e*=Math.max(1e-12,1+r);pk=Math.max(pk,e);d=Math.max(d,1-e/pk);}return d;}
function rng(seed){let x=seed>>>0;return()=>((x=(1664525*x+1013904223)>>>0)/4294967296);}
function bootstrap(a,{iterations=10000,blockLength=9,seed=15101}={}){if(!a.length)return{probPositive:0,ci95Low:0,ci95High:0};const R=rng(seed),n=a.length,z=[];for(let k=0;k<iterations;k++){let s=0,c=0;while(c<n){const st=Math.floor(R()*n);for(let j=0;j<blockLength&&c<n;j++,c++)s+=a[(st+j)%n];}z.push(s/n);}z.sort((x,y)=>x-y);return{probPositive:z.filter(x=>x>0).length/iterations,ci95Low:z[Math.floor(.025*(iterations-1))],ci95High:z[Math.floor(.975*(iterations-1))]};}
function barMap(a=[]){const m=new Map();for(const x of a){const t=Number(x.openTime??x.t??x.timestamp),o=num(x.open??x.o),q=num(x.quoteVolume??x.q??x.quote_asset_volume);if(Number.isFinite(t)&&o>0&&q!=null)m.set(t,{t,o,q});}return m;}
function fundArr(a=[]){return a.map(x=>({t:Number(x.fundingTime??x.t??x.time),r:num(x.fundingRate??x.r??x.rate)})).filter(x=>Number.isFinite(x.t)&&x.r!=null).sort((a,b)=>a.t-b.t);}
function lowerBound(a,t){let l=0,r=a.length;while(l<r){const m=(l+r)>>1;if(a[m].t<t)l=m+1;else r=m;}return l;}
function liquidity30d(bars,t){let sum=0,n=0;for(let x=t-30*24*H;x<t;x+=H){const b=bars.get(x);if(!b)continue;sum+=b.q;n++;}return{meanDailyQuoteVolume:n>=600?sum/30:0,hours:n};}
function hasFundingInside(f,t0,t1){const i=lowerBound(f,t0+1);return i<f.length&&f[i].t<=t1;}
export function evaluate(payload,{startMs,endMs,bootstrapSeed=15101}={}){
 const symbols=Object.keys(payload.symbols||{}).sort(),data={};for(const s of symbols)data[s]={bars:barMap(payload.symbols[s].bars),fund:fundArr(payload.symbols[s].funding)};
 const start=Number(startMs??Date.parse(payload.start)),end=Number(endMs??Date.parse(payload.end));
 const events=new Set();for(const s of symbols)for(const x of data[s].fund)if(x.t>=start&&x.t<end)events.add(x.t);
 const rows=[],contrib=Object.fromEntries(symbols.map(s=>[s,0]));
 for(const t of [...events].sort((a,b)=>a-b)){const cand=[];
   for(const s of symbols){const d=data[s],fi=lowerBound(d.fund,t);if(fi>=d.fund.length||d.fund[fi].t!==t)continue;const rate=d.fund[fi].r;if(rate==null)continue;if(hasFundingInside(d.fund,t,t+7*H))continue;
     const e=d.bars.get(t+H),x=d.bars.get(t+7*H);if(!e||!x)continue;const liq=liquidity30d(d.bars,t);if(liq.meanDailyQuoteVolume<5000000)continue;cand.push({s,rate,ret:x.o/e.o-1,liq:liq.meanDailyQuoteVolume});}
   if(cand.length<16)continue;cand.sort((a,b)=>b.rate-a.rate||a.s.localeCompare(b.s));const q=Math.max(3,Math.floor(cand.length*.20)),L=cand.slice(0,q),S=cand.slice(-q),w={};for(const c of cand)w[c.s]=0;for(const c of L)w[c.s]=.5/q;for(const c of S)w[c.s]=-.5/q;
   let gross=0;for(const c of cand){const z=(w[c.s]||0)*c.ret;gross+=z;contrib[c.s]+=z;}const turnover=2;
   rows.push({t,validSymbols:cand.length,long:L.map(x=>x.s),short:S.map(x=>x.s),gross,turnover,net:gross-turnover*COST.base,stress1_5:gross-turnover*COST.stress1_5,stress2:gross-turnover*COST.stress2});
 }
 const vals=rows.map(x=>x.net),m=mean(vals),v=sd(vals),months={},years={};for(const r of rows){const d=new Date(r.t),mk=d.toISOString().slice(0,7),yk=d.toISOString().slice(0,4);months[mk]=(months[mk]||0)+r.net;years[yk]=(years[yk]||0)+r.net;}
 const pos=Object.entries(contrib).filter(([,v])=>v>0),posSum=pos.reduce((s,[,v])=>s+v,0),top=pos.sort((a,b)=>b[1]-a[1])[0]||[null,0],concentration=posSum>0?top[1]/posSum:1;
 return{version:VERSION,nEvents:rows.length,meanNetReturnPerEvent:m,annualizedMean:m*YEAR_EVENTS,annualizedSharpe:v?m/v*Math.sqrt(YEAR_EVENTS):0,profitFactor:pf(vals),stress1_5MeanPerEvent:mean(rows.map(x=>x.stress1_5)),stress2xMeanPerEvent:mean(rows.map(x=>x.stress2)),maxDrawdown:maxDD(vals),positiveMonthFraction:Object.values(months).filter(x=>x>0).length/Math.max(1,Object.keys(months).length),byMonth:months,byYear:years,bootstrap:bootstrap(vals,{seed:bootstrapSeed}),singleSymbolPositiveContributionShare:concentration,maxPositiveSymbol:top[0],rows};
}
export function gate(s,stage='development'){const G={
 development:{minimumEvents:600,mean:.00015,sharpe:.75,pf:1.05,s15:0,s2:0,posM:.55,boot:.95,ci:0,dd:.25,conc:.30},
 validation:{minimumEvents:600,mean:.00008,sharpe:.45,pf:1.03,s2:0,posM:.50,boot:.85,ci:-.00015,dd:.22,conc:.35},
 holdout:{minimumEvents:900,mean:0,sharpe:.30,pf:1.00,s15:0,boot:.80,dd:.22}
 }[stage];if(!G)throw Error('bad stage');const c={minimumEvents:s.nEvents>=G.minimumEvents,meanNetReturnPerEventMin:s.meanNetReturnPerEvent>=G.mean,annualizedSharpeMin:s.annualizedSharpe>=G.sharpe,profitFactorMin:s.profitFactor>=G.pf,movingBlockBootstrapProbabilityPositiveMin:s.bootstrap.probPositive>=G.boot,maxDrawdownMax:s.maxDrawdown<=G.dd};if(G.s15!=null)c.stress1_5xMeanPerEventMin=s.stress1_5MeanPerEvent>=G.s15;if(G.s2!=null)c.stress2xMeanPerEventMin=s.stress2xMeanPerEvent>=G.s2;if(G.posM!=null)c.positiveMonthFractionMin=s.positiveMonthFraction>=G.posM;if(G.ci!=null)c.movingBlockBootstrapCi95LowMin=s.bootstrap.ci95Low>=G.ci;if(G.conc!=null)c.singleSymbolPositiveContributionMax=s.singleSymbolPositiveContributionShare<=G.conc;return{stage,pass:Object.values(c).every(Boolean),checks:c,failed:Object.entries(c).filter(([,v])=>!v).map(([k])=>k)};}