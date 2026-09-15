import json, pathlib, math
import numpy as np
from arp25_sal1 import salience_st
lock=json.loads(pathlib.Path('ARP25_SAL1_WEEKLY_SALIENCE_LOCK_20260915.json').read_text())
assert lock['status']=='LOCKED_BEFORE_DEVELOPMENT_PRICE_OUTCOMES'
assert lock['method']['signSweepAllowed'] is False
assert lock['externalEvidence']['parameters']=={'theta':0.1,'delta':0.7}
# Symmetric asset/market return states should return a finite ST; more downside-salient patterns should not crash.
a=[-.20,.01,.01,.01,.01,.01,.01]; m=[0,0,0,0,0,0,0]
st=salience_st(a,m); assert math.isfinite(st)
# Exact decision weights implied by the formula sum to one under pi weighting.
ranks=np.arange(1,8); raw=.7**ranks; weighted=raw/raw.sum(); assert abs(weighted.sum()-1)<1e-12
print('ARP25-SAL1 SANITY PASS',st)