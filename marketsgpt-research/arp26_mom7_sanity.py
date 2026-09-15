import json,pathlib
lock=json.loads(pathlib.Path('ARP26_MOM7_WEEKLY_CROSS_SECTION_LOCK_20260915.json').read_text())
assert lock['status']=='LOCKED_BEFORE_DEVELOPMENT_PRICE_OUTCOMES'
assert lock['method']['formationDays']==7 and lock['method']['holdingDays']==7
assert lock['method']['signSweepAllowed'] is False
assert lock['universe']['liquiditySelection'].startswith('retain top 40')
print('ARP26-MOM7 SANITY PASS')