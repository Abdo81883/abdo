import json, pathlib
lock=json.loads(pathlib.Path("ARP22_POS4_INDEPENDENT_TIME_REPLICATION_LOCK_20260915.json").read_text())
assert lock["generation"]=="ARP22-POS4"
assert lock["status"]=="LOCKED_BEFORE_POS4_DEVELOPMENT_OUTCOMES"
assert lock["r15MutationAllowed"] is False
assert lock["signalContract"]["thresholdRescueAllowed"] is False
assert lock["signalContract"]["parameterSweepAllowed"] is False
assert lock["developmentGate"]["bootstrapProbabilityPositiveMin"]==0.99
assert lock["developmentGate"]["bootstrapCi95LowMin"]==0.0
assert lock["developmentGate"]["netAnnualizedSharpeMin"]==1.25
assert len(lock["cohort"]["symbols"])==132
assert len(lock["freshValidationCohort"]["symbols"])==35
assert len(lock["sealedHoldoutCohort"]["symbols"])==35
assert set(lock["freshValidationCohort"]["symbols"]).isdisjoint(lock["sealedHoldoutCohort"]["symbols"])
assert set(lock["freshValidationCohort"]["symbols"]).isdisjoint(lock["cohort"]["symbols"])
assert set(lock["sealedHoldoutCohort"]["symbols"]).isdisjoint(lock["cohort"]["symbols"])
assert lock["windows"]["development"]["endExclusive"] <= lock["windows"]["freshValidation"]["start"]
assert lock["windows"]["freshValidation"]["endExclusive"] <= lock["windows"]["sealedHoldout"]["start"]
print("ARP22-POS4 SANITY PASS")
