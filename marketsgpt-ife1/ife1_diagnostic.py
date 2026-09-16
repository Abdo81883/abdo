#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
P=OUT/"MGPT_IFE1_DEVELOPMENT_PREDICTIONS_20260916.csv"

def qdict(s):
    x=pd.to_numeric(s,errors="coerce").dropna()
    qs=[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]
    return {str(q):float(x.quantile(q)) for q in qs} if len(x) else {}

def grouped(z,col):
    out=[]
    for k,g in z.groupby(col,dropna=False):
        y=(g.netR>0).astype(int)
        auc=float(roc_auc_score(y,g.pWinScore)) if y.nunique()==2 else None
        out.append({
            col:str(k),"rows":int(len(g)),
            "meanNetR":float(g.netR.mean()),
            "positiveRate":float((g.netR>0).mean()),
            "meanPredictedNetR":float(g.predictedNetR.mean()),
            "meanPWinScore":float(g.pWinScore.mean()),
            "pWinAuc":auc
        })
    return out

def main():
    d=pd.read_csv(P)
    req={"predictedNetR","ridgePredR","hgbPredR","pWinScore","netR","stressNetR","marketFamily","timeframe","signalUtc"}
    if not req.issubset(d.columns): raise RuntimeError(f"missing {sorted(req-set(d.columns))}")
    y=(d.netR>0).astype(int)
    auc=float(roc_auc_score(y,d.pWinScore)) if y.nunique()==2 else None

    rule={
      "predictedNetR_ge_0.10": d.predictedNetR>=0.10,
      "ridge_gt_0": d.ridgePredR>0,
      "hgb_gt_0": d.hgbPredR>0,
      "pWinScore_ge_0.55": d.pWinScore>=0.55,
    }
    counts={k:int(v.sum()) for k,v in rule.items()}
    combos={
      "all_locked_conditions":int((rule["predictedNetR_ge_0.10"]&rule["ridge_gt_0"]&rule["hgb_gt_0"]&rule["pWinScore_ge_0.55"]).sum()),
      "positive_both_regressors":int((rule["ridge_gt_0"]&rule["hgb_gt_0"]).sum()),
      "positive_both_plus_pWin":int((rule["ridge_gt_0"]&rule["hgb_gt_0"]&rule["pWinScore_ge_0.55"]).sum()),
      "predicted_ge_0.10_plus_pWin":int((rule["predictedNetR_ge_0.10"]&rule["pWinScore_ge_0.55"]).sum()),
    }

    rank=d.copy()
    rank["predictedDecile"]=pd.qcut(rank.predictedNetR.rank(method="first"),10,labels=False,duplicates="drop")
    dec=[]
    for k,g in rank.groupby("predictedDecile"):
        dec.append({
          "decile":int(k),
          "rows":int(len(g)),
          "meanPredictedNetR":float(g.predictedNetR.mean()),
          "meanRealizedNetR":float(g.netR.mean()),
          "meanStressNetR":float(g.stressNetR.mean()),
          "positiveRate":float((g.netR>0).mean()),
          "meanPWinScore":float(g.pWinScore.mean())
        })

    score=d.copy()
    score["pWinDecile"]=pd.qcut(score.pWinScore.rank(method="first"),10,labels=False,duplicates="drop")
    pdec=[]
    for k,g in score.groupby("pWinDecile"):
        pdec.append({
          "decile":int(k),"rows":int(len(g)),
          "meanPWinScore":float(g.pWinScore.mean()),
          "meanRealizedNetR":float(g.netR.mean()),
          "positiveRate":float((g.netR>0).mean())
        })

    diag={
      "schema":"mgpt_ife1_diagnostic_v1",
      "generatedAtUtc":datetime.now(timezone.utc).isoformat(),
      "status":"CONTAMINATED_DEVELOPMENT_DIAGNOSTIC_ONLY",
      "immutableVerdict":"REJECT_IFE1_DEVELOPMENT_GATE",
      "rows":int(len(d)),
      "baselineMeanNetR":float(d.netR.mean()),
      "baselinePositiveRate":float((d.netR>0).mean()),
      "globalPWinScoreAuc":auc,
      "distributions":{
        "predictedNetR":qdict(d.predictedNetR),
        "ridgePredR":qdict(d.ridgePredR),
        "hgbPredR":qdict(d.hgbPredR),
        "pWinScore":qdict(d.pWinScore),
        "realizedNetR":qdict(d.netR)
      },
      "lockedConditionPassCounts":counts,
      "lockedConditionCombinations":combos,
      "byPredictedNetRDecile":dec,
      "byPWinScoreDecile":pdec,
      "byMarketFamily":grouped(d,"marketFamily"),
      "byTimeframe":grouped(d,"timeframe"),
      "interpretationBoundary":"This file is diagnosis only. Decile/subgroup outcomes are contaminated and cannot be promoted, thresholded or used to rescue IFE1. They may motivate a causally distinct, prelocked next-program objective on untouched evidence."
    }
    p=OUT/"MGPT_IFE1_DIAGNOSTIC_20260916.json"
    p.write_text(json.dumps(diag,indent=2,sort_keys=True)+"\n")
    (OUT/"MGPT_IFE1_DIAGNOSTIC_SHA256SUMS_20260916.txt").write_text(
      f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
    )
    print(json.dumps({
      "rows":diag["rows"],"globalPWinScoreAuc":auc,
      "counts":counts,"combos":combos,
      "topPredictedDecile":dec[-1] if dec else None
    },indent=2))

if __name__=="__main__":main()
