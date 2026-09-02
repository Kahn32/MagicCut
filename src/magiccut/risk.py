"""Selective prediction and risk--coverage evaluation."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from magiccut.adaptive import QueryCase

def risk_coverage_curve(probabilities:Sequence[dict[int,float]],cases:Sequence[QueryCase]):
    rows=[]
    for ci,(probs,case) in enumerate(zip(probabilities,cases,strict=True)):
        for part,p in probs.items():
            if part==case.query: continue
            rows.append((2*abs(p-.5),p>=.5,part in case.target,ci,part,p))
    thresholds=sorted({0.}|{row[0] for row in rows})
    curve=[]
    for threshold in thresholds:
        accepted=[row for row in rows if row[0]>=threshold]
        errors=sum(pred!=target for _,pred,target,*_ in accepted)
        curve.append({"confidence_threshold":threshold,"coverage":len(accepted)/len(rows),
                      "risk":errors/len(accepted) if accepted else 0.,"accepted":len(accepted),
                      "errors":errors})
    return curve

def choose_threshold(curve,target_risk=.05):
    feasible=[row for row in curve if row["accepted"] and row["risk"]<=target_risk]
    if feasible: return max(feasible,key=lambda row:(row["coverage"],-row["risk"]))
    return min((row for row in curve if row["accepted"]),key=lambda row:(row["risk"],-row["coverage"]))

def evaluate_selective(probabilities,cases,confidence_threshold):
    total=accepted=errors=positive=negative=0; clarification=[]
    for probs,case in zip(probabilities,cases,strict=True):
        abstained=0; candidates=0
        for part,p in probs.items():
            if part==case.query: continue
            candidates+=1; total+=1; confidence=2*abs(p-.5)
            if confidence<confidence_threshold: abstained+=1; continue
            accepted+=1; predicted=p>=.5; positive+=int(predicted); negative+=int(not predicted)
            errors+=int(predicted!=(part in case.target))
        if candidates and abstained:
            clarification.append({"uid":case.uid,"query":case.query,"abstention_fraction":abstained/candidates})
    return {"coverage":accepted/total,"risk":errors/accepted if accepted else 0.,"accepted":accepted,
            "confident_positive":positive,"confident_negative":negative,"abstained":total-accepted,
            "clarification_queries":clarification}

def area_under_risk_coverage(curve):
    ordered=sorted(curve,key=lambda row:row["coverage"])
    return float(np.trapezoid([r["risk"] for r in ordered],[r["coverage"] for r in ordered]))
