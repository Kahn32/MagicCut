from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from magiccut.adaptive import QueryCase,distances,group_metrics
FEATURE_NAMES=("raw_l1","distance_percentile","query_local_density","mesh_size","view_distance_std")
def local_scale(case,k=5):
    values=sorted(value for part,value in distances(case).items() if part!=case.query)
    return max(float(np.median(values[:min(k,len(values))])) if values else 1.,np.finfo(float).eps)
def candidate_feature_rows(cases:Sequence[QueryCase]):
    features=[]; keys=[]
    for ci,case in enumerate(cases):
        d=distances(case); ids=sorted(part for part in case.values if part!=case.query)
        ordered=sorted((d[p],p) for p in ids); ranks={p:i/max(1,len(ordered)-1) for i,(_,p) in enumerate(ordered)}
        qviews=case.values[case.query].reshape(3,-1); density=local_scale(case)
        for part in ids:
            pviews=case.values[part].reshape(3,-1); view_d=np.abs(qviews-pviews).sum(axis=1)
            features.append([d[part],ranks[part],density,len(case.values),float(np.std(view_d))]); keys.append((ci,part))
    return np.asarray(features,float),keys
def candidate_rows(cases):
    x,keys=candidate_feature_rows(cases); y=np.asarray([int(p in cases[i].target) for i,p in keys],int)
    return x,y,keys
def fit_calibrator(cases):
    x,y,_=candidate_rows(cases)
    if len(np.unique(y))!=2: raise ValueError("Need both classes")
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=0)); model.fit(x,y); return model
def predict_case_probabilities(model,cases):
    x,keys=candidate_feature_rows(cases); p=model.predict_proba(x)[:,1]; out=[{case.query:1.} for case in cases]
    for (ci,part),value in zip(keys,p,strict=True): out[ci][part]=float(value)
    return out
def evaluate_threshold(probabilities,cases,threshold):
    rows=[group_metrics({part for part,p in probs.items() if p>=threshold},set(case.target)) for probs,case in zip(probabilities,cases,strict=True)]
    return {name:float(np.mean([r[name] for r in rows])) for name in ("precision","recall","f1")}|{"worst_query_f1":float(min(r["f1"] for r in rows))}
def tune_probability_threshold(model,cases):
    probs=predict_case_probabilities(model,cases); candidates=sorted({p for row in probs for p in row.values()})
    scored=[(t,evaluate_threshold(probs,cases,t)) for t in candidates]
    return max(scored,key=lambda x:(x[1]["f1"],x[1]["worst_query_f1"],x[0]))
def calibration_metrics(model,cases,bins=10):
    x,y,_=candidate_rows(cases); p=model.predict_proba(x)[:,1]; edges=np.linspace(0,1,bins+1); rows=[]; ece=0.
    for i in range(bins):
        mask=(p>=edges[i])&(p<(edges[i+1]) if i<bins-1 else p<=edges[i+1])
        if not mask.any(): continue
        conf=float(p[mask].mean()); acc=float(y[mask].mean()); count=int(mask.sum()); ece+=count/len(y)*abs(conf-acc)
        rows.append({"lower":float(edges[i]),"upper":float(edges[i+1]),"count":count,"mean_probability":conf,"positive_rate":acc})
    return {"candidate_count":len(y),"positive_rate":float(y.mean()),"brier_score":float(np.mean((p-y)**2)),"expected_calibration_error":float(ece),"reliability_bins":rows}
