"""Candidate uncertainty measures for calibrated and graph-cut predictions."""
from __future__ import annotations
from collections.abc import Mapping
import math,time
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score
from magiccut.graph import solve_graph_cut

def binary_entropy(probability:float)->float:
    p=min(max(float(probability),1e-12),1-1e-12)
    return float(-(p*math.log(p)+(1-p)*math.log(1-p)))

def min_marginals(unaries:Mapping[int,tuple[float,float]],pairwise:Mapping[tuple[int,int],float],
                  pairwise_lambda:float,hard_positive:set[int]|None=None,
                  hard_negative:set[int]|None=None,exclude:set[int]|None=None):
    """Return exact constrained energies and absolute label gaps per candidate."""
    positive=set(hard_positive or ()); negative=set(hard_negative or ()); omitted=set(exclude or ())
    rows={}; start=time.perf_counter()
    base=solve_graph_cut(unaries,pairwise,pairwise_lambda,positive,negative)
    for node in sorted(set(unaries)-positive-negative-omitted):
        e0=solve_graph_cut(unaries,pairwise,pairwise_lambda,positive,negative|{node}).energy
        e1=solve_graph_cut(unaries,pairwise,pairwise_lambda,positive|{node},negative).energy
        rows[node]={"energy_negative":e0,"energy_positive":e1,"gap":abs(e1-e0),
                    "preferred_label":int(e1<e0),"base_label":int(node in base.selected)}
    return rows,{"base_energy":base.energy,"seconds":time.perf_counter()-start,
                 "solves":1+2*len(rows)}

def neighbor_disagreement(graph:nx.Graph,labels:Mapping[int,bool])->dict[int,float]:
    result={}
    for node in graph:
        neighbors=list(graph.neighbors(node))
        result[node]=(sum(labels[n]!=labels[node] for n in neighbors)/len(neighbors)) if neighbors else 0.
    return result

def error_detection_summary(scores,errors):
    scores=np.asarray(scores,float); errors=np.asarray(errors,int)
    auc=float(roc_auc_score(errors,scores)) if len(np.unique(errors))==2 else None
    order=np.argsort(scores,kind="stable") # accept least uncertain first
    cumulative=np.cumsum(errors[order]); risks=cumulative/np.arange(1,len(errors)+1)
    coverage=np.arange(1,len(errors)+1)/len(errors)
    aurc=float(np.trapezoid(risks,coverage)) if len(errors)>1 else float(risks[0])
    return {"candidate_count":int(len(errors)),"errors":int(errors.sum()),
            "error_detection_auroc":auc,"aurc":aurc}
