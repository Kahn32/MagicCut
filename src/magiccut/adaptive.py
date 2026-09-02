from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping,Callable,Sequence
import numpy as np
@dataclass(frozen=True)
class QueryCase: uid:str; query:int; values:Mapping[int,np.ndarray]; target:frozenset[int]
def l1(a,b): return float(np.abs(a-b).sum())
def distances(case): return {part:l1(case.values[case.query],value) for part,value in case.values.items()}
def group_metrics(predicted,target):
    tp=len(predicted&target); p=tp/len(predicted) if predicted else 0.; r=tp/len(target) if target else 0.
    f=2*p*r/(p+r) if p+r else 0.; return {"precision":p,"recall":r,"f1":f}
def macro_rule(cases:Sequence[QueryCase],selector:Callable):
    rows=[group_metrics(selector(case),set(case.target)) for case in cases]
    return {name:float(np.mean([row[name] for row in rows])) for name in ("precision","recall","f1")}|{"worst_query_f1":float(min(row["f1"] for row in rows))}
