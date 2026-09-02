"""Deterministic simulated clarification for interactive MagicCut."""
from __future__ import annotations
from collections.abc import Callable,Mapping
from dataclasses import dataclass
import numpy as np
from magiccut.adaptive import QueryCase,group_metrics
from magiccut.graph import magiccut,unary_costs,solve_graph_cut,pair_features
from magiccut.uncertainty import min_marginals

@dataclass(frozen=True)
class FeedbackState:
    selected:frozenset[int]
    positive:frozenset[int]
    negative:frozenset[int]
    unlabeled:frozenset[int]

def infer(case,probabilities,params,positive:set[int],negative:set[int]):
    result,graph,pairwise=magiccut(case.values,probabilities,case.query,
        params["k"],params["pairwise_lambda"],params["sigma_factor"],params["decision_threshold"],
        hard_positive=positive,hard_negative=negative,edge_feature=params.get("edge_feature","combined"))
    return result,graph,pairwise

def simulate(case:QueryCase,probabilities:Mapping[int,float],params:Mapping[str,float],
             chooser:Callable,click_budget:int=3,seed:int=0):
    positive={case.query}; negative=set(); history=[]; rng=np.random.default_rng(seed)
    for click in range(click_budget+1):
        start=__import__('time').perf_counter(); result,graph,pairwise=infer(case,probabilities,params,positive,negative)
        inference_seconds=__import__('time').perf_counter()-start
        metrics=group_metrics(set(result.selected),set(case.target))
        history.append({"click":click,"selected":sorted(result.selected),"positive":sorted(positive),
                        "negative":sorted(negative),"inference_seconds":inference_seconds,**metrics})
        if click==click_budget or metrics["f1"]==1.: break
        unlabeled=set(case.values)-positive-negative
        if not unlabeled: break
        state=FeedbackState(result.selected,frozenset(positive),frozenset(negative),frozenset(unlabeled))
        start=__import__('time').perf_counter(); chosen=chooser(case,probabilities,params,state,graph,pairwise,rng)
        history[-1]["acquisition_seconds"]=__import__('time').perf_counter()-start
        if chosen is None: break
        node=int(chosen)
        if node not in unlabeled: raise AssertionError("Chooser returned labeled/unknown candidate")
        (positive if node in case.target else negative).add(node)
        history[-1]["asked_part"]=node; history[-1]["oracle_label"]=int(node in case.target)
    return history

def random_chooser(case,probabilities,params,state,graph,pairwise,rng):
    return int(rng.choice(sorted(state.unlabeled)))

def probability_uncertainty_chooser(case,probabilities,params,state,graph,pairwise,rng):
    return min(state.unlabeled,key=lambda n:(abs(probabilities[n]-.5),n))

def min_marginal_chooser(case,probabilities,params,state,graph,pairwise,rng):
    rows,_=min_marginals(unary_costs(probabilities,params["decision_threshold"]),pairwise,
        params["pairwise_lambda"],set(state.positive),set(state.negative))
    return min(state.unlabeled,key=lambda n:(rows[n]["gap"],n))

def expected_change_chooser(case,probabilities,params,state,graph,pairwise,rng):
    """Expected global label change under either possible answer; no oracle look-ahead."""
    unaries=unary_costs(probabilities,params["decision_threshold"]); current=set(state.selected); scored=[]
    for node in sorted(state.unlabeled):
        neg=set(solve_graph_cut(unaries,pairwise,params["pairwise_lambda"],set(state.positive),set(state.negative)|{node}).selected)
        pos=set(solve_graph_cut(unaries,pairwise,params["pairwise_lambda"],set(state.positive)|{node},set(state.negative)).selected)
        p=float(probabilities[node]); expected=(1-p)*len(current^neg)+p*len(current^pos)
        scored.append((expected,node))
    return max(scored,key=lambda item:(item[0],-item[1]))[1]

def _normalize(values):
    low=min(values.values());high=max(values.values())
    return {key:(value-low)/(high-low) if high>low else 0. for key,value in values.items()}

def make_influence_chooser(weights,uncertainty_method="graph_min_marginal"):
    wu,wd,wv=(float(x) for x in weights)
    def choose(case,probabilities,params,state,graph,pairwise,rng):
        if uncertainty_method=="graph_min_marginal":
            rows,_=min_marginals(unary_costs(probabilities,params["decision_threshold"]),pairwise,
                params["pairwise_lambda"],set(state.positive),set(state.negative))
            raw={n:-rows[n]["gap"] for n in state.unlabeled}
        elif uncertainty_method=="calibration_entropy":
            raw={n:-(probabilities[n]*np.log(max(probabilities[n],1e-12))+
                (1-probabilities[n])*np.log(max(1-probabilities[n],1e-12))) for n in state.unlabeled}
        else: raise ValueError(f"Unsupported influence uncertainty: {uncertainty_method}")
        uncertainty=_normalize(raw);degree={n:0. for n in state.unlabeled}
        for (a,b),weight in pairwise.items():
            if a in degree: degree[a]+=weight
            if b in degree: degree[b]+=weight
        degree=_normalize(degree);queried=set(state.positive)|set(state.negative)
        diversity=_normalize({n:min(pair_features(case.values[n],case.values[q])["combined"] for q in queried)
                              for n in state.unlabeled})
        return max(state.unlabeled,key=lambda n:(wu*uncertainty[n]+wd*degree[n]+wv*diversity[n],-n))
    return choose

def oracle_best_chooser(case,probabilities,params,state,graph,pairwise,rng):
    unaries=unary_costs(probabilities,params["decision_threshold"]);scored=[]
    for node in sorted(state.unlabeled):
        positive=set(state.positive);negative=set(state.negative)
        (positive if node in case.target else negative).add(node)
        selected=set(solve_graph_cut(unaries,pairwise,params["pairwise_lambda"],positive,negative).selected)
        scored.append((group_metrics(selected,set(case.target))["f1"],node))
    return max(scored,key=lambda item:(item[0],-item[1]))[1]

def restrict_chooser(base_chooser,label):
    def choose(case,probabilities,params,state,graph,pairwise,rng):
        allowed=frozenset(n for n in state.unlabeled if (n in case.target)==bool(label))
        if not allowed:return None
        return base_chooser(case,probabilities,params,FeedbackState(state.selected,state.positive,state.negative,allowed),graph,pairwise,rng)
    return choose

def summarize_histories(histories,click_budget=3):
    curves=[]
    for click in range(click_budget+1):
        values=[next((row["f1"] for row in reversed(h) if row["click"]<=click),h[-1]["f1"]) for h in histories]
        curves.append({"click":click,"mean_f1":float(np.mean(values)),"variance_f1":float(np.var(values)),
                       "std_f1":float(np.std(values)),"worst_f1":float(np.min(values)),
                       "perfect_fraction":float(np.mean(np.asarray(values)==1.)),
                       "f1_over_0_90_fraction":float(np.mean(np.asarray(values)>.9))})
    clicks=[]
    for h in histories:
        reached=next((row["click"] for row in h if row["f1"]>.9),click_budget+1); clicks.append(reached)
    questions=[row for h in histories for row in h if "asked_part" in row];magnitudes=[]
    for h in histories:
        for before,after in zip(h,h[1:]):magnitudes.append(after["f1"]-before["f1"])
    acquisition=[row.get("acquisition_seconds",0.) for row in questions]
    inference=[row["inference_seconds"] for h in histories for row in h]
    return {"curve":curves,"area_under_f1_click_curve":float(np.trapezoid([r["mean_f1"] for r in curves],[r["click"] for r in curves])),
            "mean_clicks_to_f1_over_0_90_censored":float(np.mean(clicks)),
            "positive_questions":sum(row["oracle_label"]==1 for row in questions),
            "negative_questions":sum(row["oracle_label"]==0 for row in questions),
            "mean_correction_delta_f1":float(np.mean(magnitudes)) if magnitudes else 0.,
            "mean_acquisition_ms":1000*float(np.mean(acquisition)) if acquisition else 0.,
            "mean_inference_ms":1000*float(np.mean(inference)) if inference else 0.,"histories":histories}
