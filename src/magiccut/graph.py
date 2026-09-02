"""Three-view graph construction and graph-cut inference."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import math
import maxflow
import networkx as nx
import numpy as np

def split_normalize(value:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    if value.shape!=(1152,): raise ValueError(f"Expected 1152-D x, got {value.shape}")
    families=[]
    for feature in value.reshape(3,384):
        norm=float(np.linalg.norm(feature)); families.append(feature/(norm if norm else 1.))
    return tuple(families)

def pair_features(a:np.ndarray,b:np.ndarray)->dict[str,float]:
    av=split_normalize(a); bv=split_normalize(b); values=[float(np.abs(x-y).sum()) for x,y in zip(av,bv,strict=True)]
    return {"part":values[0],"context":values[1],"full_object":values[2],"combined":float(np.mean(values))}

EDGE_FEATURES=("part","context","full_object","combined")

def _build_knn_graph_reference(values:Mapping[int,np.ndarray],k:int,feature:str)->nx.Graph:
    ids=sorted(values); graph=nx.Graph(); graph.add_nodes_from(ids)
    for source in ids:
        ranked=sorted(((pair_features(values[source],values[target])[feature],target)
                       for target in ids if target!=source),key=lambda item:(item[0],item[1]))
        for _,target in ranked[:min(k,len(ranked))]:
            graph.add_edge(source,target,**pair_features(values[source],values[target]))
    return graph

def _build_knn_graph_vectorized(values:Mapping[int,np.ndarray],k:int,feature:str)->nx.Graph:
    from sklearn.neighbors import NearestNeighbors
    ids=sorted(values);raw=np.stack([values[node] for node in ids]);families=raw.reshape(len(ids),3,384)
    norms=np.linalg.norm(families,axis=2,keepdims=True);families=families/np.where(norms==0,1,norms)
    search=families[:,0] if feature=="part" else (families[:,1] if feature=="context" else
        families[:,2] if feature=="full_object" else families.reshape(len(ids),-1)/3.)
    distances,indices=NearestNeighbors(n_neighbors=min(k+1,len(ids)),metric="manhattan",algorithm="brute",
        n_jobs=-1).fit(search).kneighbors(search,return_distance=True)
    graph=nx.Graph();graph.add_nodes_from(ids);candidate_edges=set()
    for source,(row_d,row_i) in enumerate(zip(distances,indices,strict=True)):
        ranked=sorted(((float(d),ids[int(target)],int(target)) for d,target in zip(row_d,row_i,strict=True)
                       if int(target)!=source),key=lambda item:(item[0],item[1]))[:min(k,len(ids)-1)]
        candidate_edges.update((min(source,target),max(source,target)) for _,_,target in ranked)
    edge_array=np.asarray(sorted(candidate_edges),dtype=np.int64)
    for start in range(0,len(edge_array),4096):
        chunk=edge_array[start:start+4096]; ds=np.abs(families[chunk[:,0]]-families[chunk[:,1]]).sum(axis=2)
        for row,(left,right) in enumerate(chunk):
            values_=ds[row];graph.add_edge(ids[int(left)],ids[int(right)],part=float(values_[0]),
                context=float(values_[1]),full_object=float(values_[2]),combined=float(np.mean(values_)))
    return graph

def build_knn_graph(values:Mapping[int,np.ndarray],k:int,feature:str="combined")->nx.Graph:
    if k<1: raise ValueError("k must be positive")
    if feature not in EDGE_FEATURES: raise ValueError(f"Unknown edge feature: {feature}")
    graph=(_build_knn_graph_reference(values,k,feature) if len(values)<=128 else
           _build_knn_graph_vectorized(values,k,feature));graph.graph["feature"]=feature
    if any(a==b for a,b in graph.edges): raise AssertionError("Self edge")
    if not all(graph.has_edge(b,a) for a,b in graph.edges): raise AssertionError("Asymmetric graph")
    return graph

def graph_statistics(graph:nx.Graph)->dict[str,float|int|bool]:
    degrees=[degree for _,degree in graph.degree]
    return {"nodes":graph.number_of_nodes(),"edges":graph.number_of_edges(),
            "degree_min":min(degrees,default=0),"degree_mean":float(np.mean(degrees)) if degrees else 0.,
            "degree_max":max(degrees,default=0),"connected":nx.is_connected(graph) if graph else True,
            "self_edges":nx.number_of_selfloops(graph)}

def weighted_edges(graph:nx.Graph,sigma_factor:float=1.,feature:str|None=None):
    feature=feature or graph.graph.get("feature","combined")
    if feature not in EDGE_FEATURES: raise ValueError(f"Unknown edge feature: {feature}")
    distances=[data[feature] for *_,data in graph.edges(data=True)]
    median=float(np.median(distances)) if distances else 1.; sigma=max(sigma_factor*median,np.finfo(float).eps)
    return {(min(a,b),max(a,b)):math.exp(-(data[feature]**2)/(2*sigma**2)) for a,b,data in graph.edges(data=True)}

def shifted_probability(p:float,decision_threshold:float)->float:
    eps=1e-8; p=min(max(p,eps),1-eps); t=min(max(decision_threshold,eps),1-eps)
    shifted=math.log(p/(1-p))-math.log(t/(1-t)); return 1/(1+math.exp(-shifted))

def unary_costs(probabilities:Mapping[int,float],decision_threshold:float):
    eps=1e-8; result={}
    for node,p in probabilities.items():
        q=min(max(shifted_probability(p,decision_threshold),eps),1-eps)
        result[node]=(-math.log(1-q),-math.log(q)) # costs for labels 0 and 1
    return result

def energy(assignment:Mapping[int,int],unaries,pairwise,pairwise_lambda):
    value=sum(unaries[node][assignment[node]] for node in assignment)
    value+=pairwise_lambda*sum(weight for (a,b),weight in pairwise.items() if assignment[a]!=assignment[b])
    return float(value)

@dataclass(frozen=True)
class CutResult: selected:frozenset[int]; energy:float; cut_value:float

def solve_graph_cut(unaries:Mapping[int,tuple[float,float]],pairwise:Mapping[tuple[int,int],float],
                    pairwise_lambda:float,hard_positive:set[int]|None=None,hard_negative:set[int]|None=None)->CutResult:
    if pairwise_lambda<0 or any(weight<0 for weight in pairwise.values()): raise ValueError("Potts weights must be non-negative")
    positive=set(hard_positive or ()); negative=set(hard_negative or ())
    if positive&negative: raise ValueError("Conflicting hard constraints")
    nodes=set(unaries); source="__source__"; sink="__sink__"
    finite=sum(sum(costs) for costs in unaries.values())+pairwise_lambda*sum(pairwise.values()); hard=finite+1.
    # An undirected capacity graph represents each Potts term exactly once.
    # Using anti-parallel arcs is unnecessary here and can make floating-point
    # residual accounting disagree with the explicit binary energy audit.
    ordered=sorted(nodes); index={node:i for i,node in enumerate(ordered)}
    graph=maxflow.Graph[float](len(ordered),len(pairwise)); graph.add_nodes(len(ordered))
    constrained=dict(unaries)
    for node,(cost0,cost1) in unaries.items():
        if min(cost0,cost1)<0: raise ValueError("Unary costs must be non-negative")
        if node in positive: cost0=hard
        if node in negative: cost1=hard
        constrained[node]=(cost0,cost1)
        graph.add_tedge(index[node],float(cost0),float(cost1))
    for (a,b),weight in pairwise.items():
        if a not in nodes or b not in nodes or a==b: raise ValueError("Invalid pairwise edge")
        capacity=float(pairwise_lambda*weight); graph.add_edge(index[a],index[b],capacity,capacity)
    cut_value=float(graph.maxflow())
    selected=frozenset(node for node in ordered if graph.get_segment(index[node])==0)
    assignment={node:int(node in selected) for node in nodes}
    # Recompute the constrained energy represented by the cut for an exact audit.
    exact=energy(assignment,constrained,pairwise,pairwise_lambda)
    if not math.isclose(exact,cut_value,rel_tol=1e-8,abs_tol=1e-8): raise AssertionError((exact,cut_value))
    return CutResult(selected,exact,float(cut_value))

def magiccut(values,probabilities,query,k,pairwise_lambda,sigma_factor,decision_threshold,
             hard_negative=None,hard_positive=None,edge_feature="combined"):
    graph=build_knn_graph(values,k,edge_feature); pairwise=weighted_edges(graph,sigma_factor,edge_feature)
    positive={query}|set(hard_positive or ())
    result=solve_graph_cut(unary_costs(probabilities,decision_threshold),pairwise,pairwise_lambda,
                           hard_positive=positive,hard_negative=set(hard_negative or ()))
    return result,graph,pairwise
