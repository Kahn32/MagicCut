#!/usr/bin/env python3
from __future__ import annotations
import json,math
from pathlib import Path
import networkx as nx
import numpy as np
from magiccut.adaptive import group_metrics
from magiccut.calibration import fit_calibrator,predict_case_probabilities,tune_probability_threshold
from magiccut.geometry import extract_audited_alfajor_features
from magiccut.graph import pair_features,solve_graph_cut,unary_costs
from magiccut.io import sha256_file,write_json
from run_interaction_experiments import load_experiment,FROZEN,VAL_UIDS

UID="ffc0e9978ea342739e7abd6abcb1a437"
def pairwise(values,geometry,alpha):
    ids=sorted(values);visual={(a,b):pair_features(values[a],values[b])["part"] for i,a in enumerate(ids) for b in ids[i+1:]}
    geo={(a,b):float(np.abs(geometry[a]-geometry[b]).sum()) for i,a in enumerate(ids) for b in ids[i+1:]}
    scale=np.median(list(visual.values()))/max(np.median(list(geo.values())),1e-12)
    distance={edge:visual[edge]+alpha*scale*geo[edge] for edge in visual};sigma=max(FROZEN["sigma_factor"]*np.median(list(distance.values())),1e-12)
    return {edge:math.exp(-(d*d)/(2*sigma*sigma)) for edge,d in distance.items()}
def main():
    output=Path("outputs/geometry_ablation");output.mkdir(parents=True,exist_ok=True);mesh=Path(f"data/raw/objaverse/{UID}.glb")
    geometry,audit=extract_audited_alfajor_features(mesh);validation,_,_=load_experiment();calibrator=fit_calibrator(validation);threshold,_=tune_probability_threshold(calibrator,validation)
    cases=[case for case in validation if case.uid==UID];probs=predict_case_probabilities(calibrator,cases);sweep=[]
    for alpha in (0.,.1,.25,.5,1.,2.):
        rows=[]
        for case,p in zip(cases,probs,strict=True):
            selected=solve_graph_cut(unary_costs(p,threshold),pairwise(case.values,geometry,alpha),FROZEN["pairwise_lambda"],{case.query},set()).selected
            rows.append(group_metrics(set(selected),set(case.target)))
        sweep.append({"geometry_alpha":alpha,**{k:float(np.mean([r[k] for r in rows])) for k in ("precision","recall","f1")}})
    winner=max(sweep,key=lambda row:(row["f1"],-row["geometry_alpha"]))
    report={"mesh_uid":UID,"mesh_sha256":sha256_file(mesh),"mesh_audit":audit,"features":["normalized_surface_area","centroid_xyz","bbox_xyz","radial_histogram_8"],
        "validation_queries":len(cases),"sweep":sweep,"selected":winner,"decision":"retain" if winner["geometry_alpha"]>0 else "do_not_retain",
        "reason":"Validation-only selection; raw geometry did not improve the audited case."}
    write_json(output/"report.json",report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
