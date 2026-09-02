#!/usr/bin/env python3
"""Canonical validation-only graph hyperparameter sweep from graph hyperparameter sweep."""
from __future__ import annotations
import itertools,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from magiccut.adaptive import group_metrics
from magiccut.calibration import fit_calibrator,predict_case_probabilities,tune_probability_threshold
from magiccut.graph import magiccut
from magiccut.io import write_json
from run_interaction_experiments import load_experiment,EXPECTED_THRESHOLD

def evaluate(cases,probabilities,config,threshold):
    rows=[];timings=[]
    for case,probs in zip(cases,probabilities,strict=True):
        started=time.perf_counter();result,_,_=magiccut(case.values,probs,case.query,**config,decision_threshold=threshold)
        timings.append(1000*(time.perf_counter()-started));rows.append(group_metrics(set(result.selected),set(case.target)))
    return {name:float(np.mean([row[name] for row in rows])) for name in ("precision","recall","f1")}|{
        "worst_query_f1":float(min(row["f1"] for row in rows)),"median_ms_per_query":float(np.median(timings)),
        "p95_ms_per_query":float(np.percentile(timings,95))}

def plot(rows,feature,path):
    selected=[row for row in rows if row["config"]["edge_feature"]==feature]
    image=Image.new("RGB",(900,520),"white");draw=ImageDraw.Draw(image);draw.text((35,18),f"Validation sweep: {feature} edges",fill="black")
    ordered=sorted(selected,key=lambda row:row["validation"]["f1"],reverse=True)[:30]
    for i,row in enumerate(ordered):
        x=50+i*27;y=470-int(380*row["validation"]["f1"]);draw.rectangle((x,y,x+18,470),fill=(50,120,190))
    path.parent.mkdir(parents=True,exist_ok=True);image.save(path)

def main():
    output=Path("reports/generated/graph_sweep");output.mkdir(parents=True,exist_ok=True)
    validation,test,_=load_experiment();calibrator=fit_calibrator(validation);threshold,_=tune_probability_threshold(calibrator,validation)
    if not np.isclose(threshold,EXPECTED_THRESHOLD):raise AssertionError("Calibration threshold drift")
    val_probs=predict_case_probabilities(calibrator,validation);test_probs=predict_case_probabilities(calibrator,test)
    sweep=[]
    for k,lam,temp,feature in itertools.product((5,10,20,50),(0.,.1,.25,.5,1.,2.),(.5,1.,2.),("part","context","combined")):
        config={"k":k,"pairwise_lambda":lam,"sigma_factor":temp,"edge_feature":feature}
        sweep.append({"config":config,"validation":evaluate(validation,val_probs,config,threshold)})
    winner=max(sweep,key=lambda row:(row["validation"]["f1"],row["validation"]["worst_query_f1"],
        row["config"]["k"],-row["config"]["pairwise_lambda"],-row["validation"]["p95_ms_per_query"]))
    frozen=winner["config"];heldout=evaluate(test,test_probs,frozen,threshold)
    for feature in ("part","context","combined"):plot(sweep,feature,output/f"validation_{feature}_edges.png")
    report={"protocol":{"selection_split":"validation only","heldout_used_for_selection":False,"configurations":len(sweep),
        "k_values":[5,10,20,50],"pairwise_lambdas":[0,.1,.25,.5,1,2],"affinity_temperatures":[.5,1,2],
        "edge_features":["part","context","combined"]},"sweep":sweep,"selected":winner,"heldout":heldout}
    write_json(output/"report.json",report);write_json(output/"frozen_graph_configuration.json",frozen)
    print(json.dumps({"configurations":len(sweep),"selected":winner,"heldout":heldout},indent=2))
if __name__=="__main__":main()
