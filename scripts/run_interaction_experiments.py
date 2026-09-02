#!/usr/bin/env python3
"""Execute prompts 21--30 without changing the frozen prompt-20 protocol."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import networkx as nx
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image,ImageDraw
from magiccut.adaptive import QueryCase,distances,group_metrics
from magiccut.calibration import fit_calibrator,predict_case_probabilities,tune_probability_threshold
from magiccut.data.metadata import load_queries,normalize_dedup_metadata
from magiccut.data.renders import safe_extract_tar
from magiccut.dedup import collapse,inverse_groups
from magiccut.feedback import (make_influence_chooser,min_marginal_chooser,oracle_best_chooser,
                               probability_uncertainty_chooser,random_chooser,restrict_chooser,
                               simulate,summarize_histories)
from magiccut.graph import build_knn_graph,magiccut,unary_costs,weighted_edges
from magiccut.inference import collect_render_triplets,encode_triplets
from magiccut.io import read_json,sha256_file,write_json
from magiccut.model import load_released_encoder
from magiccut.resources import CHECKPOINT_SHA256,DATASET,DEDUP_IMAGES_PREFIX,DEDUP_METADATA_PATH,LABELS_PREFIX
from magiccut.uncertainty import (binary_entropy,error_detection_summary,min_marginals,
                                  neighbor_disagreement)

REVISION="52b0489beef04a81453944dba4d3c098fbaffe60"
VAL_UIDS=("ffc0e9978ea342739e7abd6abcb1a437","5b1ea87674ca4c5583d55129db284aad",
          "c2ca0c6777a94f53bf38a3cb440e619b","5b01e3e82b8742709a2380056ce1599b",
          "02c42cbb51c24963b9c99d5762547bf2","49d7158c62864ba68cc787ca39a4dfb3")
TEST_UIDS=("941102c765994f8bba0cb0c8a1098e01","4b94d886c40247d8a4da3d3358706e40",
           "01c5b846e7454e5b91cc49b44ac05103","aef54b99c2f7438882c06bdf318c401a")
FROZEN={"k":50,"pairwise_lambda":.25,"sigma_factor":.5,"edge_feature":"part"}
EXPECTED_THRESHOLD=.3222533223833185

def cache_key(checkpoint,archive):
    h=hashlib.sha256(); h.update(sha256_file(checkpoint).encode()); h.update(sha256_file(archive).encode())
    h.update(b"white-rgba-bicubic-518-imagenet-small-medium-full-v1"); return h.hexdigest()

def make_cases(uids,queries,groups,embeddings):
    cases=[]
    for uid in uids:
        inverse=inverse_groups(groups[uid])
        for item in (q for q in queries if q.uid==uid):
            cases.append(QueryCase(uid,inverse[item.primary_query],embeddings[uid],
                frozenset(collapse(set(item.final_selection),groups[uid]))))
    return cases

def load_experiment():
    raw=Path("data/raw"); benchmark=raw/"benchmark"; checkpoint=raw/"checkpoints/checkpoint.pt"
    if sha256_file(checkpoint)!=CHECKPOINT_SHA256: raise ValueError("Checkpoint SHA mismatch")
    model,_=load_released_encoder(checkpoint); queries=load_queries(benchmark/LABELS_PREFIX)
    groups=normalize_dedup_metadata(read_json(benchmark/DEDUP_METADATA_PATH)); embeddings={}; cache_hits=0
    cache_root=Path("data/cache/graph_feasibility")
    for uid in (*VAL_UIDS,*TEST_UIDS):
        archive=Path(hf_hub_download(DATASET.repo_id,f"{DEDUP_IMAGES_PREFIX}{uid}.tar.gz",repo_type="dataset",
                                     revision=REVISION,local_dir=benchmark))
        key=cache_key(checkpoint,archive); cache=cache_root/f"{uid}-{key[:16]}.npz"
        if cache.exists(): payload=np.load(cache); ids,x=payload["ids"],payload["x"]; cache_hits+=1
        else:
            paths=safe_extract_tar(archive,cache_root/uid); ids,x,_=encode_triplets(model,collect_render_triplets(paths))
            np.savez_compressed(cache,ids=ids,x=x,cache_key=key)
        embeddings[uid]={int(part):value for part,value in zip(ids,x,strict=True)}
    return make_cases(VAL_UIDS,queries,groups,embeddings),make_cases(TEST_UIDS,queries,groups,embeddings),cache_hits

def macro_from_selections(cases,selections):
    rows=[group_metrics(set(s),set(c.target)) for c,s in zip(cases,selections,strict=True)]
    return {k:float(np.mean([r[k] for r in rows])) for k in ("precision","recall","f1")}|{
        "worst_query_f1":float(min(r["f1"] for r in rows)),"per_query":rows}

def tune_distance_rule(cases,adaptive=False):
    all_scores=[]
    for case in cases:
        d=distances(case); scale=local_scale(case) if adaptive else 1.
        all_scores.extend(value/scale for part,value in d.items() if part!=case.query)
    best=None
    for threshold in sorted(set(all_scores)):
        selections=[]
        for case in cases:
            scale=local_scale(case) if adaptive else 1.; d=distances(case)
            selections.append({p for p,v in d.items() if v/scale<=threshold}|{case.query})
        metrics=macro_from_selections(cases,selections)
        candidate=(metrics["f1"],metrics["worst_query_f1"],-threshold,threshold,metrics)
        if best is None or candidate[:3]>best[:3]: best=candidate
    return best[3],best[4]

def local_scale(case,k=5):
    d=sorted(v for p,v in distances(case).items() if p!=case.query)
    return max(float(np.median(d[:min(k,len(d))])) if d else 1.,np.finfo(float).eps)

def selections_for(cases,probs,method,threshold=None):
    rows=[]
    for case,p in zip(cases,probs,strict=True):
        if method=="direct":
            d=distances(case); rows.append({n for n,v in d.items() if v<=threshold}|{case.query})
        elif method=="adaptive":
            d=distances(case); scale=local_scale(case); rows.append({n for n,v in d.items() if v/scale<=threshold}|{case.query})
        elif method=="calibrated": rows.append({n for n,v in p.items() if v>=threshold})
        elif method=="magiccut":
            result,_,_=magiccut(case.values,p,case.query,**FROZEN,decision_threshold=threshold); rows.append(set(result.selected))
        else: raise ValueError(method)
    return rows

def timed_selections(cases,probs,method,threshold,repeats=5):
    timings=[]; result=None
    for _ in range(repeats):
        for case,p in zip(cases,probs,strict=True):
            start=time.perf_counter(); result=selections_for([case],[p],method,threshold)[0]
            timings.append((time.perf_counter()-start)*1000)
    return {"median_ms_per_query":float(np.median(timings)),"p95_ms_per_query":float(np.percentile(timings,95))}

def method_evaluation(cases,probs,direct_t,adaptive_t,prob_t):
    result={}
    for method,threshold in (("direct",direct_t),("adaptive",adaptive_t),("calibrated",prob_t),("magiccut",prob_t)):
        selections=selections_for(cases,probs,method,threshold); metrics=macro_from_selections(cases,selections)
        result[method]={k:v for k,v in metrics.items() if k!="per_query"}|timed_selections(cases,probs,method,threshold)
        result[method]["selections"]=selections; result[method]["per_query"]=metrics["per_query"]
    return result

def failure_analysis(cases,probs,calibrated,graph):
    rows=[]
    for i,(case,p,base,cut) in enumerate(zip(cases,probs,calibrated["selections"],graph["selections"],strict=True)):
        b=group_metrics(set(base),set(case.target)); g=group_metrics(set(cut),set(case.target)); knn=build_knn_graph(case.values,FROZEN["k"],FROZEN["edge_feature"])
        touching=[(a,b) for a,b in knn.edges if a in case.target or b in case.target]
        cross=sum((a in case.target)!=(b in case.target) for a,b in touching)
        target_sub=knn.subgraph(case.target); disconnected=len(case.target)>1 and not nx.is_connected(target_sub)
        false_pos=set(cut)-set(case.target); repair_delta=None
        if false_pos:
            repaired,_,_=magiccut(case.values,p,case.query,**FROZEN,decision_threshold=EXPECTED_THRESHOLD,
                                  hard_negative={min(false_pos)})
            repair_delta=group_metrics(set(repaired.selected),set(case.target))["f1"]-g["f1"]
        rows.append({"case_index":i,"uid":case.uid,"query":case.query,"parts":len(case.values),
            "target_size":len(case.target),"target_fraction":len(case.target)/len(case.values),
            "calibrated_f1":b["f1"],"magiccut_f1":g["f1"],"delta_f1":g["f1"]-b["f1"],
            "cross_material_edges":cross,"target_touching_edges":len(touching),
            "cross_edge_fraction":cross/len(touching) if touching else 0.,
            "disconnected_positive_subgraph":disconnected,
            "extra_false_positives_vs_calibrated":len((set(cut)-set(case.target))-(set(base)-set(case.target))),
            "negative_seed_repair_delta_f1":repair_delta})
    return {"largest_improvements":sorted(rows,key=lambda r:(-r["delta_f1"],r["uid"],r["query"]))[:10],
            "largest_regressions":sorted(rows,key=lambda r:(r["delta_f1"],r["uid"],r["query"]))[:10],"all":rows}

def case_uncertainties(case,p,selection):
    timings={}; methods={m:{} for m in
        ("raw_score_margin","neighbour_vote_disagreement","view_feature_disagreement","calibration_entropy","graph_min_marginal")}
    start=time.perf_counter()
    for node in case.values:
        if node!=case.query: methods["raw_score_margin"][node]=-abs(p[node]-EXPECTED_THRESHOLD)
    timings["raw_score_margin"]=time.perf_counter()-start
    start=time.perf_counter(); graph=build_knn_graph(case.values,FROZEN["k"],FROZEN["edge_feature"])
    labels={n:n in selection for n in case.values}; disagreement=neighbor_disagreement(graph,labels)
    for node in case.values:
        if node!=case.query: methods["neighbour_vote_disagreement"][node]=disagreement[node]
    graph_setup_seconds=time.perf_counter()-start
    timings["neighbour_vote_disagreement"]=graph_setup_seconds
    start=time.perf_counter(); qviews=case.values[case.query].reshape(3,-1)
    for node in case.values:
        if node==case.query: continue
        nviews=case.values[node].reshape(3,-1)
        methods["view_feature_disagreement"][node]=float(np.std(np.abs(qviews-nviews).sum(axis=1)))
    timings["view_feature_disagreement"]=time.perf_counter()-start
    start=time.perf_counter()
    for node in case.values:
        if node!=case.query:
            methods["calibration_entropy"][node]=binary_entropy(p[node])
    timings["calibration_entropy"]=time.perf_counter()-start
    start=time.perf_counter(); pairwise=weighted_edges(graph,FROZEN["sigma_factor"],FROZEN["edge_feature"])
    margins,_=min_marginals(unary_costs(p,EXPECTED_THRESHOLD),pairwise,
        FROZEN["pairwise_lambda"],{case.query},exclude={case.query})
    for node in case.values:
        if node==case.query: continue
        methods["graph_min_marginal"][node]=-margins[node]["gap"]
    timings["graph_min_marginal"]=graph_setup_seconds+(time.perf_counter()-start)
    return methods,{"timings":timings,"min_marginal_solves":1+2*(len(case.values)-1),"marginals":margins}

def uncertainty_evaluation(cases,probs,selections):
    scores={}; errors=[]; runtime={}; ambiguity=[]
    for ci,(case,p,selection) in enumerate(zip(cases,probs,selections,strict=True)):
        methods,audit=case_uncertainties(case,p,selection)
        for method,row in methods.items(): scores.setdefault(method,[]).extend(row[n] for n in sorted(row))
        errors.extend(int((n in selection)!=(n in case.target)) for n in sorted(case.values) if n!=case.query)
        ambiguity.extend({"case_index":ci,"uid":case.uid,"query":case.query,"part":n,
            "gap":v["gap"],"mistake":int((n in selection)!=(n in case.target))} for n,v in audit["marginals"].items())
        for method,seconds in audit["timings"].items(): runtime.setdefault(method,[]).append(seconds)
    summary={m:error_detection_summary(v,errors)|{"mean_seconds_per_query":float(np.mean(runtime[m]))}
             for m,v in scores.items()}
    ambiguity.sort(key=lambda r:(r["gap"],r["uid"],r["query"],r["part"]))
    return summary,ambiguity

def evaluate_policy(cases,probs,params,chooser,seeds=(0,)):
    histories=[]
    for seed in seeds:
        for case,p in zip(cases,probs,strict=True): histories.append(simulate(case,p,params,chooser,3,seed))
    return summarize_histories(histories,3)

def curve_plot(policy_results,output):
    image=Image.new("RGB",(820,520),"white"); draw=ImageDraw.Draw(image); colors=[(120,120,120),(45,110,190),(30,150,90),(210,95,45)]
    y_min=.65; draw.line((70,450,760,450),fill="black",width=2); draw.line((70,50,70,450),fill="black",width=2)
    for click in range(4):
        x=70+click*230; draw.line((x,450,x,456),fill="black",width=2); draw.text((x-3,462),str(click),fill="black")
    for value in (.65,.75,.85,.95,1.):
        y=450-(value-y_min)/(1-y_min)*400; draw.line((64,y,70,y),fill="black",width=2); draw.text((25,y-6),f"{value:.2f}",fill="black")
    for (name,result),color in zip(policy_results.items(),colors,strict=True):
        points=[(70+r["click"]*230,450-(r["mean_f1"]-y_min)/(1-y_min)*400) for r in result["curve"]]
        draw.line(points,fill=color,width=4)
        for x,y in points: draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
        draw.text((560,205+25*list(policy_results).index(name)),name,fill=color)
    split="Validation" if "validation" in output.name else "Held-out test"
    draw.text((70,20),f"{split} mean F1 per clarification click",fill="black")
    draw.text((365,488),"Clarification clicks",fill="black"); output.parent.mkdir(parents=True,exist_ok=True); image.save(output)

def strip_internal(evaluation):
    return {name:{k:v for k,v in row.items() if k not in ("selections","per_query")} for name,row in evaluation.items()}

def without_histories(result): return {k:v for k,v in result.items() if k!="histories"}

def question_audit(result):
    repeated=0;total=0
    for history in result["histories"]:
        asked=[row["asked_part"] for row in history if "asked_part" in row];total+=len(asked);repeated+=len(asked)-len(set(asked))
    return {"questions":total,"repeated_or_redundant_questions":repeated}

def ambiguity_examples(result,limit=5):
    rows=[]
    for history in result["histories"]:
        if not history:continue
        rows.append({"initial_f1":history[0]["f1"],"final_f1":history[-1]["f1"],
            "delta_f1":history[-1]["f1"]-history[0]["f1"],
            "positive_answers":sum(r.get("oracle_label")==1 for r in history if "oracle_label" in r),
            "negative_answers":sum(r.get("oracle_label")==0 for r in history if "oracle_label" in r),
            "asked_parts":[r["asked_part"] for r in history if "asked_part" in r]})
    return sorted(rows,key=lambda r:(-r["negative_answers"],-r["delta_f1"]))[:limit]

def main():
    output=Path("reports/generated/interaction_experiments"); output.mkdir(parents=True,exist_ok=True)
    val_cases,test_cases,cache_hits=load_experiment(); calibrator=fit_calibrator(val_cases)
    probability_threshold,_=tune_probability_threshold(calibrator,val_cases)
    if not np.isclose(probability_threshold,EXPECTED_THRESHOLD): raise AssertionError("Prompt-14 threshold drift")
    val_probs=predict_case_probabilities(calibrator,val_cases); test_probs=predict_case_probabilities(calibrator,test_cases)

    # Prompt 21: tune non-learned comparators on validation only, then evaluate all four methods.
    direct_threshold,_=tune_distance_rule(val_cases,False); adaptive_threshold,_=tune_distance_rule(val_cases,True)
    validation=method_evaluation(val_cases,val_probs,direct_threshold,adaptive_threshold,probability_threshold)
    test=method_evaluation(test_cases,test_probs,direct_threshold,adaptive_threshold,probability_threshold)
    calibrated_test=test["calibrated"];graph_test=test["magiccut"]
    val_deltas=[g["f1"]-b["f1"] for b,g in zip(validation["calibrated"]["per_query"],validation["magiccut"]["per_query"],strict=True)]
    difficult=[i for i,row in enumerate(validation["calibrated"]["per_query"]) if row["f1"]<=np.quantile([r["f1"] for r in validation["calibrated"]["per_query"]],.25)]
    catastrophic_base=sum(r["precision"]<.5 for r in validation["calibrated"]["per_query"])
    catastrophic_graph=sum(r["precision"]<.5 for r in validation["magiccut"]["per_query"])

    # Prompt 22: exhaustive case diagnostics; top tens are views over all 19 queries.
    failures={"validation":failure_analysis(val_cases,val_probs,validation["calibrated"],validation["magiccut"]),
              "test":failure_analysis(test_cases,test_probs,test["calibrated"],test["magiccut"])}

    # Prompts 23--24: exact min-marginals and five-way error-detection comparison.
    val_uncertainty,val_ambiguity=uncertainty_evaluation(val_cases,val_probs,validation["magiccut"]["selections"])
    best_auc=max(row["error_detection_auroc"] for row in val_uncertainty.values() if row["error_detection_auroc"] is not None)
    eligible=[(name,row) for name,row in val_uncertainty.items() if row["error_detection_auroc"] is not None and
              row["error_detection_auroc"]>=best_auc-.025]
    selected_uncertainty=min(eligible,key=lambda item:(item[1]["mean_seconds_per_query"],-item[1]["error_detection_auroc"],item[0]))[0]
    test_uncertainty,test_ambiguity=uncertainty_evaluation(test_cases,test_probs,test["magiccut"]["selections"])

    graph_gate={"mean_validation_f1_improved":validation["magiccut"]["f1"]>validation["calibrated"]["f1"],
        "validation_improved_queries":sum(delta>0 for delta in val_deltas),"validation_regressed_queries":sum(delta<0 for delta in val_deltas),
        "difficult_case_mean_delta_f1":float(np.mean([val_deltas[i] for i in difficult])),
        "substantial_difficult_case_gain":float(np.mean([val_deltas[i] for i in difficult]))>=.05,
        "fewer_catastrophic_false_positive_cases":catastrophic_graph<catastrophic_base,
        "useful_structured_uncertainty":val_uncertainty["graph_min_marginal"]["error_detection_auroc"]>=.8,
        "validation_delta_f1":validation["magiccut"]["f1"]-validation["calibrated"]["f1"],
        "test_delta_f1":graph_test["f1"]-calibrated_test["f1"]}
    graph_gate["passed"]=any(graph_gate[k] for k in ("mean_validation_f1_improved","substantial_difficult_case_gain",
        "fewer_catastrophic_false_positive_cases","useful_structured_uncertainty"))
    graph_gate["verdict"]="retain_for_structured_uncertainty_and_interaction; do_not_claim_held_out_one_click_gain"

    params=FROZEN|{"decision_threshold":probability_threshold}
    random_val=evaluate_policy(val_cases,val_probs,params,random_chooser,tuple(range(20)))
    uncertainty_chooser=probability_uncertainty_chooser if selected_uncertainty=="calibration_entropy" else min_marginal_chooser
    uncertainty_val=evaluate_policy(val_cases,val_probs,params,uncertainty_chooser)
    raw_weights={(a/(a+b+c),b/(a+b+c),c/(a+b+c)) for a in (0.,.5,1.) for b in (0.,.5,1.) for c in (0.,.5,1.) if a+b+c}
    weight_sweep=[]
    for weights in sorted(raw_weights):
        result=evaluate_policy(val_cases,val_probs,params,make_influence_chooser(weights,selected_uncertainty))
        weight_sweep.append({"weights":{"uncertainty":weights[0],"degree":weights[1],"diversity":weights[2]},**without_histories(result)})
    best_weights_row=max(weight_sweep,key=lambda r:(r["area_under_f1_click_curve"],r["curve"][3]["mean_f1"],
        -r["mean_acquisition_ms"],r["weights"]["uncertainty"]))
    best_weights=tuple(best_weights_row["weights"][k] for k in ("uncertainty","degree","diversity"))
    influence_chooser=make_influence_chooser(best_weights,selected_uncertainty)
    influence_val=evaluate_policy(val_cases,val_probs,params,influence_chooser)
    positive_val=evaluate_policy(val_cases,val_probs,params,restrict_chooser(influence_chooser,True))
    negative_val=evaluate_policy(val_cases,val_probs,params,restrict_chooser(influence_chooser,False))
    oracle_val=evaluate_policy(val_cases,val_probs,params,oracle_best_chooser)
    deployable_val={"uncertainty_only":uncertainty_val,"influence_aware":influence_val}
    selected_policy=max(deployable_val,key=lambda name:(deployable_val[name]["area_under_f1_click_curve"],
        deployable_val[name]["curve"][3]["mean_f1"],-deployable_val[name]["mean_acquisition_ms"]))
    random_test=evaluate_policy(test_cases,test_probs,params,random_chooser,tuple(range(20)))
    uncertainty_test=evaluate_policy(test_cases,test_probs,params,uncertainty_chooser)
    influence_test=evaluate_policy(test_cases,test_probs,params,influence_chooser)
    oracle_test=evaluate_policy(test_cases,test_probs,params,oracle_best_chooser)
    positive_test=evaluate_policy(test_cases,test_probs,params,restrict_chooser(influence_chooser,True))
    negative_test=evaluate_policy(test_cases,test_probs,params,restrict_chooser(influence_chooser,False))
    val_policies={"random":random_val,"uncertainty_only":uncertainty_val,"influence_aware":influence_val,"oracle_best":oracle_val}
    test_policies={"random":random_test,"uncertainty_only":uncertainty_test,"influence_aware":influence_test,"oracle_best":oracle_test}
    selected_val=val_policies[selected_policy];selected_test=test_policies[selected_policy]
    active_gate={"selected_on_validation":selected_policy,
        "validation_auc_delta_vs_random":selected_val["area_under_f1_click_curve"]-random_val["area_under_f1_click_curve"],
        "test_auc_delta_vs_random":selected_test["area_under_f1_click_curve"]-random_test["area_under_f1_click_curve"],
        "validation_click3_delta_vs_random":selected_val["curve"][3]["mean_f1"]-random_val["curve"][3]["mean_f1"],
        "test_click3_delta_vs_random":selected_test["curve"][3]["mean_f1"]-random_test["curve"][3]["mean_f1"],
        "validation_delta_vs_uncertainty":selected_val["area_under_f1_click_curve"]-uncertainty_val["area_under_f1_click_curve"],
        "test_delta_vs_uncertainty":selected_test["area_under_f1_click_curve"]-uncertainty_test["area_under_f1_click_curve"]}
    active_gate["passed_gate_3"]=active_gate["validation_auc_delta_vs_random"]>0 and active_gate["test_auc_delta_vs_random"]>0
    curve_plot(val_policies,output/"validation_f1_per_click.png"); curve_plot(test_policies,output/"test_f1_per_click.png")
    report={"protocol":{"dataset_revision":REVISION,"checkpoint_sha256":CHECKPOINT_SHA256,
        "validation_uids":list(VAL_UIDS),"test_uids":list(TEST_UIDS),"cache_hits":cache_hits,
        "test_not_used_for_threshold_or_policy_selection":True,"click_budget":3,"random_seeds":20,
        "canonical_user_supplied_roadmap":True},
        "prompt_21":{"tuned_on_validation":{"direct_distance_threshold":direct_threshold,
            "adaptive_normalized_threshold":adaptive_threshold,"probability_threshold":probability_threshold,
            "graph":FROZEN},"validation":strip_internal(validation),"test":strip_internal(test),"graph_gate":graph_gate},
        "prompt_22":failures|{"decision":"pivot graph from one-click claim to uncertainty-aware interaction"},
        "prompt_23":{"validation_most_ambiguous":val_ambiguity[:20],"test_most_ambiguous":test_ambiguity[:20],
            "definition":"absolute energy gap between forced-positive and forced-negative optima",
            "validation":{"candidates":len(val_ambiguity),"cut_solves":len(val_cases)+2*len(val_ambiguity),
                "mean_seconds_per_query":val_uncertainty["graph_min_marginal"]["mean_seconds_per_query"],
                "mistakes_in_20_most_ambiguous":sum(r["mistake"] for r in val_ambiguity[:20])},
            "test":{"candidates":len(test_ambiguity),"cut_solves":len(test_cases)+2*len(test_ambiguity),
                "mean_seconds_per_query":test_uncertainty["graph_min_marginal"]["mean_seconds_per_query"],
                "mistakes_in_20_most_ambiguous":sum(r["mistake"] for r in test_ambiguity[:20])}},
        "prompt_24":{"selection_rule":"within 0.025 of best validation error AUROC, then fastest mean runtime",
            "selected":selected_uncertainty,"validation":val_uncertainty,"test":test_uncertainty},
        "prompt_25":{"simulator":"benchmark oracle; hard positive/negative seeds; rerun cut; stop at perfect F1 or 3 clicks",
            "positive_response":"candidate is added to hard-positive set","negative_response":"candidate is added to hard-negative set",
            "deterministic_history_count":len(val_cases)},
        "prompt_26":{"policy":"random clarification over currently unlabeled candidates","random_seeds":20,"validation":random_val,"test":random_test},
        "prompt_27":{"policy":f"uncertainty-only using {selected_uncertainty}","validation":uncertainty_val,"test":uncertainty_test,
            "validation_question_audit":question_audit(uncertainty_val),"test_question_audit":question_audit(uncertainty_test)},
        "prompt_28":{"policy":"normalized uncertainty + weighted degree + distance from queried parts","weight_grid_size":len(weight_sweep),
            "weight_sweep":weight_sweep,"selected_weights":best_weights_row["weights"],"validation":influence_val,"test":influence_test},
        "prompt_29":{"acquisition":"validation-tuned influence-aware","positive_only":{"validation":positive_val,"test":positive_test},
            "negative_only":{"validation":negative_val,"test":negative_test},
            "unrestricted_binary":{"validation":influence_val,"test":influence_test},
            "material_ambiguity_examples":ambiguity_examples(influence_test),
            "definition_note":"Feedback-type ablations are oracle-filtered to isolate positive versus negative evidence; unrestricted policy is deployable."},
        "prompt_30":{"validation_policy_selection":selected_policy,"claim_gate":active_gate,
            "metrics":{"validation":{k:without_histories(v) for k,v in val_policies.items()},
                       "test":{k:without_histories(v) for k,v in test_policies.items()}},
            "all_validation_curves":{k:v["curve"] for k,v in val_policies.items()},"all_test_curves":{k:v["curve"] for k,v in test_policies.items()}}}
    write_json(output/"report.json",report); write_json(output/"frozen_interactive_configuration.json",{
        "dataset_revision":REVISION,"probability_threshold":probability_threshold,"graph":FROZEN,
        "uncertainty_method":selected_uncertainty,"clarification_policy":selected_policy,
        "influence_weights":best_weights_row["weights"],"click_budget":3})
    print(json.dumps({"prompt_21":{"validation":strip_internal(validation),"test":strip_internal(test),"gate":graph_gate},
        "prompt_24":{"selected":selected_uncertainty,"validation":val_uncertainty,"test":test_uncertainty},
        "prompt_30":{"selected":selected_policy,"gate":active_gate,
            "validation_curves":{k:v["curve"] for k,v in val_policies.items()},
            "test_curves":{k:v["curve"] for k,v in test_policies.items()}}},indent=2))

if __name__=="__main__": main()
