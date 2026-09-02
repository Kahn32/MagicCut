#!/usr/bin/env python3
"""Regression-check prompt 14 and execute prompts 15--20."""
from __future__ import annotations
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image,ImageDraw
from magiccut.adaptive import QueryCase,group_metrics,macro_rule
from magiccut.calibration import (calibration_metrics,evaluate_threshold,fit_calibrator,
                                  predict_case_probabilities,tune_probability_threshold)
from magiccut.data.metadata import load_queries,normalize_dedup_metadata
from magiccut.data.renders import safe_extract_tar
from magiccut.dedup import collapse,inverse_groups,canonical_groups
from magiccut.graph import (build_knn_graph,energy,graph_statistics,magiccut,solve_graph_cut,
                            unary_costs,weighted_edges)
from magiccut.inference import collect_render_triplets,encode_triplets
from magiccut.io import read_json,sha256_file,write_json
from magiccut.model import load_released_encoder
from magiccut.resources import CHECKPOINT_SHA256,DATASET,DEDUP_IMAGES_PREFIX,DEDUP_METADATA_PATH,LABELS_PREFIX
from magiccut.risk import (area_under_risk_coverage,choose_threshold,evaluate_selective,
                           risk_coverage_curve)

REVISION="52b0489beef04a81453944dba4d3c098fbaffe60"
VAL_UIDS=("ffc0e9978ea342739e7abd6abcb1a437","5b1ea87674ca4c5583d55129db284aad",
          "c2ca0c6777a94f53bf38a3cb440e619b","5b01e3e82b8742709a2380056ce1599b",
          "02c42cbb51c24963b9c99d5762547bf2","49d7158c62864ba68cc787ca39a4dfb3")
TEST_UIDS=("941102c765994f8bba0cb0c8a1098e01","4b94d886c40247d8a4da3d3358706e40",
           "01c5b846e7454e5b91cc49b44ac05103","aef54b99c2f7438882c06bdf318c401a")
EXPECTED_THRESHOLD=.3222533223833185; EXPECTED_TEST_F1=.7510416666666666

def cache_key(checkpoint,archive):
    h=hashlib.sha256(); h.update(sha256_file(checkpoint).encode()); h.update(sha256_file(archive).encode())
    h.update(b"white-rgba-bicubic-518-imagenet-small-medium-full-v1"); return h.hexdigest()

def make_cases(uids,queries,groups,embeddings):
    cases=[]
    for uid in uids:
        inverse=inverse_groups(groups[uid]); values=embeddings[uid]
        if set(values)!=set(groups[uid]): raise ValueError(f"Representatives mismatch {uid}")
        for item in [query for query in queries if query.uid==uid]:
            cases.append(QueryCase(uid,inverse[item.primary_query],values,frozenset(collapse(set(item.final_selection),groups[uid]))))
    return cases

def line_plot(rows,x_key,y_key,output,title):
    image=Image.new("RGB",(800,500),"white"); draw=ImageDraw.Draw(image); m=60; w=680; h=380
    draw.line((m,440,740,440),fill="black",width=2); draw.line((m,60,m,440),fill="black",width=2)
    points=[]
    for row in sorted(rows,key=lambda r:r[x_key]):
        x=m+float(row[x_key])*w; y=440-float(row[y_key])*h; points.append((x,y))
    if len(points)>1: draw.line(points,fill=(45,110,180),width=3)
    for x,y in points: draw.ellipse((x-3,y-3,x+3,y+3),fill=(45,110,180))
    draw.text((m,20),title,fill="black"); output.parent.mkdir(parents=True,exist_ok=True); image.save(output)

def edge_diagnostics(graph,target):
    pp=pn=nn=0
    for a,b in graph.edges:
        count=int(a in target)+int(b in target)
        if count==2: pp+=1
        elif count==1: pn+=1
        else: nn+=1
    touching=pp+pn
    return {"positive_positive":pp,"positive_negative":pn,"negative_negative":nn,
            "positive_edge_purity":pp/touching if touching else 1.,
            "positive_negative_contamination":pn/touching if touching else 0.}

def graph_figure(case,graph,output):
    image=Image.new("RGB",(700,700),"white"); draw=ImageDraw.Draw(image); ids=sorted(graph); n=len(ids); positions={}
    for i,node in enumerate(ids):
        angle=2*np.pi*i/max(1,n); positions[node]=(350+280*np.cos(angle),350+280*np.sin(angle))
    for a,b in graph.edges:
        color=(210,65,65) if ((a in case.target)!=(b in case.target)) else (175,175,175)
        draw.line((*positions[a],*positions[b]),fill=color,width=2)
    for node,(x,y) in positions.items():
        color=(55,160,90) if node in case.target else (125,125,125); radius=13 if node==case.query else 9
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=(245,180,35) if node==case.query else color,outline="black")
    draw.text((20,15),f"{case.uid} query {case.query}: orange query, green target, red cross-edge",fill="black")
    image.save(output)

def sweep_heatmap(rows,output):
    ks=(2,3,5,8); lambdas=(0.,.1,.25,.5,1.,2.); cell=90; margin=100
    image=Image.new("RGB",(margin+len(lambdas)*cell+20,margin+len(ks)*cell+20),"white"); draw=ImageDraw.Draw(image)
    for yi,k in enumerate(ks):
        for xi,lam in enumerate(lambdas):
            best=max(row["f1"] for row in rows if row["k"]==k and row["pairwise_lambda"]==lam)
            color=(int(255*(1-best)),int(210*best),90); x=margin+xi*cell; y=margin+yi*cell
            draw.rectangle((x,y,x+cell-2,y+cell-2),fill=color); draw.text((x+12,y+35),f"{best:.3f}",fill="black")
        draw.text((15,margin+yi*cell+35),f"k={k}",fill="black")
    for xi,lam in enumerate(lambdas): draw.text((margin+xi*cell+15,65),f"L={lam:g}",fill="black")
    draw.text((15,15),"Validation F1; each cell uses best sigma",fill="black"); image.save(output)

def toy_solver_audit():
    unaries={0:(2.,.1),1:(1.4,.2),2:(.2,1.2),3:(.1,1.5)}
    pairwise={(0,1):.8,(1,2):.5,(2,3):.7}; lam=.6; positive={0}; negative={3}
    hard=sum(sum(x) for x in unaries.values())+lam*sum(pairwise.values())+1
    constrained=dict(unaries); constrained[0]=(hard,unaries[0][1]); constrained[3]=(unaries[3][0],hard)
    assignments=[]
    for bits in itertools.product((0,1),repeat=4):
        assignment=dict(enumerate(bits)); assignments.append((energy(assignment,constrained,pairwise,lam),bits))
    manual=min(assignments); result=solve_graph_cut(unaries,pairwise,lam,positive,negative)
    solver_bits=tuple(int(i in result.selected) for i in range(4))
    if not np.isclose(manual[0],result.energy) or manual[1]!=solver_bits: raise AssertionError((manual,result))
    return {"assignments_enumerated":16,"manual_minimum_energy":manual[0],"manual_assignment":list(manual[1]),
            "solver_energy":result.energy,"solver_assignment":list(solver_bits),"match":True}

def evaluate_graph(cases,probabilities,decision_threshold,k,lam,sigma):
    rows=[]; selections=[]
    for case,probs in zip(cases,probabilities,strict=True):
        result,_,_=magiccut(case.values,probs,case.query,k,lam,sigma,decision_threshold)
        rows.append(group_metrics(set(result.selected),set(case.target))); selections.append(set(result.selected))
    metrics={name:float(np.mean([row[name] for row in rows])) for name in ("precision","recall","f1")}
    metrics["worst_query_f1"]=float(min(row["f1"] for row in rows)); return metrics,selections

def main():
    raw=Path("data/raw"); benchmark=raw/"benchmark"; checkpoint=raw/"checkpoints/checkpoint.pt"
    cache_root=Path("data/cache/graph_feasibility"); output=Path("reports/generated/graph_feasibility")
    cache_root.mkdir(parents=True,exist_ok=True); output.mkdir(parents=True,exist_ok=True)
    if sha256_file(checkpoint)!=CHECKPOINT_SHA256: raise ValueError("Checkpoint SHA mismatch")
    model,model_audit=load_released_encoder(checkpoint); queries=load_queries(benchmark/LABELS_PREFIX)
    groups=normalize_dedup_metadata(read_json(benchmark/DEDUP_METADATA_PATH)); embeddings={}; cold_seconds=0.; mesh_audit={}
    for uid in (*VAL_UIDS,*TEST_UIDS):
        remote=f"{DEDUP_IMAGES_PREFIX}{uid}.tar.gz"; archive=Path(hf_hub_download(DATASET.repo_id,remote,repo_type="dataset",revision=REVISION,local_dir=benchmark))
        key=cache_key(checkpoint,archive); cache=cache_root/f"{uid}-{key[:16]}.npz"; hit=cache.exists()
        paths=safe_extract_tar(archive,cache_root/uid); triplets=collect_render_triplets(paths)
        if hit: payload=np.load(cache); ids,x=payload["ids"],payload["x"]
        else:
            start=time.perf_counter(); ids,x,_=encode_triplets(model,triplets); cold_seconds+=time.perf_counter()-start
            np.savez_compressed(cache,ids=ids,x=x,cache_key=key)
        embeddings[uid]={int(part):value for part,value in zip(ids,x,strict=True)}
        mesh_audit[uid]={"cache_hit":hit,"cache_key":key,"archive_sha256":sha256_file(archive),"representatives":len(ids)}
    val_cases=make_cases(VAL_UIDS,queries,groups,embeddings)

    # Prompt 14 regression and Prompt 15 freeze happen before test cases exist.
    calibrator=fit_calibrator(val_cases); probability_threshold,val_group=tune_probability_threshold(calibrator,val_cases)
    if not np.isclose(probability_threshold,EXPECTED_THRESHOLD): raise AssertionError("Prompt 14 threshold regression")
    val_probs=predict_case_probabilities(calibrator,val_cases); val_curve=risk_coverage_curve(val_probs,val_cases)
    selective_choice=choose_threshold(val_curve,target_risk=.05); confidence_threshold=selective_choice["confidence_threshold"]
    val_selective=evaluate_selective(val_probs,val_cases,confidence_threshold)

    # Prompts 16--17 validation-only graph preparation and diagnostics.
    graph_stats={}; per_k={}
    val_scores=[group_metrics({part for part,p in probs.items() if p>=probability_threshold},set(case.target))["f1"]
                for case,probs in zip(val_cases,val_probs,strict=True)]
    easy_index=int(np.argmax(val_scores)); difficult_index=int(np.argmin(val_scores))
    for k in (2,3,5,8):
        diagnostics=[]
        for case in val_cases:
            graph=build_knn_graph(case.values,k); diagnostics.append(edge_diagnostics(graph,set(case.target)))
            graph_stats[f"{case.uid}/{case.query}/k{k}"]=graph_statistics(graph)
        per_k[str(k)]={name:float(np.mean([row[name] for row in diagnostics])) for name in
                       ("positive_edge_purity","positive_negative_contamination")}
    for name,index in (("easy",easy_index),("difficult",difficult_index)):
        graph_figure(val_cases[index],build_knn_graph(val_cases[index].values,3),output/f"{name}_graph.png")

    # Prompt 18 exact toy correctness.
    toy=toy_solver_audit()

    # Prompt 19 fixed default one-click inference.
    default_params={"k":3,"pairwise_lambda":.5,"sigma_factor":1.}
    default_val,default_val_selections=evaluate_graph(val_cases,val_probs,probability_threshold,3,.5,1.)

    # Prompt 20 validation-only graph sweep; tie-break toward simpler/weaker smoothing.
    sweep=[]
    for k,lam,sigma in itertools.product((2,3,5,8),(0.,.1,.25,.5,1.,2.),(.5,1.,2.)):
        metrics,_=evaluate_graph(val_cases,val_probs,probability_threshold,k,lam,sigma)
        sweep.append({"k":k,"pairwise_lambda":lam,"sigma_factor":sigma,**metrics})
    best=max(sweep,key=lambda row:(row["f1"],row["worst_query_f1"],-row["pairwise_lambda"],-row["k"],-row["sigma_factor"]))
    frozen_graph={key:best[key] for key in ("k","pairwise_lambda","sigma_factor")}
    sweep_heatmap(sweep,output/"validation_graph_sweep.png")

    # Only now construct and evaluate test cases once.
    test_cases=make_cases(TEST_UIDS,queries,groups,embeddings); assert set(VAL_UIDS).isdisjoint(TEST_UIDS)
    test_probs=predict_case_probabilities(calibrator,test_cases); test_group=evaluate_threshold(test_probs,test_cases,probability_threshold)
    if not np.isclose(test_group["f1"],EXPECTED_TEST_F1): raise AssertionError("Prompt 14 test regression")
    test_curve=risk_coverage_curve(test_probs,test_cases); test_selective=evaluate_selective(test_probs,test_cases,confidence_threshold)
    default_test,default_test_selections=evaluate_graph(test_cases,test_probs,probability_threshold,3,.5,1.)
    frozen_test,frozen_test_selections=evaluate_graph(test_cases,test_probs,probability_threshold,**{
        "k":int(best["k"]),"lam":float(best["pairwise_lambda"]),"sigma":float(best["sigma_factor"])})

    # Dedup expansion correctness for one-click output on every held-out query.
    expanded=[]
    for case,selection in zip(test_cases,default_test_selections,strict=True):
        expanded_parts={part for rep in selection for part in canonical_groups(groups[case.uid])[rep]}
        original_query=next(q for q in queries if q.uid==case.uid
            and inverse_groups(groups[case.uid])[q.primary_query]==case.query
            and collapse(set(q.final_selection),groups[case.uid])==set(case.target))
        metrics=group_metrics(expanded_parts,set(original_query.final_selection))
        expanded.append({"uid":case.uid,"query_representative":case.query,"selected_representatives":len(selection),
                         "expanded_parts":len(expanded_parts),"original_part_metrics":metrics})

    line_plot(val_curve,"coverage","risk",output/"validation_risk_coverage.png","Validation risk vs coverage")
    line_plot(test_curve,"coverage","risk",output/"test_risk_coverage.png","Held-out risk vs coverage")
    report={"scope":{"dataset_revision":REVISION,"validation_uids":list(VAL_UIDS),"test_uids":list(TEST_UIDS),
        "validation_queries":len(val_cases),"test_queries":len(test_cases),"surrogate_split":True},
        "integrity":{"model":model_audit,"checkpoint_sha256":CHECKPOINT_SHA256,"mesh_disjoint":True,
                     "test_constructed_after_freeze":True,"cache_hits":sum(m["cache_hit"] for m in mesh_audit.values())},
        "meshes":mesh_audit,"cold_encode_seconds_this_run":cold_seconds,
        "prompt_14_regression":{"probability_threshold":probability_threshold,"validation":val_group,"test":test_group,
                                "test_calibration":calibration_metrics(calibrator,test_cases)},
        "prompt_15":{"target_risk":.05,"frozen_confidence_threshold":confidence_threshold,
            "validation":val_selective|{"aurc":area_under_risk_coverage(val_curve)},
            "test":test_selective|{"aurc":area_under_risk_coverage(test_curve)}},
        "prompt_16":{"feature_families":["part","context","full_object"],"normalization":"per-family L2",
                     "graph_statistics":graph_stats},
        "prompt_17":{"easy":{"uid":val_cases[easy_index].uid,"query":val_cases[easy_index].query,"baseline_f1":val_scores[easy_index]},
                     "difficult":{"uid":val_cases[difficult_index].uid,"query":val_cases[difficult_index].query,"baseline_f1":val_scores[difficult_index]},
                     "k_diagnostics":per_k},
        "prompt_18":toy,
        "prompt_19":{"default_parameters":default_params,"validation":default_val,"test":default_test,
                     "held_out_expanded_results":expanded,
                     "expanded_original_macro":{name:float(np.mean([row["original_part_metrics"][name] for row in expanded]))
                                                for name in ("precision","recall","f1")}},
        "prompt_20":{"combinations":len(sweep),"selected_on_validation":frozen_graph,
                     "selected_validation_metrics":{key:best[key] for key in ("precision","recall","f1","worst_query_f1")},
                     "frozen_test_metrics":frozen_test,"sweep":sweep}}
    write_json(output/"report.json",report); write_json(output/"frozen_configuration.json",{
        "dataset_revision":REVISION,"calibration_probability_threshold":probability_threshold,
        "selective_confidence_threshold":confidence_threshold,"graph":frozen_graph})
    print(json.dumps({"prompt_14":report["prompt_14_regression"],"prompt_15":report["prompt_15"],
        "prompt_17":report["prompt_17"],"prompt_18":toy,"prompt_19":{"validation":default_val,"test":default_test},
        "prompt_20":{key:report["prompt_20"][key] for key in ("combinations","selected_on_validation","selected_validation_metrics","frozen_test_metrics")},
        "cold_encode_seconds":cold_seconds},indent=2))
if __name__=="__main__": main()
