#!/usr/bin/env python3
"""Resumable exact-FP32 embedding extraction for the 100-mesh benchmark."""
from __future__ import annotations
import argparse,json,os,resource,tempfile,time
from pathlib import Path
import numpy as np
import torch
from magiccut.data.renders import safe_extract_tar
from magiccut.inference import collect_render_triplets,encode_triplets
from magiccut.io import read_json,sha256_file,write_json,write_json_atomic
from magiccut.model import load_released_encoder
from magiccut.resources import CHECKPOINT_SHA256,DEDUP_METADATA_PATH

def validate_cache(path,expected_ids):
    payload=np.load(path);ids=payload["ids"];x=payload["x"]
    if x.shape!=(len(expected_ids),1152) or set(map(int,ids))!=set(expected_ids) or not np.isfinite(x).all():raise ValueError(f"Invalid cache arrays {path}")
    return {"representatives":len(ids),"x_shape":list(x.shape),"cache_sha256":sha256_file(path)}
def prior_cache(uid,expected_ids):
    for path in Path("data/cache/graph_feasibility").glob(f"{uid}-*.npz"):
        try:return path,validate_cache(path,expected_ids)
        except Exception:pass
    return None,None
def atomic_npz(path,ids,x):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+".tmp.npz")
    np.savez_compressed(temporary,ids=ids,x=x,checkpoint_sha256=CHECKPOINT_SHA256);os.replace(temporary,path)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--max-new-meshes",type=int,default=None);args=parser.parse_args()
    started=time.time();output=Path("reports/generated/embeddings");output.mkdir(parents=True,exist_ok=True);manifest_path=output/"manifest.json"
    p33=read_json("reports/generated/benchmark_preparation/manifest.json");benchmark=Path("data/raw/benchmark");metadata=read_json(benchmark/DEDUP_METADATA_PATH)
    checkpoint=Path("data/raw/checkpoints/checkpoint.pt")
    if sha256_file(checkpoint)!=CHECKPOINT_SHA256:raise ValueError("Checkpoint integrity failure")
    torch.set_num_threads(min(9,os.cpu_count() or 1));model,model_audit=load_released_encoder(checkpoint);model.eval();batch_size=128
    if manifest_path.exists():manifest=read_json(manifest_path)
    else:manifest={"checkpoint_sha256":CHECKPOINT_SHA256,"dataset_revision":p33["dataset_revision"],"device":"cpu","dtype":"float32",
        "torch_threads":torch.get_num_threads(),"model":model_audit,"meshes":{},"failures":[]}
    manifest["batch_size_current"]=batch_size;manifest["batch_equivalence_audit"]={"tested_batch_sizes":[8,16,32,64,96,128],
        "largest_equivalence_mesh_representatives":183,"max_absolute_embedding_difference":0.0,"selected_batch_size":128,
        "batch_96_seconds_on_largest_audit":159.60094874899914,"batch_128_seconds_on_largest_audit":146.96625862900055,
        "batch_128_peak_rss_mib":11236.29296875,
        "selection_reason":"bit-identical FP32 output and lower tail runtime within the measured memory ceiling"}
    manifest["batch_policy"]={"representatives_below_128":64,"representatives_at_least_128":128,
        "reason":"batch 64 is faster on smaller CPU workloads; batch 128 is faster on the large tail; outputs are bit-identical"}
    work=Path("data/work");work.mkdir(parents=True,exist_ok=True);cache_root=Path("data/cache/full_embeddings")
    ordered=sorted(metadata,key=lambda uid:(int(metadata[uid]["unique_components"]),uid));new_count=0
    for index,uid in enumerate(ordered,1):
        expected_ids=sorted(map(int,metadata[uid]["unique_ids"]));existing=manifest["meshes"].get(uid)
        if existing and existing.get("status") in ("encoded","reused"):
            try:validate_cache(existing["cache_path"],expected_ids);print(f"[{index:03d}/100] {uid} cached {len(expected_ids)}",flush=True);continue
            except Exception:pass
        prior,audit=prior_cache(uid,expected_ids)
        if prior is not None:
            manifest["meshes"][uid]={"status":"reused","cache_path":str(prior),"source":"graph_feasibility",**audit};write_json_atomic(manifest_path,manifest)
            print(f"[{index:03d}/100] {uid} reused {len(expected_ids)}",flush=True);continue
        if args.max_new_meshes is not None and new_count>=args.max_new_meshes:continue
        row={"status":"running","representatives_expected":len(expected_ids)};manifest["meshes"][uid]=row;write_json_atomic(manifest_path,manifest)
        try:
            archive_info=p33["archives"][uid];archive=Path(archive_info["local_path"])
            if sha256_file(archive)!=archive_info["sha256"]:raise ValueError("Archive SHA drift")
            cache=cache_root/f"{uid}-{archive_info['sha256'][:12]}-{CHECKPOINT_SHA256[:12]}.npz";start=time.perf_counter()
            with tempfile.TemporaryDirectory(prefix=f"{uid}-",dir=work) as temporary:
                paths=safe_extract_tar(archive,temporary);triplets=collect_render_triplets(paths)
                if len(triplets)!=len(expected_ids) or {t.part_id for t in triplets}!=set(expected_ids):raise ValueError("Extracted triplets do not match metadata representatives")
                mesh_batch_size=64 if len(triplets)<128 else batch_size
                ids,x,_=encode_triplets(model,triplets,batch_size=mesh_batch_size)
            atomic_npz(cache,ids,x);audit=validate_cache(cache,expected_ids);row.update({"status":"encoded","cache_path":str(cache),
                "source":"full_fp32_extraction","archive_sha256":archive_info["sha256"],"batch_size_used":mesh_batch_size,
                "elapsed_seconds":time.perf_counter()-start,
                "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**audit});new_count+=1
            if uid in manifest["failures"]:manifest["failures"].remove(uid)
        except Exception as exc:
            row.update({"status":"failed","error":f"{type(exc).__name__}: {exc}"})
            if uid not in manifest["failures"]:manifest["failures"].append(uid)
        write_json_atomic(manifest_path,manifest);print(f"[{index:03d}/100] {uid} {row['status']} {len(expected_ids)} in {row.get('elapsed_seconds',0):.1f}s",flush=True)
    completed=sum(row.get("status") in ("encoded","reused") for row in manifest["meshes"].values());all_ids=sum(row.get("representatives",0) for row in manifest["meshes"].values() if row.get("status") in ("encoded","reused"))
    manifest.update({"last_run_seconds":time.time()-started,"completed_meshes":completed,"representatives_cached":all_ids,
        "all_100_complete":completed==100 and not manifest["failures"],"cache_bytes":sum(Path(row["cache_path"]).stat().st_size for row in manifest["meshes"].values() if row.get("status") in ("encoded","reused"))})
    write_json_atomic(manifest_path,manifest);write_json(output/"report.json",{k:v for k,v in manifest.items() if k!="meshes"});print(json.dumps({k:v for k,v in manifest.items() if k!="meshes"},indent=2))
    if args.max_new_meshes is None and not manifest["all_100_complete"]:raise SystemExit(1)
if __name__=="__main__":main()
