#!/usr/bin/env python3
"""Resumable, integrity-checked preparation of all 100 deduplicated render archives."""
from __future__ import annotations
import argparse,json,shutil,tarfile,time
from pathlib import Path,PurePosixPath
from huggingface_hub import HfApi,hf_hub_download
from magiccut.data.renders import parse_render_name
from magiccut.io import read_json,sha256_file,write_json,write_json_atomic
from magiccut.resources import DATASET,DEDUP_IMAGES_PREFIX,DEDUP_METADATA_PATH,UIDS_PATH

REVISION="52b0489beef04a81453944dba4d3c098fbaffe60"
def official_files():
    info=HfApi().repo_info(DATASET.repo_id,repo_type="dataset",revision=REVISION,files_metadata=True)
    result={}
    for sibling in info.siblings:
        if sibling.rfilename.startswith(DEDUP_IMAGES_PREFIX) and sibling.rfilename.endswith(".tar.gz"):
            result[sibling.rfilename]={"size":int(sibling.size),"sha256":sibling.lfs.sha256}
    return result
def validate_archive(path,uid,expected_parts):
    names=[]
    with tarfile.open(path,"r:gz") as handle:
        for member in handle.getmembers():
            pure=PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:raise ValueError(f"Unsafe archive member {member.name}")
            if member.isfile():names.append(member.name)
    parsed=[parse_render_name(name) for name in names];parts={item.part_id for item in parsed};views={item.view for item in parsed}
    if len(names)!=3*expected_parts or len(parts)!=expected_parts or views!={"0"}:raise ValueError(f"Archive topology mismatch for {uid}")
    return {"files":len(names),"representatives":len(parts),"view_tokens":sorted(views),"safe_paths":True}
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--max-new",type=int,default=None);args=parser.parse_args()
    started=time.time();benchmark=Path("data/raw/benchmark");output=Path("outputs/benchmark_preparation");output.mkdir(parents=True,exist_ok=True)
    metadata=read_json(benchmark/DEDUP_METADATA_PATH);listed=set(read_json(benchmark/UIDS_PATH));rendered=set(metadata);official=official_files()
    expected_paths={uid:f"{DEDUP_IMAGES_PREFIX}{uid}.tar.gz" for uid in rendered};missing=set(expected_paths.values())-set(official)
    if missing:raise ValueError(f"Metadata archives absent upstream: {sorted(missing)}")
    manifest_path=output/"manifest.json";manifest=read_json(manifest_path) if manifest_path.exists() else {"dataset_revision":REVISION,"archives":{},"failures":[]}
    disk_before=shutil.disk_usage(".").free;new_count=0
    for index,uid in enumerate(sorted(rendered),1):
        remote=expected_paths[uid];expected=official[remote];local=benchmark/remote;row=manifest["archives"].get(uid,{})
        try:
            valid=local.exists() and local.stat().st_size==expected["size"] and sha256_file(local)==expected["sha256"]
            if not valid:
                if args.max_new is not None and new_count>=args.max_new:continue
                local=Path(hf_hub_download(DATASET.repo_id,remote,repo_type="dataset",revision=REVISION,local_dir=benchmark,
                    force_download=local.exists()));new_count+=1
            if local.stat().st_size!=expected["size"] or sha256_file(local)!=expected["sha256"]:raise ValueError("LFS size/SHA mismatch")
            audit=validate_archive(local,uid,int(metadata[uid]["unique_components"]))
            row={"status":"verified","remote_path":remote,"local_path":str(local),"size":expected["size"],"sha256":expected["sha256"],**audit}
            if uid in manifest["failures"]:manifest["failures"].remove(uid)
        except Exception as exc:
            row={"status":"failed","error":f"{type(exc).__name__}: {exc}"}
            if uid not in manifest["failures"]:manifest["failures"].append(uid)
        manifest["archives"][uid]=row;write_json_atomic(manifest_path,manifest);print(f"[{index:03d}/100] {uid} {row['status']}",flush=True)
    complete=sum(row.get("status")=="verified" for row in manifest["archives"].values())
    manifest.update({"metadata_meshes":len(rendered),"completed_archives":complete,"all_100_valid":complete==100 and not manifest["failures"],
        "estimated_download_bytes":sum(official[path]["size"] for path in expected_paths.values()),"disk_free_before":disk_before,
        "disk_free_after":shutil.disk_usage(".").free,"elapsed_seconds":time.time()-started,
        "uid_release_mismatch":{"listed_without_render":sorted(listed-rendered),"rendered_without_listing":sorted(rendered-listed)}})
    write_json_atomic(manifest_path,manifest);write_json(output/"report.json",{k:v for k,v in manifest.items() if k not in ("archives",)})
    print(json.dumps({k:v for k,v in manifest.items() if k!="archives"},indent=2))
    if args.max_new is None and not manifest["all_100_valid"]:raise SystemExit(1)
if __name__=="__main__":main()
