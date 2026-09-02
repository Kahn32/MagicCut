from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Iterable
from magiccut.io import read_json

@dataclass(frozen=True)
class MaterialQuery:
    uid:str;source_path:Path;material_id:int|float|None;primary_query:int
    final_selection:tuple[int,...];original_selection:tuple[int,...]|None

def _int_tuple(value,field,path):
    if not isinstance(value,list) or not all(isinstance(x,int) for x in value):raise ValueError(f"{path}: {field} must be a list of integers")
    return tuple(value)
def load_query(path):
    source=Path(path);payload=read_json(source)
    if not {"primary_query","final_selection"}<=payload.keys():raise ValueError(f"{source}: missing required fields")
    primary=payload["primary_query"]
    if not isinstance(primary,int):raise ValueError(f"{source}: primary_query must be an integer")
    original=payload.get("original_selection")
    return MaterialQuery(source.parent.name,source,payload.get("material_id"),primary,
        _int_tuple(payload["final_selection"],"final_selection",source),None if original is None else _int_tuple(original,"original_selection",source))
def load_queries(root):return [load_query(path) for path in sorted(Path(root).glob("*/*.json"))]
def benchmark_summary(uids:Iterable[str],queries:Iterable[MaterialQuery]):
    uids=list(uids);queries=list(queries);counts=Counter(q.uid for q in queries);sizes=sorted(len(q.final_selection) for q in queries)
    if not sizes:raise ValueError("No benchmark queries found")
    return {"mesh_count":len(uids),"unique_uid_count":len(set(uids)),"query_count":len(queries),"labeled_mesh_count":len(counts),
        "queries_per_mesh_min":min(counts.values()),"queries_per_mesh_max":max(counts.values()),"target_size_min":sizes[0],
        "target_size_median":sizes[len(sizes)//2],"target_size_max":sizes[-1],
        "primary_query_in_final_count":sum(q.primary_query in q.final_selection for q in queries),
        "queries_with_original_selection":sum(q.original_selection is not None for q in queries),
        "queries_missing_material_id":sum(q.material_id is None for q in queries),
        "missing_label_uids":sorted(set(uids)-set(counts)),"unexpected_label_uids":sorted(set(counts)-set(uids))}
def normalize_dedup_metadata(payload:Any):
    if not isinstance(payload,dict):raise ValueError("Dedup metadata root must be an object")
    result={}
    for uid,value in payload.items():
        iterator=value["unique_ids"].items() if isinstance(value,dict) and "unique_ids" in value else value.items()
        mapping={}
        for representative,duplicates in iterator:
            if isinstance(duplicates,dict):duplicates=duplicates.get("duplicates",duplicates.get("part_ids"))
            if not isinstance(duplicates,list) or not all(isinstance(x,int) for x in duplicates):raise ValueError(f"Invalid duplicate list for {uid}/{representative}")
            mapping[int(representative)]=tuple(duplicates)
        result[str(uid)]=mapping
    return result
def representative_for_part(representatives,part_id):
    if part_id in representatives:return part_id
    matches=[rep for rep,duplicates in representatives.items() if part_id in duplicates]
    if len(matches)!=1:raise ValueError(f"Expected one representative for part {part_id}, found {len(matches)}")
    return matches[0]
