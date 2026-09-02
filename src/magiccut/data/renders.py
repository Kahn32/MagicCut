from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tarfile

@dataclass(frozen=True)
class RenderName:
    uid:str;part_id:int;representative_id:int;view:str;size:str
def parse_render_name(path):
    stem=Path(path).stem;prefix,size=stem.rsplit("_",1);prefix,view=prefix.rsplit("_",1);prefix,rep=prefix.rsplit("_",1);uid,part=prefix.rsplit("_",1)
    if size not in {"small","medium","full"}:raise ValueError(f"Unknown render size: {size}")
    return RenderName(uid,int(part),int(rep),view,size)
def safe_extract_tar(archive,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);paths=[]
    with tarfile.open(archive,"r:gz") as handle:
        for member in handle.getmembers():
            target=(destination/member.name).resolve()
            if destination.resolve() not in target.parents and target!=destination.resolve():raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isfile():handle.extract(member,destination);paths.append(target)
    return paths
