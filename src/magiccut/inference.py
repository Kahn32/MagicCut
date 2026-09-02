from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
import numpy as np,torch
from PIL import Image
from magiccut.data.renders import parse_render_name
from magiccut.model import INPUT_SIZE,VIEW_ORDER
MEAN=torch.tensor((.485,.456,.406)).view(3,1,1); STD=torch.tensor((.229,.224,.225)).view(3,1,1)
@dataclass(frozen=True)
class RenderTriplet: uid:str; part_id:int; representative_id:int; view:str; paths:tuple[Path,Path,Path]
def preprocess_render(path):
    with Image.open(path) as source:
        rgba=source.convert("RGBA"); white=Image.new("RGBA",rgba.size,"white")
        image=Image.alpha_composite(white,rgba).convert("RGB").resize((INPUT_SIZE,INPUT_SIZE),Image.Resampling.BICUBIC)
        array=np.asarray(image,dtype=np.float32).copy()/255
    return (torch.from_numpy(array).permute(2,0,1)-MEAN)/STD
def collect_render_triplets(paths:Iterable[str|Path]):
    grouped={}
    for raw in paths:
        path=Path(raw); item=parse_render_name(path); key=(item.uid,item.part_id,item.representative_id,item.view)
        if item.size in grouped.setdefault(key,{}): raise ValueError("Duplicate render")
        grouped[key][item.size]=path
    result=[]
    for (uid,part,rep,view),by_size in sorted(grouped.items()):
        if set(by_size)!=set(VIEW_ORDER): raise ValueError(f"Incomplete triplet {uid}/{part}")
        result.append(RenderTriplet(uid,part,rep,view,tuple(by_size[size] for size in VIEW_ORDER)))
    return result
def encode_triplets(model,triplets,batch_size=2):
    ids=[]; xs=[]; zs=[]
    with torch.inference_mode():
        for start in range(0,len(triplets),batch_size):
            batch=triplets[start:start+batch_size]
            views=torch.stack([torch.stack([preprocess_render(p) for p in item.paths]) for item in batch])
            x,z=model(views); ids.extend(item.part_id for item in batch); xs.append(x.numpy()); zs.append(z.numpy())
    return np.asarray(ids,dtype=np.int64),np.concatenate(xs),np.concatenate(zs)
