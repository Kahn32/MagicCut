"""Raw-geometry features and exact component mapping for the audited Objaverse mesh."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import trimesh

def _feature(meshes,scene_scale):
    area=np.asarray([max(float(m.area),1e-12) for m in meshes]);vertices=np.concatenate([m.vertices for m in meshes])
    centroid=np.average(np.stack([m.centroid for m in meshes]),axis=0,weights=area);bounds=np.stack([vertices.min(0),vertices.max(0)])
    radial=np.linalg.norm(vertices-centroid,axis=1)/scene_scale;hist,_=np.histogram(radial,bins=np.linspace(0,1.5,9));hist=hist/max(hist.sum(),1)
    return np.r_[area.sum()/(scene_scale**2),centroid/scene_scale,(bounds[1]-bounds[0])/scene_scale,hist].astype(np.float32)

def extract_audited_alfajor_features(path):
    scene=trimesh.load(path,force="scene",process=False);transformed={}
    for node in scene.graph.nodes_geometry:
        transform,geometry_name=scene.graph[node];mesh=scene.geometry[geometry_name].copy();mesh.apply_transform(transform);mesh.merge_vertices()
        transformed[geometry_name]=list(mesh.split(only_watertight=False))
    groups=defaultdict(list);alfajors=["Alfajor","Alfajor001","Alfajor002","Alfajor007","Alfajor005","Alfajor006","Alfajor004","Alfajor003"]
    for index,name in enumerate(alfajors):
        biscuit=transformed[f"{name}_Biscuits_0"];dulce=transformed[f"{name}_Dulce de Leche_0"];coconut=transformed[f"{name}_Coconuts_0"]
        if [len(biscuit),len(dulce),len(coconut)]!=[2,1,1941]:raise ValueError(f"Unexpected Alfajor topology for {name}")
        if any(len(coconut[i:i+3])!=3 for i in range(0,len(coconut),3)):raise ValueError("Coconut shells do not form triplets")
        groups[0].extend(biscuit);groups[2].extend(dulce);groups[3].extend(coconut)
    groups[650].extend(transformed["Plate_Plate_0"])
    coffee=transformed["Coffee_Espuma Cafe_0"];cup=transformed["Cup_Cups_0"]
    sugar=transformed["Sugar_Sugar_0"];sugar_cup=transformed["Sugar_Cup_Cups_0"]
    if [len(coffee),len(cup),len(sugar),len(sugar_cup)]!=[1,5,2,7]:raise ValueError("Unexpected coffee/sugar topology")
    groups[5201].extend(coffee);groups[5202].extend(cup);groups[5203].append(sugar[0]);groups[5204].append(sugar[1])
    groups[5205].extend(sugar_cup[:4]);groups[5206].extend(sugar_cup[4:])
    all_vertices=np.concatenate([piece.vertices for pieces in groups.values() for piece in pieces]);scale=max(np.linalg.norm(all_vertices.max(0)-all_vertices.min(0)),1e-12)
    features={rep:_feature(pieces,scale) for rep,pieces in groups.items()}
    return features,{"scene_geometries":len(scene.geometry),"mesh_shells_after_vertex_merge":sum(len(x) for x in groups.values()),
        "benchmark_components_reconstructed":5207,"representatives":len(features),"feature_dimension":15,"alfajor_shell_pattern":[2,1,1941],
        "mapping_representatives":sorted(features)}
