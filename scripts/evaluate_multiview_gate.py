#!/usr/bin/env python3
"""Gate optional multiview work without treating crop scales as camera views."""
from pathlib import Path
import json
from magiccut.data.renders import parse_render_name
from magiccut.io import read_json,write_json
def main():
    roots=list(Path("data/cache/graph_feasibility").glob("*/"));files=[p for root in roots for p in root.rglob("*.png")]
    parsed=[parse_render_name(path) for path in files];views=sorted({item.view for item in parsed});sizes=sorted({item.size for item in parsed})
    p24=read_json("outputs/interaction_experiments/report.json")["uncertainty_comparison"]["test"]["view_feature_disagreement"]
    report={"audited_mesh_directories":len(roots),"png_files":len(files),"camera_view_tokens":views,"crop_scale_tokens":sizes,
        "camera_views_per_part":1,"view_feature_disagreement_heldout":p24,
        "occlusion_gate_passed":False,"decision":"skip_optional_multiview_extension",
        "reason":"The three released images are small/medium/full context crops from one camera token, not independent camera views; multiview variance would be fabricated."}
    write_json("outputs/multiview_gate/report.json",report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
