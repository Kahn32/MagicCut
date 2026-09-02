from collections.abc import Mapping,Sequence
def canonical_groups(groups:Mapping[int,Sequence[int]]):
    out={}; seen=set()
    for rep,raw in groups.items():
        members=tuple(sorted(set(raw)|{rep})); overlap=seen&set(members)
        if overlap: raise ValueError(f"Overlapping parts {sorted(overlap)}")
        seen.update(members); out[rep]=members
    return out
def inverse_groups(groups): return {part:rep for rep,members in canonical_groups(groups).items() for part in members}
def collapse(parts:set[int],groups):
    inverse=inverse_groups(groups); missing=parts-inverse.keys()
    if missing: raise ValueError(f"Missing parts {sorted(missing)}")
    return {inverse[p] for p in parts}
