import itertools
import numpy as np
from magiccut.graph import (build_knn_graph,energy,graph_statistics,pair_features,solve_graph_cut,
                            split_normalize,unary_costs)

def test_three_view_features_and_symmetric_knn():
    values={i:np.full(1152,float(i)) for i in range(4)}
    # Constant nonzero blocks normalize identically; shapes and graph invariants still hold.
    assert len(split_normalize(values[1]))==3 and set(pair_features(values[1],values[2]))=={"part","context","full_object","combined"}
    graph=build_knn_graph(values,2); stats=graph_statistics(graph)
    assert stats["nodes"]==4 and stats["self_edges"]==0
    assert all(graph.has_edge(b,a) for a,b in graph.edges)

def test_graph_cut_matches_all_assignments_and_constraints():
    unaries={0:(2.,.1),1:(1.4,.2),2:(.2,1.2),3:(.1,1.5)}; pairwise={(0,1):.8,(1,2):.5,(2,3):.7}; lam=.6
    hard=sum(sum(x) for x in unaries.values())+lam*sum(pairwise.values())+1
    constrained=dict(unaries); constrained[0]=(hard,.1); constrained[3]=(.1,hard)
    brute=min((energy(dict(enumerate(bits)),constrained,pairwise,lam),bits) for bits in itertools.product((0,1),repeat=4))
    result=solve_graph_cut(unaries,pairwise,lam,{0},{3}); bits=tuple(int(i in result.selected) for i in range(4))
    assert np.isclose(result.energy,brute[0]) and bits==brute[1] and 0 in result.selected and 3 not in result.selected

def test_zero_pairwise_matches_shifted_probability_threshold():
    probabilities={0:.1,1:.4,2:.9}; threshold=.3; result=solve_graph_cut(unary_costs(probabilities,threshold),{},0.)
    assert result.selected==frozenset({1,2})

def test_negative_weights_and_conflicts_rejected():
    import pytest
    with pytest.raises(ValueError): solve_graph_cut({0:(1.,1.),1:(1.,1.)},{(0,1):-.1},1.)
    with pytest.raises(ValueError): solve_graph_cut({0:(1.,1.)},{},1.,{0},{0})
