import numpy as np
from magiccut.adaptive import QueryCase
from magiccut.feedback import simulate,random_chooser,probability_uncertainty_chooser
from magiccut.graph import unary_costs
from magiccut.uncertainty import binary_entropy,min_marginals

def test_min_marginal_matches_independent_binary_unary_gap():
    probs={0:.9,1:.6,2:.2}; unaries=unary_costs(probs,.5)
    rows,audit=min_marginals(unaries,{},0.,{0})
    assert audit["solves"]==5 and rows[1]["preferred_label"]==1 and rows[2]["preferred_label"]==0
    assert np.isclose(rows[1]["gap"],abs(unaries[1][1]-unaries[1][0]))
    assert binary_entropy(.5)>binary_entropy(.9)

def test_feedback_is_deterministic_and_constraints_hold():
    values={i:np.r_[np.eye(4)[i%4].repeat(96),np.eye(4)[i%4].repeat(96),np.eye(4)[i%4].repeat(96)] for i in range(4)}
    case=QueryCase("toy",0,values,frozenset({0,1})); probs={0:1.,1:.45,2:.55,3:.1}
    params={"k":1,"pairwise_lambda":.2,"sigma_factor":1.,"decision_threshold":.5}
    a=simulate(case,probs,params,random_chooser,2,7); b=simulate(case,probs,params,random_chooser,2,7)
    nondeterministic={"inference_seconds","acquisition_seconds"}
    assert [{k:v for k,v in row.items() if k not in nondeterministic} for row in a]==[
        {k:v for k,v in row.items() if k not in nondeterministic} for row in b]
    for row in simulate(case,probs,params,probability_uncertainty_chooser,2):
        assert set(row["positive"])<=set(row["selected"])
        assert not (set(row["negative"])&set(row["selected"]))
