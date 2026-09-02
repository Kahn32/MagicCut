import numpy as np
from magiccut.adaptive import QueryCase
from magiccut.risk import area_under_risk_coverage,choose_threshold,evaluate_selective,risk_coverage_curve
def test_risk_coverage_hand_case():
    case=QueryCase("a",0,{i:np.array([float(i)]) for i in range(5)},frozenset({0,1,2}))
    probs=[{0:1.,1:.9,2:.6,3:.4,4:.1}]; curve=risk_coverage_curve(probs,[case])
    assert curve[0]["coverage"]==1 and curve[0]["risk"]==0
    chosen=choose_threshold(curve,.05); result=evaluate_selective(probs,[case],chosen["confidence_threshold"])
    assert result["risk"]==0 and result["coverage"]==1 and result["abstained"]==0
    assert 0<=area_under_risk_coverage(curve)<=1
