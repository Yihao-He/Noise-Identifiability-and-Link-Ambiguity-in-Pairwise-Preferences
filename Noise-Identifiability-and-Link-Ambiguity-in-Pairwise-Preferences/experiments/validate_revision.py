"""Non-duplicative checks of the new geometry, numerical orders and output records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sympy as sp
from graph_generation import suite,geometry,connected
from finite_sample import robust_interval

ROOT=Path(__file__).resolve().parent

def main():
    B=sp.Matrix([[1,-1,0],[1,0,-1],[1,0,0],[0,1,-1],[0,0,1]])
    C=sp.Matrix([[1,-1,0,1,0],[0,1,-1,0,1]])
    d=B*sp.Matrix([3,2,1]);P=C.T*(C*C.T).inv()*C
    assert C*B==sp.zeros(2,3)
    r3=P*d.applyfunc(lambda x:x**3);r5=P*d.applyfunc(lambda x:x**5)
    residual=r5-r3*(r3.dot(r5)/r3.dot(r3))
    assert r3.dot(r3)==162 and residual.dot(residual)==400
    assert r3.dot(r5)/r3.dot(r3)==sp.Rational(95,9)
    gs=suite();assert len(gs)==13 and all(connected(g['k'],g['edges']) for g in gs)
    scaling=pd.read_csv(ROOT/'results/scaling.csv')
    assert len(scaling)==104 and scaling.success.all()
    cyclic=scaling[scaling.family!='tree'];small=cyclic[cyclic.window=='near_tie_audit']
    icoef=np.pi**2/576*small.G3
    jcoef=2*(.6*np.pi**2/30720)**2*small.G5
    iratio=small.information/(icoef*small.delta**6)
    jratio=small.kl/(jcoef*small.delta**10)
    assert max(abs(iratio-1))<.002
    assert max(abs(jratio-1))<.002
    tree=scaling[scaling.family=='tree'];assert (tree[['information','kl','G3','G5']]==0).all().all()
    # Deterministic coverage implication whenever the Hoeffding event occurs.
    rng=np.random.default_rng(8026)
    for _ in range(100):
        eta=rng.uniform(0,.5);n=10000;M=11
        true_m=(1-2*eta)*rng.uniform(-1,1,M);rad=np.sqrt(2*np.log(2*M/.05)/n)
        empirical=np.clip(true_m+rng.uniform(-rad,rad,M),-1,1)
        lo,hi=robust_interval((1+empirical)/2,n)
        assert lo<=eta<=hi+1e-14
    raw=pd.read_csv(ROOT/'results/finite_sample_raw.csv')
    assert len(raw)==2400 and raw.groupby(['delta','n']).size().eq(200).all()
    assert (raw.profile_lo<=raw.profile_hi).all()
    assert raw.success.all() and raw.profile_failures.sum()==0
    report=dict(graphs=13,population_cells=104,finite_datasets=2400,
        exact_sparse_G3=162,exact_sparse_G5=400,all_population_solvers_converged=True,
        max_information_coefficient_relative_error=float(max(abs(iratio-1))),
        max_kl_coefficient_relative_error=float(max(abs(jratio-1))),
        max_scaled_population_score=float(cyclic.relative_score.max()),
        finite_final_failed_fits=int((~raw.success).sum()),finite_final_profile_failures=int(raw.profile_failures.sum()),
        finite_boundary_mles=int(raw.utility_boundary.sum()),status='passed')
    (ROOT/'results/revision_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
