"""Small deterministic CPU checks, distinct from historical simulations."""
from pathlib import Path
from itertools import combinations
import json, hashlib
import numpy as np
import pandas as pd
import sympy as s
import mpmath as mp
from reproduce_tables import conservative_interval
ROOT=Path(__file__).resolve().parents[1]

def main():
    (ROOT/'build').mkdir(exist_ok=True)
    a,b,z=s.symbols('a b z',positive=True)
    f=s.erf(s.sqrt(s.pi)*z/4)
    inv=s.series(2*s.atanh(a*f/b),z,0,7).removeO().expand()
    c3=s.simplify(inv.coeff(z,3));c5=s.simplify(inv.coeff(z,5))
    b0=2*a/s.sqrt(s.pi)
    assert s.simplify(c3.subs(b,b0))==0
    assert s.simplify(c5.subs(b,b0)*b0/4+a*s.pi**2/30720)==0
    assert s.simplify(s.diff(c3,b).subs(b,b0)+a*s.pi/(24*b0**2))==0
    B=s.zeros(6,3)
    for row,(i,j) in enumerate(combinations(range(4),2)):
        if i<3:B[row,i]=1
        if j<3:B[row,j]=-1
    d=B*s.Matrix([3,2,1]);d3=d.applyfunc(lambda x:x**3);d5=d.applyfunc(lambda x:x**5)
    P=B*(B.T*B).inv()*B.T;T=B.row_join(d3)
    r3=(s.eye(6)-P)*d3;r5=(s.eye(6)-T*(T.T*T).inv()*T.T)*d5
    assert list(r5)==[6,-12,6,18,-12,6]
    assert (r5.T*r5)[0]==720
    mp.mp.dps=70
    # Efficient information for eta with free vertex utilities, using weighted QR projection.
    deltas=[mp.mpf(str(x)) for x in [.005,.0075,.01,.015,.025,.05,.1,.2]]
    Bm=mp.matrix(B.tolist());dm=mp.matrix(list(d));out=[]
    for delta in deltas:
        zz=[delta*x for x in dm];aa=mp.mpf('.6')
        ff=[mp.erf(mp.sqrt(mp.pi)*x/4) for x in zz]
        fp=[mp.exp(-mp.pi*x*x/16)/2 for x in zz]
        pp=[(1+aa*x)/2 for x in ff]
        score=mp.matrix([-ff[i]/mp.sqrt(pp[i]*(1-pp[i])) for i in range(6)])
        X=mp.matrix([[aa*fp[i]*Bm[i,j]/(2*mp.sqrt(pp[i]*(1-pp[i]))) for j in range(3)] for i in range(6)])
        beta=mp.lu_solve(X.T*X,X.T*score);res=score-X*beta
        out.append(dict(delta=float(delta),known_information=float((score.T*score)[0]),
                        unknown_information=float((res.T*res)[0])))
    info=pd.DataFrame(out);info.to_csv(ROOT/'data/information_check.csv',index=False)
    use=info.delta<=.025
    slopes={k:float(np.polyfit(np.log(info.loc[use,'delta']),np.log(info.loc[use,k]),1)[0]) for k in ['known_information','unknown_information']}
    fdf=pd.read_csv(ROOT/'data/finite.csv')
    old=pd.read_csv(ROOT/'data/finite_summary.csv')
    recomputed=fdf.groupby(['k','delta','n','world']).mean(numeric_only=True).reset_index()
    merged=old.merge(recomputed,on=['k','delta','n','world'],suffixes=('_old','_new'))
    diffs={key:float(np.max(np.abs(merged[key+'_old']-merged[key+'_new']))) for key in ['L_eta','L_width','P_width','union_diameter','union_length']}
    assert max(diffs.values())<1e-12
    mechs=json.loads((ROOT/'data/mechanisms.json').read_text())
    pairs=[]
    lookup={(m['k'],m['delta']):m for m in mechs}
    regenerated_count_matches=0
    for row in fdf.itertuples():
        m=lookup[row.k,row.delta];world=0 if row.world=='probit_P' else 1
        rng=np.random.default_rng(np.random.SeedSequence([20260920,row.k,int(row.delta*100000),world,row.rep]))
        counts=rng.binomial(row.n,np.array(m['p'] if world==0 else m['q'],float))
        regenerated_count_matches+=int(np.array_equal(counts,np.array(json.loads(row.counts))))
    rounding=[]
    for m in mechs:
        assert abs(float(m['power_bound'])-(.05+np.sqrt(float(m['nkl'])/2)))<1e-12
        pairs.append({key:m[key] for key in ['k','delta','n','etaQ','nkl','tv_bound','power_bound']})
        pp=[mp.mpf(float(x)) for x in m['p']];qq=[mp.mpf(float(x)) for x in m['q']]
        kl=m['n']*mp.fsum(x*mp.log(x/y)+(1-x)*mp.log((1-x)/(1-y)) for x,y in zip(pp,qq))
        rounding.append(dict(k=m['k'],delta=m['delta'],rounded_probability_joint_kl=float(kl),
                            rounded_probability_power_bound=float(mp.mpf('.05')+mp.sqrt(kl/2))))
    edge=pd.read_csv(ROOT/'data/cache_edges.csv')
    assert len(edge)==192 and edge.id.nunique()==32
    assert np.max(np.abs(edge.probability_mean-(edge.forward_p_i+edge.reversed_mapped_p_i)/2))<1e-14
    logistic_averaged=1/(1+np.exp(-(edge.logit_forward+edge.logit_reverse_mapped)/2))
    assert np.max(np.abs(edge.logit_mean_probability-logistic_averaged))<1e-14
    cachefits=pd.read_csv(ROOT/'data/cache_fits.csv')
    mismatch={str(key):int(value) for key,value in cachefits.groupby(['aggregation','link']).mismatch.sum().items()}
    savedcache=json.loads((ROOT/'data/cache_summary.json').read_text())
    assert mismatch==savedcache['mismatch_counts']
    checks={'symbolic_inverse_coefficients':'pass','G3':str((r3.T*r3)[0]),'G5':720,
        'fifth_projection':list(map(int,r5)),'etaQ_limit':float((1-2*.6/np.sqrt(np.pi))/2),
        'information_slopes_delta_le_0025':slopes,'summary_max_differences':diffs,
        'independent_saved_datasets':len(fdf),'same_count_reanalyses':len(fdf),
        'L_failed':int((~fdf.L_success).sum()),'P_failed':int((~fdf.P_success).sum()),
        'L_boundary':int(fdf.L_boundary.sum()),'P_boundary':int(fdf.P_boundary.sum()),
        'specified_pairs':pairs,'new_model_calls':0,'new_monte_carlo_datasets':0,
        'new_deterministic_information_points':len(info),
        'regenerated_count_matches':regenerated_count_matches,'count_regeneration_total':len(fdf),
        'rounded_sampling_distributions':rounding,
        'cache_numeric_edges':len(edge),'cache_directed_outputs':2*len(edge),
        'cache_averaging_arithmetic':'pass','cache_mismatch_counts':mismatch,
        'cache_failed_fits_retained':int((~cachefits.success).sum())}
    (ROOT/'build/key_claim_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
