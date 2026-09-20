"""Correct-link calibration at the decisive moderate-sample cell."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from graph_generation import make_graph,geometry
from scaling_experiment import PopulationFit
from finite_sample import fit_dataset,ROOT

def main():
    t=time.perf_counter();g=make_graph(10,'complete');B,w,d,_,_=geometry(g)
    f=PopulationFit(B,w,d,.2);fit=f.run();b=fit['b'];v=.6/b*.2*w+.2**5*f.best_x[1:]
    q=(1+b*np.tanh((B@v)/2))/2;eta=(1-b)/2
    rng=np.random.default_rng(20260922);n=100000
    jobs=[(10,.2,n,r,rng.binomial(n,q)) for r in range(200)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fit_dataset,jobs))
    raw=pd.DataFrame(rows);raw['eta_true']=eta;raw['generation']='logistic_Q';raw['seed']=20260922
    out=ROOT/'results';raw.to_csv(out/'profile_control_raw.csv',index=False)
    cov=((raw.profile_lo<=eta)&(raw.profile_hi>=eta)).mean()
    result=dict(eta_true=eta,coverage=float(cov),coverage_count=int(round(200*cov)),replicates=200,
        mean_width=float((raw.profile_hi-raw.profile_lo).mean()),bias=float(raw.eta_hat.mean()-eta),
        failed_fits=int((~raw.success).sum()),failed_profiles=int(raw.profile_failures.sum()),seconds=time.perf_counter()-t)
    (out/'profile_control.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
