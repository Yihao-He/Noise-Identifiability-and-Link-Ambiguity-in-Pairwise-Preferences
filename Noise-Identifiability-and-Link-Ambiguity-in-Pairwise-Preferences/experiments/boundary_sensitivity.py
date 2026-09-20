"""Check all saturation-boundary fits against doubled utility limits."""
import json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from graph_generation import geometry,make_graph
from finite_sample import ROOT
from refine_profiles import grid_profile

def work(row):
    with threadpool_limits(limits=1):
        B,*_=geometry(make_graph(int(row['k']),'complete'))
        fit=grid_profile(np.array(json.loads(row['counts']))/row['n'],int(row['n']),B,bound=320.)
        return dict(delta=row['delta'],n=row['n'],replicate=row['replicate'],
            eta_160=row['eta_hat'],eta_320=fit['eta_hat'],lo_160=row['profile_lo'],lo_320=fit['profile_lo'],
            hi_160=row['profile_hi'],hi_320=fit['profile_hi'],deviance_160=row['refit_deviance'],
            deviance_320=fit['refit_deviance'],failure_count=fit['profile_failures'])

def main():
    t=time.perf_counter();out=ROOT/'results';raw=pd.read_csv(out/'finite_sample_raw.csv')
    rows=raw[raw.utility_boundary].to_dict('records')
    with ProcessPoolExecutor(max_workers=8) as pool:result=pd.DataFrame(pool.map(work,rows))
    result.to_csv(out/'boundary_sensitivity.csv',index=False)
    cover160=(result.lo_160<=.2)&(result.hi_160>=.2);cover320=(result.lo_320<=.2)&(result.hi_320>=.2)
    summary=dict(checked=len(rows),max_eta_change=float(abs(result.eta_320-result.eta_160).max()),
        max_lower_change=float(abs(result.lo_320-result.lo_160).max()),max_upper_change=float(abs(result.hi_320-result.hi_160).max()),
        max_deviance_change=float(abs(result.deviance_320-result.deviance_160).max()),
        coverage_decisions_changed=int((cover160!=cover320).sum()),profile_failures=int(result.failure_count.sum()),seconds=time.perf_counter()-t)
    (out/'boundary_sensitivity.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':main()
