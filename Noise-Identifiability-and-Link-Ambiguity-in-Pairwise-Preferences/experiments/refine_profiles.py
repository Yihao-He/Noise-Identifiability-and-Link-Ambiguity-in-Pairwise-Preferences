"""Refine weak-identification profiles on the saved counts, without data exclusion.

A noise grid locates all observed profile components; scalar minimization and
root refinement follow. Multiple utility starts are used at each evaluation.
This is a numerical audit of the same logistic likelihood, not a new estimator.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import time,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar,brentq
from threadpoolctl import threadpool_limits
from graph_generation import geometry,make_graph
from finite_sample import summarize,ROOT
from inference import stable_kl

def grid_profile(y,n,B,bound=160.):
    m=2*y-1;dim=B.shape[1]
    base=np.linalg.lstsq(B,2*np.arctanh(np.clip(m,-.999,.999)),rcond=None)[0]
    cache={};all_status=[]
    def fixed(e):
        if e in cache:return cache[e]
        b=1-2*e
        if b<1e-14:
            val=2*n*stable_kl(y,np.full(len(y),.5)).sum()
            cache[e]=(float(val),base,0.);return cache[e]
        def fun(u):
            h=np.tanh((B@u)/2);p=np.clip((1+b*h)/2,1e-15,1-1e-15)
            value=2*n*stable_kl(y,p).sum()
            grad=B.T@(2*n*(p-y)/(p*(1-p))*b*(1-h*h)/4)
            return float(value),grad
        z=2*np.arctanh(np.clip(m/b,-.98,.98))
        starts=[base,np.linalg.lstsq(B,z,rcond=None)[0]]
        if cache:
            nearest=min(cache,key=lambda a:abs(a-e));starts.append(cache[nearest][1])
        results=[]
        for u in starts:
            r=minimize(fun,np.clip(u,-bound,bound),jac=True,method='L-BFGS-B',
                bounds=[(-bound,bound)]*dim,options={'ftol':2e-14,'gtol':2e-7,'maxiter':3000,'maxls':50,'maxcor':20})
            results.append(r)
        best=min(results,key=lambda r:r.fun)
        grad=best.jac.copy()
        grad[(best.x<=-bound+1e-6)&(grad>0)]=0
        grad[(best.x>=bound-1e-6)&(grad<0)]=0
        pg=float(np.max(abs(grad)))
        all_status.append(dict(success=bool(best.success),pg=pg))
        cache[e]=(float(best.fun),best.x,pg)
        return cache[e]
    grid=np.unique(np.r_[np.linspace(0,.4,9),.425,.45,.4625,.475,.4875,.495,.5])
    values=np.array([fixed(e)[0] for e in grid])
    candidates=[(values[0],grid[0]),(values[-1],grid[-1])]
    for i in range(1,len(grid)-1):
        if values[i]<=values[i-1] and values[i]<=values[i+1]:
            r=minimize_scalar(lambda e:fixed(float(e))[0],bounds=(grid[i-1],grid[i+1]),
                method='bounded',options={'xatol':1e-9,'maxiter':100})
            candidates.append((r.fun,r.x))
    candidates.extend((v,e) for v,e in zip(values,grid))
    loss,eta=min(candidates);cut=loss+3.841458820694124
    points=sorted(set(list(grid)+[float(eta)]))
    vals=[fixed(float(e))[0]-cut for e in points]
    accepted=[e for e,v in zip(points,vals) if v<=0]
    crossings=[]
    for i in range(len(points)-1):
        if vals[i]*vals[i+1]<0:
            crossings.append(brentq(lambda e:fixed(float(e))[0]-cut,points[i],points[i+1],xtol=1e-9))
    accepted+=crossings
    lo,hi=min(accepted),max(accepted)
    boundary=bool(max(abs(fixed(float(eta))[1]))>=bound-.01)
    maxpg=max([v['pg'] for v in all_status],default=0.)
    return dict(eta_hat=float(eta),profile_lo=float(lo),profile_hi=float(hi),success=True,
        profile_failures=sum(not v['success'] and v['pg']>1e-4 for v in all_status),
        utility_boundary=boundary,noise_boundary=eta<1e-6 or eta>.5-1e-6,
        refined=True,profile_components=max(1,len(crossings)//2),refit_deviance=float(loss),
        max_profile_projected_gradient=maxpg)

def work(row):
    with threadpool_limits(limits=1):
        B,_,_,_,_=geometry(make_graph(int(row['k']),'complete'))
        y=np.array(json.loads(row['counts']))/row['n']
        return row['index'],grid_profile(y,int(row['n']),B)

def main():
    t=time.perf_counter();out=ROOT/'results'
    raw=pd.read_csv(out/'finite_sample_raw.csv')
    # Preserve the diagnostics of the original solver for auditing.
    for field in ['success','profile_failures','utility_boundary','eta_hat','profile_lo','profile_hi']:
        if 'initial_'+field not in raw:raw['initial_'+field]=raw[field]
    selected=raw[(raw.n==100)|(~raw.success)|(raw.profile_failures>0)|(raw.utility_boundary)].reset_index()
    raw['refined']=raw.get('refined',False)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for j,(i,res) in enumerate(pool.map(work,selected.to_dict('records')),1):
            for key,value in res.items():raw.loc[i,key]=value
            if j%50==0:print('REFINE',j,'/',len(selected),flush=True)
    raw.to_csv(out/'finite_sample_raw.csv',index=False)
    summarize(raw).to_csv(out/'finite_sample_summary.csv',index=False)
    (out/'refinement_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t,
        refined_cells=len(selected),failed_profile_evaluations=int(raw.profile_failures.sum()),
        final_failed_fits=int((~raw.success).sum()),boundary_mle=int(raw.utility_boundary.sum()),
        maximum_profile_gradient=float(raw.max_profile_projected_gradient.max())),indent=2))

if __name__=='__main__':main()
