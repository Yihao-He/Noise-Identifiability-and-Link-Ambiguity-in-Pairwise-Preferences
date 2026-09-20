"""Moderate per-edge sample sizes; nominal logistic profiles vs honest link-robust bounds.

Counts are sampled exactly as binomials. No individual-comparison expansion,
no GPU, and no use of the generating noise in the fitted objective.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import least_squares,brentq
from scipy.special import erf
from threadpoolctl import threadpool_limits
from graph_generation import make_graph,geometry

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'code'/'original'))
from inference import residual_jac

SEED=20260921

def profile(y,n,B,bound=40.):
    nn=np.full(len(y),n,dtype=float)
    lower=np.r_[0,np.full(B.shape[1],-bound)]
    upper=np.r_[.5,np.full(B.shape[1],bound)]
    fits=[]
    for eta in (.02,.16,.32,.45):
        z=2*np.arctanh(np.clip((2*y-1)/(1-2*eta),-.995,.995))
        u=np.linalg.lstsq(B,z,rcond=None)[0]
        start=np.clip(np.r_[eta,u],lower+1e-8,upper-1e-8)
        f=least_squares(lambda x:residual_jac(x,y,nn,B,'L')[0],start,
            jac=lambda x:residual_jac(x,y,nn,B,'L')[1],bounds=(lower,upper),
            x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=500)
        fits.append(f)
    f=min(fits,key=lambda f:f.fun@f.fun)
    loss=float(f.fun@f.fun);eta=float(f.x[0]);u=f.x[1:]
    ok=bool(f.success);u_boundary=bool(max(abs(u))>bound-.01)
    cache={};profile_failures=0
    def objective(e):
        nonlocal profile_failures,u_boundary
        if e==.5:
            r=residual_jac(u,y,nn,B,'L',e)[0]
            return float(r@r)-loss-3.841458820694124
        if e in cache:return cache[e]
        r=least_squares(lambda v:residual_jac(v,y,nn,B,'L',e)[0],u,
            jac=lambda v:residual_jac(v,y,nn,B,'L',e)[1],bounds=(-bound,bound),
            ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=400)
        if not r.success:profile_failures+=1
        u_boundary=u_boundary or bool(max(abs(r.x))>bound-.01)
        cache[e]=float(r.fun@r.fun)-loss-3.841458820694124
        return cache[e]
    try:
        lo=0. if objective(0.)<=0 else brentq(objective,0.,eta,xtol=1e-8)
        hi=.5 if objective(.5)<=0 else brentq(objective,eta,.5,xtol=1e-8)
    except ValueError:
        lo,hi=0.,.5;ok=False
    return dict(eta_hat=eta,profile_lo=lo,profile_hi=hi,success=ok,
        profile_failures=profile_failures,utility_boundary=u_boundary,
        noise_boundary=eta<1e-6 or eta>.5-1e-6,
        start_loss_spread=float(max(s.fun@s.fun for s in fits)-loss))

def robust_interval(y,n,alpha=.05):
    # Simultaneous centered-probability Hoeffding box over every edge.
    radius=np.sqrt(2*np.log(2*len(y)/alpha)/n)
    lower_amplitude=max(0.,float(np.max(np.abs(2*y-1)))-radius)
    return 0.,.5*(1-lower_amplitude)

def logistic_triangle_interval(y,n,edges,triangle=(0,4,9),alpha=.05):
    # Original model-conditional Hoeffding interval, retained as a diagnostic.
    lookup={tuple(e):2*y[i]-1 for i,e in enumerate(edges)}
    i,j,k=triangle;x,y0,z=[lookup[e] for e in ((i,j),(j,k),(i,k))]
    r=np.sqrt(2*np.log(6/alpha)/n)
    xm,ym,zm=x-r,y0-r,z-r;xp,yp,zp=x+r,y0+r,z+r
    if min(xm,ym,zm)<=0 or zm<=max(xp,yp) or xm+ym<=zp:return 0.,.5
    H=lambda a,b,c:a*b*c/(a+b-c)
    low=max(0.,H(xp,yp,zm));high=min(1.,H(xm,ym,zp))
    if low>high:return 0.,.5
    return .5*(1-np.sqrt(high)),.5*(1-np.sqrt(low))

def summarize(raw):
    rows=[]
    for (k,delta,n),sub in raw.groupby(['k','delta','n']):
        R=len(sub)
        for method,lo,hi,estimate in [('Logistic profile','profile_lo','profile_hi','eta_hat'),
            ('Hoeffding link-robust','robust_lo','robust_hi',None),
            ('Hoeffding logistic-only','triangle_lo','triangle_hi',None)]:
            cover=(sub[lo]<=.2)&(sub[hi]>=.2);width=sub[hi]-sub[lo]
            # Bias of a set is undefined. Report midpoint bias explicitly for intervals.
            point=sub[estimate] if estimate else (sub[lo]+sub[hi])/2
            phat=float(cover.mean());z=1.959963984540054;den=1+z*z/R
            center=(phat+z*z/(2*R))/den
            half=z*np.sqrt(phat*(1-phat)/R+z*z/(4*R*R))/den
            rows.append(dict(k=k,delta=delta,n=n,total_comparisons=int(n*k*(k-1)/2),method=method,
                replicates=R,coverage=phat,coverage_count=int(cover.sum()),coverage_lo=center-half,
                coverage_hi=center+half,width=float(width.mean()),width_mcse=float(width.std(ddof=1)/np.sqrt(R)),
                bias=float(point.mean()-.2),bias_definition='MLE minus truth' if estimate else 'interval midpoint minus truth',
                bias_mcse=float(point.std(ddof=1)/np.sqrt(R))))
    return pd.DataFrame(rows)

def fit_dataset(job):
    k,delta,n,rep,counts=job
    with threadpool_limits(limits=1):
        g=make_graph(k,'complete');B,w,d,_,_=geometry(g);y=counts/n
        fit=profile(y,n,B);widened=False
        if fit['utility_boundary'] or not fit['success'] or fit['profile_failures']:
            fit=profile(y,n,B,bound=80.);widened=True
        rlo,rhi=robust_interval(y,n)
        tlo,thi=logistic_triangle_interval(y,n,g['edges'],triangle=(0,k//2-1,k-1))
        return dict(k=k,delta=delta,n=n,replicate=rep,seed=SEED,
            counts=json.dumps(counts.tolist()),**fit,robust_lo=rlo,robust_hi=rhi,
            triangle_lo=tlo,triangle_hi=thi,widened_utility_bound=widened)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replicates',type=int,default=200)
    ap.add_argument('--k',type=int,default=10);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    t=time.perf_counter();g=make_graph(args.k,'complete');B,w,d,_,_=geometry(g)
    rows=[];out=ROOT/'results';out.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for di,delta in enumerate((.05,.1,.2)):
            p=(1+.6*erf(np.sqrt(np.pi)*delta*d/4))/2
            for ni,n in enumerate((100,1000,10000,100000)):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,args.k,di,ni]))
                jobs=[(args.k,delta,n,rep,rng.binomial(n,p)) for rep in range(args.replicates)]
                for count,row in enumerate(pool.map(fit_dataset,jobs),1):
                    rows.append(row)
                    if count%50==0:print('PROGRESS',delta,n,count,flush=True)
                raw=pd.DataFrame(rows);raw.to_csv(out/'finite_sample_raw.csv',index=False)
                summarize(raw).to_csv(out/'finite_sample_summary.csv',index=False)
                print('FINITE',args.k,delta,n,'done; elapsed',round(time.perf_counter()-t,1),flush=True)
    (out/'finite_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t,seed=SEED,
        replicates_per_cell=args.replicates,workers=args.workers,failures=sum(not r['success'] for r in rows),
        profile_failures=sum(r['profile_failures'] for r in rows),
        utility_boundary=sum(r['utility_boundary'] for r in rows),
        widened_fits=sum(r['widened_utility_bound'] for r in rows)),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
