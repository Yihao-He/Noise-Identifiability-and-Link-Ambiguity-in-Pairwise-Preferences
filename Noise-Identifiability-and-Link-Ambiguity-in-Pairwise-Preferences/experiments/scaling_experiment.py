"""Population information and genuinely refitted KL, evaluated at 60 digits.

Refitting uses noise b=b0+delta^2 beta and free utilities
v=(a/b)delta*w+delta^5*t. This invertible reparameterization stabilizes
near-tie optimization; t has K-1 free coordinates. Residuals and derivatives
are evaluated with mpmath before conversion to double for the least-squares
step. The computed KL is never replaced by its asymptotic leading term.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time
import mpmath as mp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from graph_generation import suite,geometry

ROOT=Path(__file__).resolve().parent
mp.mp.dps=60
A=mp.mpf('0.6');B0=2*A/mp.sqrt(mp.pi)

def info(B,d,delta):
    z=[mp.mpf(str(delta))*int(x) for x in d]
    f=[mp.erf(mp.sqrt(mp.pi)*x/4) for x in z]
    fp=[mp.exp(-mp.pi*x*x/16)/2 for x in z]
    var=[(1-(A*y)**2)/4 for y in f]
    # Subtract the exactly representable linear nuisance direction first.
    y=np.array([float(-(f0-x*fp0)/mp.sqrt(v)) for x,f0,fp0,v in zip(z,f,fp,var)])
    D=np.array([float(A*fp0/(2*mp.sqrt(v))) for fp0,v in zip(fp,var)])[:,None]*B
    residual=y-D@np.linalg.lstsq(D,y,rcond=None)[0]
    return float(residual@residual),sum(float(f0*f0/v) for f0,v in zip(f,var))

class PopulationFit:
    def __init__(self,B,w,d,delta):
        self.B=B;self.w=w;self.d=d;self.dt=mp.mpf(str(delta));self.dt5=self.dt**5
        self.p=[(1+A*mp.erf(mp.sqrt(mp.pi)*self.dt*int(x)/4))/2 for x in d]
        self.edge_indices=[np.flatnonzero(row) for row in B]
        self.cached_x=None

    def evaluate(self,x):
        if self.cached_x is not None and np.array_equal(x,self.cached_x): return self.cached
        beta=mp.mpf(float(x[0]));b=B0+self.dt**2*beta
        t=[mp.mpf(float(v)) for v in x[1:]]
        r=[];jb=[];ju=[];kl=mp.mpf(0)
        for row,inds,de,p in zip(self.B,self.edge_indices,self.d,self.p):
            corr=mp.fsum(int(row[j])*t[j] for j in inds)
            z=A/b*self.dt*int(de)+self.dt5*corr
            h=mp.tanh(z/2);q=(1+b*h)/2;dif=q-p
            if abs(dif)<mp.mpf('1e-45'):
                dev=mp.mpf(0);rr=mp.mpf(0);dr=1/mp.sqrt(p*(1-p))
            else:
                dev=p*mp.log1p(dif/p)*(-1)+(1-p)*(-mp.log1p(-dif/(1-p)))
                dev=max(mp.mpf(0),dev)
                rr=mp.sign(dif)*mp.sqrt(2*dev)
                dr=dif/(q*(1-q)*rr)
            kl+=dev
            r.append(float(rr/self.dt5))
            qb=self.dt**2*(h-A*self.dt*int(de)*(1-h*h)/(2*b))/2
            jb.append(float(dr*qb/self.dt5))
            ju.append(float(dr*b*(1-h*h)/4))
        J=np.column_stack([jb,np.array(ju)[:,None]*self.B])
        self.cached_x=x.copy();self.cached=(np.array(r),J,float(kl),float(b))
        return self.cached

    def run(self):
        dt=float(self.dt);b0=float(B0)
        low=np.r_[(.3-b0)/dt**2,np.full(len(self.w),-np.inf)]
        high=np.r_[(.95-b0)/dt**2,np.full(len(self.w),np.inf)]
        # Geometry-based warm start minimizes the actual fifth-order residual.
        d=self.d;B=self.B
        r3=d**3-B@np.linalg.lstsq(B,d**3,rcond=None)[0]
        r5=d**5-B@np.linalg.lstsq(B,d**5,rcond=None)[0]
        a3prime=-float(A)*np.pi/(24*b0*b0)
        a5=-float(A)*np.pi**2/(7680*b0)
        beta=-a5/a3prime*(r3@r5)/(r3@r3)
        corr=np.linalg.lstsq(B,beta*a3prime*d**3+a5*d**5,rcond=None)[0]
        starts=[np.r_[beta,corr]]
        if dt*max(abs(d))>.5:
            # Additional noise starts for the pre-asymptotic cells.
            for b in (.4,b0,.85):
                m=np.array([float(2*p-1) for p in self.p])
                h=2*np.arctanh(np.clip(m/b,-.999,.999))
                v=np.linalg.lstsq(B,h,rcond=None)[0]
                starts.append(np.r_[(b-b0)/dt**2,(v-float(A)/b*dt*self.w)/dt**5])
        results=[]
        for x0 in starts:
            x0=np.minimum(np.maximum(x0,low+1e-10),high-1e-10)
            res=least_squares(lambda x:self.evaluate(x)[0],x0,
                jac=lambda x:self.evaluate(x)[1],bounds=(low,high),x_scale='jac',
                ftol=2e-13,xtol=2e-13,gtol=2e-11,max_nfev=200)
            results.append(res)
        res=min(results,key=lambda x:x.fun@x.fun)
        self.best_x=res.x.copy()
        r,J,kl,b=self.evaluate(res.x)
        grad=J.T@r
        relative_score=float(np.max(abs(grad))/(1+np.linalg.norm(J,axis=0).max()*np.linalg.norm(r)))
        return dict(kl=kl,b=b,eta_refit=(1-b)/2,success=bool(res.success),
                    nfev=res.nfev,relative_score=relative_score,
                    start_kl_spread=float(max(t.fun@t.fun for t in results)-res.fun@res.fun)*dt**10/2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--graphs',nargs='*');args=ap.parse_args()
    t=time.perf_counter();out=ROOT/'results';out.mkdir(exist_ok=True)
    graphs=suite();(out/'graphs.json').write_text(json.dumps(graphs,indent=2))
    requested=[.2,.1,.05,.025]
    # A separate fixed window with maximum edge gap <=0.049 for K<=50.
    audit=[.001,.0005,.00025,.000125]
    rows=[];stats=[]
    for g in graphs:
        if args.graphs and g['name'] not in args.graphs:continue
        B,w,d,G3,G5=geometry(g);k=g['k'];M=len(d)
        for window,deltas in [('requested',requested),('near_tie_audit',audit)]:
            for delta in deltas:
                I,known=info(B,d,delta)
                if g['family']=='tree':
                    I=0.;fit=dict(kl=0.,b=float(B0),eta_refit=(1-float(B0))/2,success=True,nfev=0,relative_score=0.,start_kl_spread=0.)
                else:fit=PopulationFit(B,w,d,delta).run()
                row=dict(graph=g['name'],k=k,M=M,family=g['family'],window=window,delta=delta,max_gap=delta*max(abs(d)),G3=G3,G5=G5,information=I,known_information=known,**fit)
                rows.append(row)
                print(g['name'],window,delta,'KL',fit['kl'],'score',fit['relative_score'],flush=True)
        st=dict(graph=g['name'],family=g['family'],k=k,M=M,cycle_rank=M-k+1,density=2*M/(k*(k-1)),G3=G3,G5=G5,seed=g['seed'],attempts=g['attempts'])
        for window in ('requested','near_tie_audit'):
            sub=[r for r in rows if r['graph']==g['name'] and r['window']==window]
            for key in ('information','kl'):
                st[window+'_'+key+'_slope']=float(np.polyfit(np.log([r['delta'] for r in sub]),np.log([r[key] for r in sub]),1)[0]) if G3>0 and (key!='kl' or G5>0) else None
        stats.append(st)
        pd.DataFrame(rows).to_csv(out/'scaling.csv',index=False)
        pd.DataFrame(stats).to_csv(out/'graph_summary.csv',index=False)
    (out/'scaling_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t,precision=mp.mp.dps,noise_domain_b=[.3,.95],graphs=len(stats)),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
