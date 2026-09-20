import json,itertools
from pathlib import Path
import mpmath as mp
import numpy as np
import pandas as pd
import sympy as sp
mp.mp.dps=80
BASE=Path(__file__).resolve().parents[1];R=BASE/'results';R.mkdir(exist_ok=True)
def link(z,kind):return mp.tanh(z/2) if kind=='L' else mp.erf(mp.sqrt(mp.pi)*z/4)
def probs(a,u,kind):return [(1+a*link(u[i]-u[j],kind))/2 for i,j in itertools.combinations(range(len(u)),2)]
def kl(p,q):return mp.fsum(x*mp.log(x/y)+(1-x)*mp.log((1-x)/(1-y)) for x,y in zip(p,q))
def fitted(k,d):
    a=mp.mpf('.6');u=[(k-1-i)*d for i in range(k)];p=probs(a,u,'P');c=mp.sqrt(mp.pi)/2;b=a/c
    explicit=[c*x for x in u];qexp=probs(b,explicit,'L')
    if k==3:
        m=2*p[0]-1;M=2*p[1]-1;ba=mp.sqrt(m*m*M/(2*m-M));x=2*mp.atanh(m/ba);uu=[2*x,x,mp.mpf(0)]
        stationarity=mp.mpf(0)
    else:
        edges=list(itertools.combinations(range(k),2))
        def score(*theta):
            aa=theta[0];vv=list(theta[1:])+[mp.mpf(0)];out=[mp.mpf(0)]*k
            for (i,j),pp in zip(edges,p):
                t=mp.tanh((vv[i]-vv[j])/2);q=(1+aa*t)/2;rr=(q-pp)/(q*(1-q))
                out[0]+=rr*t/2;gg=rr*aa*(1-t*t)/4
                if i<k-1:out[i+1]+=gg
                if j<k-1:out[j+1]-=gg
            return tuple(out)
        root=mp.findroot(score,tuple([b]+explicit[:-1]),tol=mp.mpf('1e-65'),maxsteps=100)
        ba=root[0];uu=list(root[1:])+[mp.mpf(0)];stationarity=max(abs(v) for v in score(*root))
    q=probs(ba,uu,'L')
    return p,q,qexp,ba,uu,stationarity

def main():
    rows=[];mechanisms=[]
    ds=['0.005','0.0075','0.01','0.015','0.02','0.025','0.04','0.05','0.075','0.1','0.15','0.2','0.3','0.4']
    for k in [3,4]:
        for ds0 in ds:
            d=mp.mpf(ds0);p,q,qe,a,u,stationarity=fitted(k,d)
            rows.append(dict(k=k,delta=float(d),etaL=float((1-a)/2),explicit_max_probability_difference=float(max(abs(x-y) for x,y in zip(p,qe))),optimized_max_probability_difference=float(max(abs(x-y) for x,y in zip(p,q))),explicit_kl=float(kl(p,qe)),optimized_kl=float(max(0,kl(p,q))),stationarity=float(stationarity)))
            if ds0 in ['0.025','0.05','0.1','0.2']:
                n=int(mp.nint(100*d**-8));qp=p if k==3 else q
                kval=mp.mpf(0) if k==3 else kl(p,qp)
                coef=[mp.log(pp*(1-qq)/(qq*(1-pp))) for pp,qq in zip(p,qp)]
                mechanisms.append(dict(k=k,delta=float(d),n=n,etaP=.2,etaQ=float((1-a)/2),aQ=mp.nstr(a,75),uQ=[mp.nstr(x,75) for x in u],p=[mp.nstr(x,75) for x in p],q=[mp.nstr(x,75) for x in qp],llr_coef=[mp.nstr(x,75) for x in coef],nkl=mp.nstr(n*kval,75),tv_bound=float(min(1,mp.sqrt(n*kval/2))),power_bound=float(min(1,mp.mpf('.05')+mp.sqrt(n*kval/2))),honest_diameter_probability_lower=float(max(0,mp.mpf('.9')-mp.sqrt(n*kval/2))),n_delta6=float(n*d**6),n_delta10=float(n*d**10)))
            print('HIGH_PRECISION',k,ds0,flush=True)
    df=pd.DataFrame(rows);df.to_csv(R/'population.csv',index=False)
    x=df[(df.k==4)&(df.delta<=.025)];slopes={col:float(np.polyfit(np.log(x.delta),np.log(x[col]),1)[0]) for col in ['explicit_max_probability_difference','optimized_max_probability_difference','explicit_kl','optimized_kl']}
    v=sp.Matrix([3,2,1]);B=sp.Matrix([[1,-1,0],[1,0,-1],[1,0,0],[0,1,-1],[0,1,0],[0,0,1]]);w=B*v;w3=w.applyfunc(lambda x:x**3);w5=w.applyfunc(lambda x:x**5);A=B.row_join(w3);res=w5-A*(A.T*A).inv()*A.T*w5
    coefficient=2*(sp.Rational(3,5)*sp.pi**2/30720)**2*(res.T*res)[0]
    output={'eta_limit':float((1-mp.mpf('1.2')/mp.sqrt(mp.pi))/2),'slopes':slopes,'k4_fifth_residual':[str(x) for x in res],'k4_fifth_residual_norm2':str((res.T*res)[0]),'k4_optimized_kl_leading_coefficient':str(sp.simplify(coefficient)),'k4_optimized_kl_leading_coefficient_float':float(coefficient),'precision_decimal_digits':80}
    (R/'high_precision_summary.json').write_text(json.dumps(output,indent=2));(R/'mechanisms.json').write_text(json.dumps(mechanisms,indent=2));print(json.dumps(output,indent=2),flush=True)
if __name__=='__main__':main()
