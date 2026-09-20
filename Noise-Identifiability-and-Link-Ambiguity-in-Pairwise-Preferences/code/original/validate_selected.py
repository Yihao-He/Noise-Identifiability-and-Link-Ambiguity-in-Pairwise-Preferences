"""Independent 70-digit checks of the most dispersed multistart fits per small-delta cell."""
import json,itertools
from pathlib import Path
import numpy as np,pandas as pd,mpmath as mp
from inference import graph
mp.mp.dps=70
BASE=Path(__file__).resolve().parents[1];R=BASE/'results'
df=pd.read_csv(R/'finite.csv');rows=[]
for keys,g in df[df.delta<=.05].groupby(['k','delta','world']):
    for kind in ['L','P']:
        row=g.loc[g[kind+'_spread_deviance'].idxmax()];k=int(row.k);n=int(row.n)
        edges=list(itertools.combinations(range(k),2));b,_=graph(k)
        y=[mp.mpf(v)/n for v in json.loads(row.counts)]
        eta=mp.mpf(str(row[kind+'_eta']));a=1-2*eta
        def lf(z):return mp.tanh(z/2) if kind=='L' else mp.erf(mp.sqrt(mp.pi)*z/4)
        def ld(z):return (1-mp.tanh(z/2)**2)/2 if kind=='L' else mp.exp(-mp.pi*z*z/16)/2
        zz=[2*mp.atanh((2*v-1)/a) if kind=='L' else 4/mp.sqrt(mp.pi)*mp.erfinv((2*v-1)/a) for v in y]
        u=np.linalg.lstsq(b,np.array(zz,float),rcond=None)[0]
        def score(a,v,free=True):
            out=[mp.mpf(0)]*(k if free else k-1)
            for (i,j),yy in zip(edges,y):
                z=v[i]-v[j];f=lf(z);p=(1+a*f)/2;r=(p-yy)/(p*(1-p));gg=r*a*ld(z)/2
                if free:out[0]+=r*f/2
                if i<k-1:out[i+int(free)]+=gg
                if j<k-1:out[j+int(free)]-=gg
            return tuple(out)
        root=mp.findroot(lambda *t:score(t[0],list(t[1:])+[mp.mpf(0)]),tuple([a]+list(u)),tol=mp.mpf('1e-55'),maxsteps=100)
        def dev(aa,v):
            p=[(1+aa*lf(v[i]-v[j]))/2 for i,j in edges]
            return 2*n*mp.fsum(yy*mp.log(yy/pp)+(1-yy)*mp.log((1-yy)/(1-pp)) for yy,pp in zip(y,p))
        vv=list(root[1:])+[mp.mpf(0)];loss=dev(root[0],vv);errs=[]
        for end in ['lo','hi']:
            aa=1-2*mp.mpf(str(row[kind+'_'+end]))
            rr=mp.findroot(lambda *v:score(aa,list(v)+[mp.mpf(0)],False),tuple(vv[:-1]),tol=mp.mpf('1e-55'),maxsteps=100)
            errs.append(float(abs(dev(aa,list(rr)+[mp.mpf(0)])-loss-mp.mpf('3.841458820694124'))))
        rows.append(dict(k=k,delta=row.delta,world=row.world,rep=int(row.rep),link=kind,eta_difference=float(abs((1-root[0])/2-eta)),deviance_difference=float(abs(loss-mp.mpf(str(row[kind+'_deviance'])))),profile_cutoff_max_error=max(errs)))
out={'selection':'largest multistart deviance spread in each k/delta/world/link cell with delta <= .05','precision':70,'checks':rows}
out['max_eta_difference']=max(r['eta_difference'] for r in rows)
out['max_deviance_difference']=max(r['deviance_difference'] for r in rows)
out['max_profile_cutoff_error']=max(r['profile_cutoff_max_error'] for r in rows)
assert out['max_eta_difference']<1e-5 and out['max_deviance_difference']<1e-3 and out['max_profile_cutoff_error']<1e-3,out
(R/'selected_high_precision_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
