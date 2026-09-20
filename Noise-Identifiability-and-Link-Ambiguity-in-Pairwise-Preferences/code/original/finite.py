import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from inference import graph,fit,probability,stable_kl
BASE=Path(__file__).resolve().parents[1];R=BASE/'results'
SEED=20260920
def one(job):
    mech,world,rep=job;k=mech['k'];d=mech['delta'];n=mech['n'];b,_=graph(k)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,k,int(d*100000),world,rep]))
    p=np.array(mech['p'],float);q=np.array(mech['q'],float);true=p if world==0 else q;eta_true=.2 if world==0 else mech['etaQ']
    counts=rng.binomial(n,true);y=counts/n;L=fit(y,n,b,'L');P=fit(y,n,b,'P')
    lo=min(L['lo'],P['lo']);hi=max(L['hi'],P['hi']);overlap=max(0,min(L['hi'],P['hi'])-max(L['lo'],P['lo']))
    union_length=L['width']+P['width']-overlap
    if k==3:reject=bool(rng.random()<.05);llr=0.
    else:
        coef=np.array(mech['llr_coef'],float)
        llr=float(np.sum((counts.astype(np.longdouble)-np.longdouble(n)*p.astype(np.longdouble))*coef.astype(np.longdouble))+np.longdouble(mech['nkl']))
        reject=llr>mech['np_threshold']
    row=dict(k=k,delta=d,n=n,world='probit_P' if world==0 else 'logistic_Q',rep=rep,eta_true=eta_true,etaQ=mech['etaQ'],n_delta6=mech['n_delta6'],n_delta10=mech['n_delta10'],tv_bound=mech['tv_bound'],power_bound=mech['power_bound'],honest_diameter_probability_lower=mech['honest_diameter_probability_lower'],counts=json.dumps(counts.tolist()),llr=llr,np_reject=reject,union_diameter=hi-lo,union_length=union_length,union_gap=(hi-lo)-union_length,union_covers_true=(L['lo']<=eta_true<=L['hi']) or (P['lo']<=eta_true<=P['hi']),union_covers_both=((L['lo']<=.2<=L['hi']) or (P['lo']<=.2<=P['hi'])) and ((L['lo']<=mech['etaQ']<=L['hi']) or (P['lo']<=mech['etaQ']<=P['hi'])),diameter_ge_separation=hi-lo>=abs(.2-mech['etaQ']))
    for name,f in [('L',L),('P',P)]:
        for key in ['eta','lo','hi','width','deviance','success','boundary','spread_deviance']:row[name+'_'+key]=f[key]
        row[name+'_covers_true']=f['lo']<=eta_true<=f['hi']
        row[name+'_covers_link_target']=f['lo']<=(mech['etaQ'] if name=='L' else .2)<=f['hi']
    return row
def main():
    mechanisms=json.loads((R/'mechanisms.json').read_text());rng=np.random.default_rng(SEED)
    for m in mechanisms:
        n=m['n'];p=np.array(m['p'],np.longdouble);q=np.array(m['q'],float);coef=np.array(m['llr_coef'],np.longdouble)
        if m['k']==3:m['np_threshold']=0.
        else:
            null=rng.binomial(n,q,size=(10000,len(q))).astype(np.longdouble)
            llrs=np.sum((null-n*p)*coef,axis=1)+np.longdouble(m['nkl'])
            m['np_threshold']=float(np.quantile(llrs,.95))
    (R/'test_calibration.json').write_text(json.dumps(mechanisms,indent=2))
    jobs=[(m,w,r) for m in mechanisms for w in [0,1] for r in range(200)];start=time.time()
    with ProcessPoolExecutor(max_workers=10) as pool,(R/'finite.jsonl').open('w') as out:
        for i,row in enumerate(pool.map(one,jobs,chunksize=1)):
            out.write(json.dumps(row)+'\n');out.flush()
            if (i+1)%100==0:print('FINITE',i+1,len(jobs),'seconds',round(time.time()-start),flush=True)
    df=pd.read_json(R/'finite.jsonl',lines=True);df.to_csv(R/'finite.csv',index=False)
    cols=['L_eta','P_eta','L_width','P_width','L_covers_true','P_covers_true','union_covers_true','union_covers_both','union_diameter','union_length','union_gap','diameter_ge_separation','np_reject','L_deviance','P_deviance','L_success','P_success','L_spread_deviance','P_spread_deviance']
    df.groupby(['k','delta','n','world'])[cols].mean().reset_index().to_csv(R/'finite_summary.csv',index=False)
    print('COMPLETED finite',flush=True)
if __name__=='__main__':main()
