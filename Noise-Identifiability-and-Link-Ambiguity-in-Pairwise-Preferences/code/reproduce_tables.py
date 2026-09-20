"""Reanalyse saved counts; no simulations, model calls, or network access."""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def conservative_interval(counts, n, k, alpha=.05):
    # Fixed structural rule: widest span with first two vertices; no per-dataset selection.
    edges = list(combinations(range(k), 2))
    inds = [edges.index(e) for e in [(0, 1), (1, k-1), (0, k-1)]]
    # Integer subtraction preserves the centred sufficient statistic.
    m = np.array([(2*int(counts[i])-int(n))/int(n) for i in inds])
    r = np.sqrt(2*np.log(6/alpha)/int(n))
    l, u = m-r, m+r
    # On this rectangle H=xyz/(x+y-z) decreases in x,y and increases in z.
    regular = (l>0).all() and l[2]>max(u[0],u[1]) and l[0]+l[1]>u[2]
    if not regular:
        return 0., .5, True
    b2lo = u[0]*u[1]*l[2]/(u[0]+u[1]-l[2])
    b2hi = l[0]*l[1]*u[2]/(l[0]+l[1]-u[2])
    if b2lo>1 or b2hi<0:  # Empty physical intersection: use safe fallback.
        return 0., .5, True
    return (1-np.sqrt(min(1,b2hi)))/2, (1-np.sqrt(max(0,b2lo)))/2, False

def main():
    data = ROOT/'data'
    f = pd.read_csv(data/'finite.csv')
    assert len(f)==3200 and not f.duplicated(['k','delta','world','rep']).any()
    rows=[]
    for t in f.itertuples():
        lo,hi,fall=conservative_interval(json.loads(t.counts),t.n,t.k)
        rows.append(dict(k=t.k,delta=t.delta,n=t.n,world=t.world,rep=t.rep,
                         eta_true=t.eta_true,lo=lo,hi=hi,width=hi-lo,fallback=fall,
                         covers_true=lo<=t.eta_true<=hi,
                         covers_Q=lo<=t.etaQ<=hi))
    c=pd.DataFrame(rows)
    c.to_csv(data/'conservative_intervals.csv',index=False)
    group=['k','delta','n','world']
    summary=f.groupby(group).agg(repeats=('rep','size'),L_eta=('L_eta','mean'),
        L_width=('L_width','mean'),L_width_se=('L_width','sem'),
        L_cover_count=('L_covers_true','sum'),P_cover_count=('P_covers_true','sum'),
        L_boundary_count=('L_boundary','sum'),P_boundary_count=('P_boundary','sum'),
        L_failed_count=('L_success',lambda x:int((~x).sum())),
        P_failed_count=('P_success',lambda x:int((~x).sum())),
        union_coverage=('union_covers_true','mean'),union_diameter=('union_diameter','mean'),
        union_length=('union_length','mean')).reset_index()
    cs=c.groupby(group).agg(C_width=('width','mean'),C_width_se=('width','sem'),
        C_cover_count=('covers_true','sum'),C_Q_cover_count=('covers_Q','sum'),
        C_fallback_count=('fallback','sum')).reset_index()
    summary=summary.merge(cs,on=group,validate='one_to_one')
    summary.to_csv(data/'paper_summary.csv',index=False)
    key=summary[(summary.k==4)&(summary.delta==.025)&(summary.world=='probit_P')]
    print(key.to_json(orient='records',indent=2))
    return summary

if __name__=='__main__':main()
