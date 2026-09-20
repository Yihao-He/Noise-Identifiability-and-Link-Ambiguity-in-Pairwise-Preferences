"""Fixed, reproducible comparison graphs; no utility rescaling or edge filtering."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np

SEED = 20260920

def connected(k, edges):
    seen = {0}
    while True:
        more = seen | {j for i,j in edges if i in seen} | {i for i,j in edges if j in seen}
        if more == seen:
            return len(seen) == k
        seen = more

def incidence(k, edges):
    B = np.zeros((len(edges), k-1))
    for r,(i,j) in enumerate(edges):
        if i < k-1: B[r,i] = 1
        if j < k-1: B[r,j] = -1
    return B

def make_graph(k, family, p=None, seed=SEED):
    rng = np.random.default_rng(seed)
    all_edges = list(combinations(range(k),2))
    attempts = 1
    if family == 'complete': edges = all_edges
    elif family == 'er':
        while True:
            edges = [e for e,keep in zip(all_edges,rng.random(len(all_edges)) < p) if keep]
            if connected(k,edges): break
            attempts += 1
    elif family == 'sparse':
        # A fixed spanning path plus K uniformly sampled distinct non-path chords.
        path = [(i,i+1) for i in range(k-1)]
        rest = [e for e in all_edges if e not in path]
        edges = sorted(path + [rest[i] for i in rng.choice(len(rest),k,replace=False)])
    elif family == 'tree': edges = [(i,i+1) for i in range(k-1)]
    else: raise ValueError(family)
    name = f'{family}_K{k}' + (f'_p{p:g}' if p is not None else '')
    return dict(name=name,k=k,family=family,p=p,seed=seed,attempts=attempts,edges=edges)

def geometry(g):
    k = g['k']; B = incidence(k,g['edges']); w = np.arange(k-1,0,-1,dtype=float)
    d = B @ w
    Q = np.linalg.qr(B,mode='reduced')[0]
    r3 = d**3 - Q @ (Q.T @ d**3)
    r5 = d**5 - Q @ (Q.T @ d**5)
    G3 = r3@r3
    if len(d)==k-1: G3=0.;r3[:]=0
    s5 = r5-r3*(r3@r5/G3) if G3>1e-15 else r5
    G5 = s5@s5 if len(d)>k else 0.
    return B,w,d,float(G3),float(G5)

def suite():
    out = [make_graph(k,'complete') for k in (5,10,20)]
    out += [make_graph(k,'er',p,SEED+100*k+int(100*p)) for k in (10,20,50) for p in (.2,.5)]
    out += [make_graph(k,'sparse',seed=SEED+10*k) for k in (10,20,50)]
    out += [make_graph(10,'tree')]
    return out

if __name__=='__main__':
    path=Path(__file__).parent/'results';path.mkdir(exist_ok=True)
    (path/'graphs.json').write_text(json.dumps(suite(),indent=2))
