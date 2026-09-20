import numpy as np
from scipy.special import erf,xlogy
from scipy.optimize import least_squares,brentq
from itertools import combinations
def graph(k):
    edges=list(combinations(range(k),2));b=np.zeros((len(edges),k-1))
    for r,(i,j) in enumerate(edges):
        if i<k-1:b[r,i]=1
        if j<k-1:b[r,j]=-1
    return b,edges
def link(z,kind):
    if kind=='L':
        f=np.tanh(z/2);return f,(1-f*f)/2
    return erf(np.sqrt(np.pi)*z/4),.5*np.exp(-np.pi*z*z/16)
def probability(eta,u,b,kind):return (1+(1-2*eta)*link(b@u,kind)[0])/2
def stable_kl(y,p):
    y=np.asarray(y);p=np.clip(p,1e-15,1-1e-15);d=p-y
    out=np.zeros_like(p);small=np.abs(d)<.01*np.minimum(y,1-y)
    ys=y[small];ds=d[small]
    for power in range(2,9):out[small]+=(((-1.)**power/ys**(power-1)+1/(1-ys)**(power-1))*ds**power/power)
    z=~small
    out[z]=xlogy(y[z],y[z]/p[z])+xlogy(1-y[z],(1-y[z])/(1-p[z]))
    return np.maximum(out,0)
def residual_jac(theta,y,n,b,kind,fixed_eta=None):
    eta=theta[0] if fixed_eta is None else fixed_eta;u=theta[1:] if fixed_eta is None else theta
    f,fp=link(b@u,kind);p=np.clip((1+(1-2*eta)*f)/2,1e-15,1-1e-15)
    diff=p-y;res=np.sign(diff)*np.sqrt(2*n*stable_kl(y,p));dr=np.sqrt(n/(p*(1-p)))
    nz=np.abs(res)>1e-9;dr[nz]=n[nz]*diff[nz]/(p[nz]*(1-p[nz])*res[nz])
    j=(.5*(1-2*eta)*fp)[:,None]*b
    if fixed_eta is None:j=np.column_stack([-f,j])
    return res,dr[:,None]*j
def fixed(y,n,b,kind,eta,u):
    def fun(x):return residual_jac(x,y,n,b,kind,eta)[0]
    def jac(x):return residual_jac(x,y,n,b,kind,eta)[1]
    r=least_squares(fun,u,jac=jac,bounds=(-10,10),xtol=1e-12,ftol=1e-12,gtol=1e-10,max_nfev=200,x_scale='jac')
    return r,float(r.fun@r.fun)
def fit(y,n,b,kind,interval=True,bound=10,eta_max=.49,eta_starts=None,extra_starts=None,max_nfev=300):
    y=np.asarray(y,float);n=np.broadcast_to(n,y.shape).astype(float);m=2*y-1
    starts=[]
    initial_vectors=[]
    for eta in ([.05,.16,.2,.35] if eta_starts is None else eta_starts):
        scaled=np.clip(m/(1-2*eta),-.95,.95)
        if kind=='L':z=2*np.arctanh(scaled)
        else:
            from scipy.special import erfinv
            z=4/np.sqrt(np.pi)*erfinv(scaled)
        u=np.linalg.lstsq(b,z,rcond=None)[0]
        initial_vectors.append(np.r_[eta,u])
    initial_vectors.extend([] if extra_starts is None else extra_starts)
    for initial in initial_vectors:
        fun=lambda x:residual_jac(x,y,n,b,kind)[0]
        jac=lambda x:residual_jac(x,y,n,b,kind)[1]
        lower=np.r_[0,np.full(b.shape[1],-bound)];upper=np.r_[eta_max,np.full(b.shape[1],bound)]
        r=least_squares(fun,np.clip(initial,lower+1e-10,upper-1e-10),jac=jac,bounds=(lower,upper),x_scale='jac',xtol=1e-12,ftol=1e-12,gtol=1e-10,max_nfev=max_nfev)
        starts.append(r)
    r=min(starts,key=lambda r:r.fun@r.fun);loss=float(r.fun@r.fun);eta=float(r.x[0]);u=r.x[1:]
    out={'eta':eta,'u':u,'deviance':loss,'success':bool(r.success),'boundary':bool(eta<1e-6 or eta>eta_max-1e-6),'spread_deviance':float(max(t.fun@t.fun for t in starts)-loss)}
    if interval:
        cutoff=loss+3.841458820694124
        def f(e):return fixed(y,n,b,kind,e,u)[1]-cutoff
        lo=0 if f(0)<=0 else brentq(f,0,eta,xtol=1e-9)
        hi=.49 if f(.49)<=0 else brentq(f,eta,.49,xtol=1e-9)
        out.update(lo=lo,hi=hi,width=hi-lo)
    return out
