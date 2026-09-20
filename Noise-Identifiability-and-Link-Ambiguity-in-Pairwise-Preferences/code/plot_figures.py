"""Paper figures from saved data, at their final 6.5 inch width."""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
ROOT=Path(__file__).resolve().parents[1]
BLUE='#0072B2';ORANGE='#B64B00';GREEN='#008060';PURPLE='#8E5D9F'

def wilson(k,n):
    p=np.asarray(k)/n;z=1.95996398454;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return np.vstack([p-(mid-h),(mid+h)-p])

def save(fig,name):
    dest=ROOT/'manuscript/figures'/name
    fig.savefig(dest.with_suffix('.pdf'),facecolor='white')
    fig.savefig(dest.with_suffix('.png'),dpi=300,facecolor='white')
    plt.close(fig)

def main():
    (ROOT/'manuscript/figures').mkdir(parents=True,exist_ok=True)
    (ROOT/'internal').mkdir(exist_ok=True)
    style={'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
        'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'ps.fonttype':42,'lines.linewidth':1.2,'lines.markersize':4}
    with plt.rc_context(style):
        info=pd.read_csv(ROOT/'data/information_check.csv')
        pop=pd.read_csv(ROOT/'data/population.csv');pop=pop[pop.k==4]
        fig,axs=plt.subplots(1,2,figsize=(6.5,2.05),layout='constrained')
        ax=axs[0];x=info.delta
        ax.loglog(x,info.known_information,'o-',color=BLUE,label='Known utilities')
        ax.loglog(x,info.unknown_information,'s-',color=ORANGE,label='Free utilities')
        ax.loglog(x,20*x**2,':',color=BLUE,label=r'Theory: $20\delta^2$')
        ax.loglog(x,np.pi**2*180/576*x**6,':',color=ORANGE,label=r'Theory: $(5\pi^2/16)\delta^6$')
        ax.set(xlabel=r'Gap scale $\delta$',ylabel=r'Information for $\eta$ ($n=1$)',title='(a) Probit information')
        ax.text(.06,.91,r'Known utilities ($\delta^2$)',color=BLUE,transform=ax.transAxes,fontsize=8)
        ax.text(.50,.11,r'Free utilities ($\delta^6$)',color=ORANGE,transform=ax.transAxes,fontsize=8)
        ax=axs[1];x=pop.delta
        ax.loglog(x,pop.explicit_kl,'^-',color=PURPLE,label='Scaled alternative')
        ax.loglog(x,pop.optimized_kl,'o-',color=GREEN,label='Refitted alternative')
        ax.loglog(x,9*np.pi**4/16384000*x**10,':',color='black',label=r'Theory: $C\delta^{10}$')
        ax.set(xlabel=r'Gap scale $\delta$',ylabel='Sum of edge KL divergences',title='(b) Probit to logistic distance')
        ax.legend(loc='upper left',frameon=False)
        save(fig,'fig1_orders')
        summary=pd.read_csv(ROOT/'data/paper_summary.csv');summary=summary[summary.k==4]
        fig,axs=plt.subplots(1,2,figsize=(6.5,2.05),layout='constrained')
        s=summary[summary.world=='probit_P'].sort_values('delta');ax=axs[0]
        for pref,label,color,mark in [('L','Profile',BLUE,'o'),('C','Hoeffding triangle',ORANGE,'s')]:
            ax.errorbar(s.delta,s[pref+'_width'],yerr=1.96*s[pref+'_width_se'],color=color,
                        marker=mark,label=label,capsize=2)
        ax.set(xscale='log',yscale='log',xlabel=r'Gap scale $\delta$',ylabel='Mean interval width',title='(a) Interval width under probit')
        ax.set_xticks(s.delta,[str(x) for x in s.delta]);ax.xaxis.set_minor_formatter(NullFormatter());ax.legend(frameon=False,loc='upper left')
        ax=axs[1]
        for world,color in [('probit_P',BLUE),('logistic_Q',ORANGE)]:
            s=summary[summary.world==world].sort_values('delta')
            for j,(pref,marker,line) in enumerate([('L','o','-'),('C','s','--')]):
                label=('Probit' if world=='probit_P' else 'Logistic')+' / '+('profile' if pref=='L' else 'Hoeffding')
                # Horizontal displacement is cosmetic and recorded here.
                offset=1+(.03 if world=='probit_P' else -.03)+(j-.5)*.03
                ax.errorbar(s.delta*offset,s[pref+'_cover_count']/200,
                    yerr=wilson(s[pref+'_cover_count'],200),color=color,marker=marker,
                    linestyle=line,capsize=2,label=label,markerfacecolor='white' if j else color)
        ax.axhline(.95,color='0.5',ls=':',lw=1)
        ax.set(xscale='log',xlabel=r'Gap scale $\delta$',ylabel='Coverage of generating noise',ylim=(-.04,1.04),title='(b) Coverage (200 repeats)')
        ax.set_xticks(s.delta,[str(x) for x in s.delta]);ax.xaxis.set_minor_formatter(NullFormatter());ax.legend(frameon=False,loc='center left',fontsize=7.5)
        save(fig,'fig2_intervals')
    rows=[['1a','data/information_check.csv','All 8 deterministic points; K=4, a=.6','code/verify_key_claims.py + code/plot_figures.py','manuscript/figures/fig1_orders.pdf'],
          ['1b','data/population.csv','k=4; all 14 deltas; slope fitted only at delta<=.025','code/plot_figures.py','manuscript/figures/fig1_orders.pdf'],
          ['2','data/finite.csv + data/paper_summary.csv','k=4; all deltas/worlds/repeats, no fit exclusion','code/reproduce_tables.py + code/plot_figures.py','manuscript/figures/fig2_intervals.pdf']]
    with (ROOT/'internal/figure_sources.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['figure','input_files','filter_and_transformation','generator','output']);w.writerows(rows)
    print('Wrote two vector PDFs and 300 dpi PNG previews.')

if __name__=='__main__':main()
