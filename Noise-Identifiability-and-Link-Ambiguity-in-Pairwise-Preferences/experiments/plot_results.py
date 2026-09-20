"""Recreate all revision figures/tables from the measured CSV files."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'manuscript'/'figures'
TABLES=ROOT.parent/'supplementary'/'tables'
COLORS=['#0072B2','#D55E00','#009E73']

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=220,facecolor='white')
    plt.close(fig)

def graph_label(name):
    if name.startswith('complete'):return 'Complete '+name.split('K')[1]
    if name.startswith('er'):return 'ER '+name.split('K')[1].replace('_p',', ')
    if name.startswith('sparse'):return 'Sparse '+name.split('K')[1]
    return 'Tree 10'

def plot_finite_intervals(fs):
    """Facet by delta; show two methods per panel with all recorded uncertainty."""
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.25), sharex=True, sharey='row')
    fig.subplots_adjust(left=.11, right=.985, bottom=.16, top=.81,
                        wspace=.15, hspace=.25)
    styles = [
        ('Logistic profile', '#0072B2', 'o', '-', '#0072B2', -.035),
        ('Hoeffding link-robust', '#D55E00', 's', '--', 'white', .035),
    ]
    for j, delta in enumerate([.05, .1, .2]):
        axs[0, j].set_title(rf'({chr(97+j)}) $\delta={delta:g}$', fontsize=10, pad=7)
        for method, color, marker, ls, face, offset in styles:
            x = fs[(fs.delta == delta) & (fs.method == method)].sort_values('n')
            assert len(x) == 4 and list(x.n) == [100, 1000, 10000, 100000]
            # Symmetric offsets in log10(n) separate overlapping intervals.
            # Y values and interval endpoints are unchanged.
            xp = x.n.to_numpy() * 10**offset
            kw = dict(color=color, marker=marker, linestyle=ls, markersize=4,
                      linewidth=1.25, markerfacecolor=face, markeredgewidth=1,
                      capsize=2, elinewidth=.9, zorder=3)
            ci = np.maximum(0, np.array([x.coverage-x.coverage_lo,
                                        x.coverage_hi-x.coverage]))
            axs[0, j].errorbar(xp, x.coverage, yerr=ci, **kw)
            axs[1, j].errorbar(xp, x.width, yerr=1.96*x.width_mcse, **kw)
        axs[0, j].axhline(.95, color='#777777', ls=':', lw=.9, zorder=1)
        axs[0, j].set(ylim=(-.045, 1.07), yticks=[0, .5, 1])
        axs[1, j].set(ylim=(-.02, .54), yticks=[0, .25, .5])
        for ax in axs[:, j]:
            ax.set(xscale='log', xlim=(65, 155000),
                   xticks=[100, 1000, 10000, 100000])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis='y', color='#E3E6E8', lw=.55, zorder=0)
            ax.tick_params(axis='both', length=3, width=.7, pad=3)
            ax.spines['left'].set_linewidth(.7)
            ax.spines['bottom'].set_linewidth(.7)
        if j:
            for ax in axs[:, j]:
                ax.tick_params(axis='y', left=False)
                ax.spines['left'].set_visible(False)
    axs[0, 0].set_ylabel('Coverage', labelpad=7)
    axs[1, 0].set_ylabel('Mean width', labelpad=7)
    fig.supxlabel(r'Repeats per edge $n$', y=.02, fontsize=10)
    handles = [
        Line2D([], [], color='#0072B2', marker='o', lw=1.25, ms=4,
               label='Logistic profile'),
        Line2D([], [], color='#D55E00', marker='s', ls='--', lw=1.25,
               ms=4, mfc='white', label=r'Link-robust $C_H$'),
        Line2D([], [], color='#777777', ls=':', lw=.9, label='95% coverage'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.535, 1.01),
               ncol=3, frameon=False, handlelength=2, columnspacing=1.25,
               handletextpad=.5, fontsize=10)
    save(fig, 'moderate_sample_intervals')
    manifest = OUT/'revision_figure_manifest.json'
    metadata = json.loads(manifest.read_text()) if manifest.exists() else {}
    metadata.update(
        finite_sample_layout='2 rows (coverage, mean width) x 3 columns (delta=0.05, 0.1, 0.2); shared exterior legend',
        finite_sample_encoding='Method encoded by color, marker, and line style; x positions offset by +/-0.035 in log10(n); y values and intervals unchanged',
        finite_sample_height_inches=3.25)
    manifest.write_text(json.dumps(metadata, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figure', choices=['all', 'finite'], default='all')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True);TABLES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
        'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    if args.figure == 'finite':
        plot_finite_intervals(pd.read_csv(ROOT/'results/finite_sample_summary.csv'))
        return
    df=pd.read_csv(ROOT/'results/scaling.csv');gs=pd.read_csv(ROOT/'results/graph_summary.csv')
    selected=['complete_K20','er_K50_p0.2','sparse_K50']
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.45),layout='constrained')
    for ix,(name,c,marker) in enumerate(zip(selected,COLORS,['o','s','^'])):
        sub=df[df.graph==name]
        for window in ['near_tie_audit','requested']:
            x=sub[sub.window==window].sort_values('delta')
            for ax,key,power in zip(axs,['information','kl'],[6,10]):
                coef=np.pi**2/576*x.G3 if key=='information' else 2*(.6*np.pi**2/30720)**2*x.G5
                ax.loglog(x.delta,x[key]/coef,color=c,marker=marker,ms=3.7,lw=1,
                    label=graph_label(name) if window=='requested' else None)
    xx=np.geomspace(.000125,.2,200)
    for ax,power,title,ylabel in zip(axs,[6,10],['(a) Noise information','(b) Refitted link distance'],
        [r'$I_{\eta,\mathrm{eff}}/(\pi^2G_3/576)$',r'$J_\delta/[2(a\pi^2/30720)^2G_5]$']):
        ax.loglog(xx,xx**power,':',color='.3',lw=1,label=rf'$\delta^{{{power}}}$')
        ax.set(xlabel=r'Utility scale $\delta$',ylabel=ylabel,title=title)
        ax.set_xticks([.0001,.001,.01,.1]);ax.grid(alpha=.12)
    axs[0].legend(loc='upper left',frameon=False,fontsize=10)
    # Labels are designed at 6.5 inches; no resizing during manuscript inclusion.
    save(fig,'general_graph_scaling')

    fig,axs=plt.subplots(3,4,figsize=(6.5,7.8),layout='constrained')
    for ax,(name,sub) in zip(axs.flat,df[df.family!='tree'].groupby('graph',sort=False)):
        for window in ('near_tie_audit','requested'):
            x=sub[sub.window==window].sort_values('delta')
            ax.loglog(x.delta,x.information/(np.pi**2/576*x.G3),'o-',color=COLORS[0],ms=2.5,lw=.8)
            ax.loglog(x.delta,x.kl/(2*(.6*np.pi**2/30720)**2*x.G5),'s-',color=COLORS[1],ms=2.5,lw=.8)
        ax.loglog(xx,xx**6,':',color=COLORS[0],lw=.6)
        ax.loglog(xx,xx**10,':',color=COLORS[1],lw=.6)
        ax.set(title=graph_label(name),xlabel=r'$\delta$')
        ax.tick_params(labelsize=10)
    save(fig,'all_graph_scaling')

    # Tables retain every graph, including the exact-zero tree control.
    lines=[]
    for r in gs.itertuples():
        slope=lambda v:'--' if pd.isna(v) else f'{v:.3f}'
        def sci(v):
            if v==0:return '0'
            exponent=int(np.floor(np.log10(abs(v))));mantissa=v/10**exponent
            return f'{mantissa:.2f}\\!\\times\\!10^{{{exponent}}}'
        lines.append(f"{graph_label(r.graph)} & {r.M} & {r.cycle_rank} & {r.density:.3f} & ${sci(r.G3)}$ & ${sci(r.G5)}$ & {slope(r.requested_information_slope)} & {slope(r.requested_kl_slope)} & {slope(r.near_tie_audit_information_slope)} & {slope(r.near_tie_audit_kl_slope)} \\")
    (TABLES/'graph_rows.tex').write_text('\n'.join(line+'\\' for line in lines)+'\n\\bottomrule\n')
    if not (ROOT/'results/finite_sample_summary.csv').exists():return
    fs=pd.read_csv(ROOT/'results/finite_sample_summary.csv')
    # Publish a cell only after the complete sample-size grid has been run.
    if len(fs)<36:return
    plot_finite_intervals(fs)
    lines=[]
    for delta in (.05,.1,.2):
        for n in (100,1000,10000,100000):
            p=fs[(fs.delta==delta)&(fs.n==n)&(fs.method=='Logistic profile')].iloc[0]
            h=fs[(fs.delta==delta)&(fs.n==n)&(fs.method=='Hoeffding link-robust')].iloc[0]
            lines.append(f'{delta:g} & ${n:g}$ & {p.coverage:.3f} & {p.width:.4f} & {p.bias:+.4f} & {h.coverage:.3f} & {h.width:.4f} & {h.bias:+.4f} \\\\')
    (TABLES/'finite_rows.tex').write_text('\n'.join(lines)+'\n\\bottomrule\n')
    p=fs[(fs.delta==.2)&(fs.n==100000)&(fs.method=='Logistic profile')].iloc[0]
    h=fs[(fs.delta==.2)&(fs.n==100000)&(fs.method=='Hoeffding link-robust')].iloc[0]
    macros=dict(RequestedInfoRange=f'{gs.requested_information_slope.min():.3f}--{gs.requested_information_slope.max():.3f}',
        RequestedKLRange=f'{gs.requested_kl_slope.min():.3f}--{gs.requested_kl_slope.max():.3f}',
        AuditInfoRange=f'{gs.near_tie_audit_information_slope.min():.5f}--{gs.near_tie_audit_information_slope.max():.5f}',
        AuditKLRange=f'{gs.near_tie_audit_kl_slope.min():.5f}--{gs.near_tie_audit_kl_slope.max():.5f}',
        FiniteCoverage=f'{100*p.coverage:g}\\%',FiniteWidth=f'{p.width:.5f}',FiniteBias=f'${p.bias:+.5f}$',
        RobustCoverage=f'{100*h.coverage:g}\\%',RobustWidth=f'{h.width:.5f}')
    (ROOT.parent/'manuscript/revision_results.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    (OUT/'revision_figure_manifest.json').write_text(json.dumps(dict(
        inputs=['experiments/results/scaling.csv','experiments/results/finite_sample_summary.csv'],
        normalization='I/(pi^2 G3/576); KL/[2(a pi^2/30720)^2 G5]',
        zero_handling='Tree values are exactly zero, reported in the table, omitted from logarithmic axes.',
        selected_graphs=selected,uncertainty='95% Wilson for coverage; 1.96 MCSE for mean width',
        transforms='Separate line segments for the two disjoint delta windows; no fitted values substituted.',
        width_inches=6.5,
        finite_sample_layout='2 rows (coverage, mean width) x 3 columns (delta=0.05, 0.1, 0.2); shared exterior legend',
        finite_sample_encoding='Method encoded by color, marker, and line style; x positions offset by +/-0.035 in log10(n); y values and intervals unchanged',
        finite_sample_height_inches=3.25),indent=2))

if __name__=='__main__':main()
