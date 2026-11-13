"""Panel builders for probability, screening, transfer, trend and measurement evidence.

Since the six-figure consolidation these builders are imported by generate_six_main_figures.py;
the SPEC below documents the former ten-figure layouts only.

All values are read from completed analyses.  This file contains neutral panel
builders and does not import former supplementary-figure modules.
"""
from __future__ import annotations

import argparse
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

from figkit_compact import (BASEMAP, INK, MUTED, PAL, ROLE, ROOT, SEQ_ROSE, TCOL,
                            TMARK, basemap, build, map_proj, style)

RES, TAB = ROOT/'results', ROOT/'tables'
METALS = ['Cd','Hg','Pb']
THRESH = {'Cd':.08,'Pb':1.2,'Hg':.07}
PB = pd.read_csv(TAB/'T_pseudo_blindspot.csv')
PB = PB[PB['variant'].fillna('main').eq('main')].set_index('target')
REL = pd.read_csv(TAB/'T_pseudo_blindspot_reliability.csv')
REL = REL[REL['variant'].fillna('main').eq('main')]
TARGETS = ['Cd','CdDis','Hg','Pb','Ni','Glyphosate','PFOS']
LAB = {'CdDis':'Cd (dissolved)'}
V = pd.read_csv(TAB/'T_robustness_variants.csv')
VAR_LABELS = list(dict.fromkeys(V['label']))
VAR_SHORT = {'Main (LightGBM-Tobit, normal)':'Main: LGB-Tobit',
             'XGB-AFT, normal errors':'AFT: normal',
             'XGB-AFT, logistic errors':'AFT: logistic',
             'XGB-AFT, extreme errors':'AFT: extreme',
             'No country feature':'Without country', 'No year feature':'Without year',
             'LOD/LOQ values re-labelled':'Relabelled limits'}
HID = pd.read_csv(RES/'hidden_exceedance_summary.csv')
HID['variant'] = HID['variant'].fillna('main')
CI = pd.read_csv(TAB/'T_prevalence_rules_ci.csv')
HARD = pd.read_csv(TAB/'T_cd_hardness_sensitivity.csv').set_index('target')
CTY = pd.read_csv(TAB/'T_country_rule_prevalence.csv')
CONF = pd.concat([pd.read_csv(RES/f'conformal_{t}_real.csv') for t in METALS])
TRANSFER = pd.concat([pd.read_csv(RES/f'transfer_{t}.csv') for t in METALS])
BENCH = pd.concat([pd.read_csv(RES/f'bench_{t}_real.csv') for t in METALS])
TREND = pd.read_csv(TAB/'T_trend_artefacts.csv')
TREND['variant'] = TREND['variant'].fillna('main')
MAIN = TREND[TREND['variant'].eq('main') & TREND['target'].isin(METALS)]
NO_YEAR = TREND[TREND['variant'].eq('_noYear')]
COUNTRY_PRIORITY = pd.read_csv(TAB/'T_retest_priority_country.csv')
STATION_PRIORITY = pd.read_csv(TAB/'T_retest_priority_stations.csv')
SHORT = lambda c:c.split(' (')[0].replace('United States of America','USA')


def below_legend(ax,pad=48,**kwargs):
    """Keep a physical reading gap after canvas/axis compaction."""
    transform=ax.transAxes+ScaledTranslation(0,-pad/72,ax.figure.dpi_scale_trans)
    return ax.legend(loc='upper center',bbox_to_anchor=(.5,0),bbox_transform=transform,
                     borderaxespad=0,**kwargs)


def above_legend(ax,pad=8,**kwargs):
    transform=ax.transAxes+ScaledTranslation(0,pad/72,ax.figure.dpi_scale_trans)
    return ax.legend(loc='lower center',bbox_to_anchor=(.5,1),bbox_transform=transform,
                     borderaxespad=0,**kwargs)


def standalone(ss):
    return ss.get_gridspec().get_geometry()==(1,1)


def shared_key(fig, ss, handles, columns):
    """Put one legend in the space reserved below its related panels."""
    box=ss.get_position(fig)
    return fig.legend(handles=handles,loc='center',ncol=columns,frameon=False,
                      bbox_to_anchor=(box.x0+box.width/2,box.y0+box.height/2),
                      fontsize=12.5,handlelength=1.4,handletextpad=.35,
                      columnspacing=1.0,borderaxespad=0,borderpad=0)


def metal_keys():
    return [Line2D([],[],marker=TMARK[t],ls='',mfc=TCOL[t],mec=INK,
                   mew=.65,ms=7.5,label=t) for t in METALS]


def triage_keys():
    return [Patch(facecolor=colour,edgecolor=INK,linewidth=.6,label=label)
            for colour,label in [(PAL[3],'Likely below threshold'),
                                 (PAL[4],'Uncertain'),
                                 (PAL[0],'Likely above threshold')]]


def reliability(fig, ss):
    ax=fig.add_subplot(ss)
    ax.plot([0,1],[0,1],color=MUTED,ls='--',lw=1.1)
    for t in TARGETS:
        g=REL[REL.target.eq(t)&REL.n.ge(150)]
        ax.plot(g.mean_post,g.obs,marker=TMARK[t],color=TCOL[t],mec=INK,mew=.65,
                ms=6.8,lw=1.8,ls=':' if t in ['PFOS','Glyphosate'] else '-',label=LAB.get(t,t))
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=np.linspace(0,1,6),yticks=np.linspace(0,1,6))
    style(ax,'Conditional exceedance probability','Observed exceedance fraction')
    ax.legend(loc='lower right',ncol=2,fontsize=12,columnspacing=.55,
              handletextpad=.3,handlelength=1.35,frameon=False)


def recensor_triage(fig,ss,ncol=3,fontsize=12.5):
    ax=fig.add_subplot(ss)
    lc=PB.loc[TARGETS,'likely_compliant_share(<0.05)'].to_numpy()*100
    le=PB.loc[TARGETS,'likely_exceed_share(>0.5)'].to_numpy()*100
    un=100-lc-le;y=np.arange(len(TARGETS))
    for v,left,col,label in [(lc,np.zeros(len(y)),PAL[3],'Likely below threshold'),
                             (un,lc,PAL[4],'Uncertain'),(le,lc+un,PAL[0],'Likely above threshold')]:
        ax.barh(y,v,left=left,color=col,ec=INK,lw=.65,height=.68,label=label)
    for i,t in enumerate(TARGETS):
        rate=PB.loc[t,'false_compliance_in_likely(<0.05)']*100
        ax.text(102,i,f'{rate:.1f}',va='center',fontsize=12.2,color=INK,clip_on=False)
    ax.text(102,-.7,'False below\nthreshold (%)',fontsize=11.1,color=INK,va='bottom')
    ax.set_yticks(y,[LAB.get(t,t) for t in TARGETS]);ax.invert_yaxis();ax.set_xlim(0,100)
    style(ax,'Re-censored events (%)',None)
    below_legend(ax,ncol=ncol,fontsize=fontsize,columnspacing=.7,handlelength=1.1,handletextpad=.4)


def sensitivity(which):
    def fn(fig,ss):
        ax=fig.add_subplot(ss);positions={s:i for i,s in enumerate(VAR_LABELS)}
        for t,off in [('Cd',-.2),('Hg',0),('Pb',.2)]:
            g=V[V.t.eq(t)]
            ax.scatter(g[which]*100,[positions[s]+off for s in g.label],marker=TMARK[t],s=63,
                       color=TCOL[t],ec=INK,lw=.65,label=t,zorder=3)
            if which=='prev':
                half=HID[HID.target.eq(t)&HID.variant.eq('main')].iloc[0].prev_half
                ax.axvline(half*100,color=TCOL[t],ls=':',lw=1.4)
        ax.set_yticks(range(len(VAR_LABELS)),[VAR_SHORT.get(v,v) for v in VAR_LABELS])
        ax.set_ylim(len(VAR_LABELS)-.5,-.5)
        if which=='prev':
            ax.set_xlim(0,60);style(ax,'Exceedance prevalence (%)',None)
        else:
            ax.axvline(5,color=MUTED,ls='--',lw=1.0)
            ax.set_xlim(0,6);style(ax,'False-below-threshold rate (%)',None)
            if not standalone(ss):ax.tick_params(axis='y',labelleft=False)
        if standalone(ss):
            below_legend(ax,ncol=3,fontsize=12.5,handletextpad=.3,columnspacing=1.0)
    return fn


def layout6(fig):
    g=fig.add_gridspec(4,12,left=.025,right=.975,bottom=.025,top=.97,
                      hspace=.065,wspace=.22,height_ratios=[1,1.04,1,.07])
    shared_key(fig,g[3,:],metal_keys(),3)
    return dict(a=g[0,:],b=g[1,:],c=g[2,:7],d=g[2,7:])


def prevalence_rows():
    out=[]
    for _,r in CI[CI.variant.eq('main')&~CI.target.str.endswith('LQ')].iterrows():
        out.append(dict(label=LAB.get(r.target,r.target),t=r.target,und=r.und,ndc=r.ndc,dl=r.dl,
                        model=r.model,model_lo=r.model_lo,model_hi=r.model_hi,
                        half=r.half,half_lo=r.half_lo,half_hi=r.half_hi))
    for t,label in [('Cd','Cd, hardness'),('CdDis','Dissolved Cd, hardness')]:
        h=HARD.loc[t]
        out.append(dict(label=label,t=t,und=h.undetermined,ndc=h.prev_nd_compliant,dl=h.prev_dl,
                        model=h.prev_model,model_lo=np.nan,model_hi=np.nan,half=h.prev_half,
                        half_lo=np.nan,half_hi=np.nan))
    for _,r in CI[CI.variant.eq('_noMX')].iterrows():
        out.append(dict(label=f'{r.target}, without Mexico',t=r.target,und=r.und,ndc=r.ndc,dl=r.dl,
                        model=r.model,model_lo=r.model_lo,model_hi=r.model_hi,
                        half=r.half,half_lo=r.half_lo,half_hi=r.half_hi))
    return pd.DataFrame(out).sort_values('und').reset_index(drop=True)

DB=prevalence_rows()


def rule_keys():
    return [Line2D([],[],color='#D9DDE2',lw=8.5,label='Interval-compatible range'),
            Line2D([],[],marker='|',ls='',color=ROLE['ndc'],mew=2.3,ms=12,label='Lower bound'),
            Line2D([],[],marker='|',ls='',color=ROLE['dl'],mew=2.3,ms=12,label='Upper bound'),
            Line2D([],[],marker='s',ls='',mfc=ROLE['half'],mec=INK,mew=.65,ms=7.3,label='Half-limit'),
            Line2D([],[],marker='o',ls='',mfc=ROLE['model'],mec=INK,mew=.65,ms=7.3,label='Censored model')]


def rule_prevalence(fig,ss,legend=None):
    ax=fig.add_subplot(ss);y=np.arange(len(DB))
    ax.hlines(y,DB.ndc*100,DB.dl*100,color='#D9DDE2',lw=8.5,zorder=1,label='Interval-compatible range')
    ax.scatter(DB.ndc*100,y,marker='|',s=135,color=ROLE['ndc'],lw=2.3,zorder=2,label='Lower bound')
    ax.scatter(DB.dl*100,y,marker='|',s=135,color=ROLE['dl'],lw=2.3,zorder=2,label='Upper bound')
    for key,col,label,mk in [('half',ROLE['half'],'Half-limit','s'),('model',ROLE['model'],'Censored model','o')]:
        error=np.vstack([(DB[key]-DB[key+'_lo']).fillna(0),(DB[key+'_hi']-DB[key]).fillna(0)])*100
        ax.errorbar(DB[key]*100,y,xerr=error,fmt=mk,color=col,mec=INK,mew=.65,ms=7.3,
                    ecolor=INK,elinewidth=.9,zorder=3,label=label)
    ax.set_yticks(y,DB.label);ax.set_xlim(0,100)
    for i,r in DB.iterrows():ax.text(101.5,i,f'{r.und*100:.0f}%',va='center',fontsize=12,color=INK)
    ax.text(101.5,len(DB)-.15,'Blind\nspot',fontsize=11.5,va='bottom',color=INK)
    style(ax,'Exceedance prevalence (%)',None)
    if legend is None: legend=standalone(ss)
    if legend:
        below_legend(ax,pad=40,handles=rule_keys(),ncol=3,fontsize=12,frameon=False,
                     handlelength=1.4,columnspacing=.8,handletextpad=.4)


def prevalence_ratio(fig,ss):
    ax=fig.add_subplot(ss)
    for t,g in CTY[CTY.model.gt(.005)].groupby('t'):
        ax.scatter(g.und*100,g.half/g.model,marker=TMARK[t],s=np.clip(np.sqrt(g.n)/2.2,12,160),
                   color=TCOL[t],ec=INK,lw=.6,alpha=.9,label=LAB.get(t,t),zorder=3)
    ax.axhline(1,color=INK,lw=.9);ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator([.5,1,2,5,10,20]));ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}×'))
    style(ax,'Blind-spot share (%)','Half-limit / model prevalence')
    above_legend(ax,pad=3,ncol=4,fontsize=12,columnspacing=.55,handletextpad=.25)


COUNTRY_NAMES={'Netherlands (-the )':'Netherlands','United States of America (the)':'United States of America',
               'Macedonia (the former Yugoslav Republic of)':'North Macedonia','Bosnia and Herzegovina':'Bosnia and Herz.'}

def prevalence_maps(fig,ss):
    import geopandas as gpd
    countries=gpd.read_file(BASEMAP/'ne_110m_admin_0_countries.geojson').to_crs('+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs')
    value=CTY[CTY.t.eq('Cd')].assign(NAME=lambda d:d.country.map(lambda s:COUNTRY_NAMES.get(s,s)))
    value['delta']=(value.half-value.model)*100
    countries=countries.merge(value[['NAME','delta']],on='NAME',how='inner')
    g=ss.subgridspec(2,2,height_ratios=[1,.04],wspace=.035,hspace=.04)
    for k,(name,extent) in enumerate([('North America',(-128,-62,13,62)),('Europe',(-11,32,35,70))]):
        ax=fig.add_subplot(g[0,k]);basemap(ax,extent=extent,aspect=1.05)
        countries.plot(ax=ax,column='delta',cmap=SEQ_ROSE,vmin=0,vmax=80,edgecolor=INK,lw=.35,zorder=1)
        ax.set_anchor('S')
        ax.set_xlabel('');ax.set_ylabel('');ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set(visible=True,color=MUTED,linewidth=.8)
        ax.text(0,1.025,name,transform=ax.transAxes,fontsize=13.1,color=INK,va='bottom')
    cax=fig.add_subplot(g[1,:]);pos=cax.get_position()
    cax.set_position([pos.x0+pos.width*.12,pos.y0,pos.width*.76,pos.height])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,80),cmap=SEQ_ROSE),cax=cax,orientation='horizontal',extend='max')
    cb.set_label('Cd half-limit minus model prevalence (pp)',fontsize=12.6)
    cb.ax.tick_params(labelsize=11.5,length=3);cb.outline.set_edgecolor(INK)


def layout7(fig):
    g=fig.add_gridspec(3,1,left=.025,right=.975,bottom=.025,top=.975,
                      hspace=.065,height_ratios=[4.15,3.55,6.95])
    return dict(a=g[0],b=g[1],c=g[2])


def bound_frontier(fig,ss):
    ax=fig.add_subplot(ss)
    markers={'parametric':'o','conformal':'s','mondrian':'D'}
    labels={'parametric':'Gaussian','conformal':'Conformal','mondrian':'Country conformal'}
    selected=CONF[CONF.model.isin(['XGB-AFT-N','CSL','LGB-Tobit','D-CSL'])]
    for (t,mode),g in selected.groupby(['target','mode']):
        ax.scatter(g.cov_lower,g.certified_share*100,marker=markers[mode],s=65,color=TCOL[t],ec=INK,lw=.65,zorder=3)
    ax.axvline(.9,color=INK,ls='--',lw=1)
    handles=[Line2D([],[],marker=markers[m],ls='',mfc='white',mec=INK,ms=7.5,label=labels[m]) for m in markers]
    handles += [Line2D([],[],marker='o',ls='',mfc=TCOL[t],mec=INK,ms=7.5,label=t) for t in METALS]
    ax.legend(handles=handles,loc='upper right',bbox_to_anchor=(.82,.99),ncol=2,
              fontsize=12,frameon=False,handletextpad=.25,columnspacing=.55)
    ax.set_xlim(.39,1.0);ax.set_ylim(-2,75)
    ax.set_xticks(np.linspace(.4,1,7))
    style(ax,'Identified coverage lower bound','Upper bound ≤ threshold\n(% of blind-spot events)')


TRANSFER_MODELS=[('Sub-DL/2',ROLE['Substitution'],'o'),('XGB-AFT-N',ROLE['Censored'],'D'),
                 ('LGB-Tobit',ROLE['Censored'],'s'),('CSL',ROLE['Ensemble'],'^')]

def transfer_keys():
    return [Line2D([],[],marker=marker,ls='',mfc=colour,mec=INK,
                   mew=.6,ms=7,label=model) for model,colour,marker in TRANSFER_MODELS]


def country_transfer(metric):
    def fn(fig,ss):
        ax=fig.add_subplot(ss)
        for i,t in enumerate(METALS):
            group=TRANSFER[TRANSFER.target.eq(t)]
            for j,(model,col,mk) in enumerate(TRANSFER_MODELS):
                values=group[group.model.eq(model)][metric].to_numpy()
                jitter=np.random.default_rng(j).uniform(-.035,.035,len(values))
                ax.scatter(i+(j-1.5)*.20+jitter,values,s=43,marker=mk,color=col,ec=INK,lw=.6,
                           label=model if i==0 else None,zorder=3)
            if metric=='cindex':
                within=BENCH[BENCH.target.eq(t)&BENCH.model.eq('XGB-AFT-N')].iloc[0].cindex
                ax.plot([i-.38,i+.38],[within,within],color=ROLE['Censored'],lw=2)
        ax.set_xticks(range(3),METALS)
        ax.set_xlim(-.48,2.48)
        if metric=='cindex':
            ax.axhline(.5,color=MUTED,ls='--',lw=1);ax.set_ylim(.35,.95)
            style(ax,None,'Interval C-index\nin unseen country')
        else:
            ax.axhline(.9,color=INK,ls='--',lw=1);ax.set_ylim(.15,1.02)
            style(ax,None,'Identified coverage\nlower bound')
        if standalone(ss):
            below_legend(ax,pad=34,ncol=2,fontsize=12.5,columnspacing=.7,handletextpad=.25,handlelength=1.0)
    return fn


def layout8(fig):
    g=fig.add_gridspec(4,1,left=.035,right=.975,bottom=.022,top=.97,
                      hspace=.083,height_ratios=[1.14,1,1,.10])
    shared_key(fig,g[3,0],transfer_keys(),4)
    return dict(a=g[0,0],b=g[1,0],c=g[2,0])


def trend_association(fig,ss):
    ax=fig.add_subplot(ss)
    for t,g in MAIN.groupby('target'):
        ax.errorbar(g.r_und,g.artefact,yerr=[g.artefact-g.artefact_lo,g.artefact_hi-g.artefact],
                    fmt=TMARK[t],color=TCOL[t],mec=INK,mew=.6,ms=7.6,elinewidth=1,ecolor=MUTED,label=t,zorder=3)
    ax.axhline(0,color=INK,lw=.85);ax.axvline(0,color=INK,lw=.85)
    correlation=np.corrcoef(MAIN.r_und,MAIN.artefact)[0,1]
    ax.text(.97,.04,f'r = {correlation:.2f}, n = {len(MAIN)}',transform=ax.transAxes,ha='right',fontsize=12,color=INK)
    positions={('Cd','France'):(-86,-60,'left'),('Cd','Italy'):(-48,-45,'left'),
               ('Hg','France'):(-70,-17,'left'),('Hg','Italy'):(5,-17,'left'),
               ('Cd','Belgium'):(4,16,'left'),('Hg','Mexico'):(29,29,'right'),
               ('Pb','Mexico'):(7,42,'left')}
    for _,r in MAIN[MAIN.artefact.abs().gt(15)].iterrows():
        x,y,align=positions[(r.target,r.country)]
        ax.annotate(f'{r.target} {SHORT(r.country)}',(r.r_und,r.artefact),xytext=(x,y),textcoords='data',
                    fontsize=10.9,color=INK,ha=align,
                    arrowprops=dict(arrowstyle='-',color=MUTED,lw=.7))
    style(ax,'Blind-spot trend (pp per decade)','Half-limit minus model trend\n(pp per decade)')
    ax.legend(loc='upper left',fontsize=12)


def annual_series(target,country):
    d=pd.read_parquet(RES/f'crossfit_{target}.parquet',columns=['country','date','nd','y_lo','y_hi','post_exceed','undetermined'])
    d=d[d.country.eq(country)];nd=d.nd.to_numpy();tau=np.log10(THRESH[target])
    known=(~nd)&d.y_lo.to_numpy().__gt__(tau)
    d=d.assign(year=d.date.dt.year,half=known|(nd&(d.y_hi.to_numpy()-np.log10(2)>tau)),
               model=known+d.post_exceed.to_numpy(),und=d.undetermined)
    result=d.groupby('year').agg(n=('half','size'),half=('half','mean'),model=('model','mean'),und=('und','mean'))
    return result[result.n.ge(150)]

SERIES={('Cd','France'):annual_series('Cd','France'),('Pb','Mexico'):annual_series('Pb','Mexico')}


def draw_annual_case(ax,key,year_label=None):
    g=SERIES[key]
    ax.fill_between(g.index,0,g.und*100,color=ROLE['nd_undetermined'],alpha=.35,lw=0,label='Blind spot')
    ax.plot(g.index,g.half*100,color=ROLE['half'],marker='s',ms=5.8,mec=INK,mew=.6,lw=2,label='Half-limit')
    ax.plot(g.index,g.model*100,color=ROLE['model'],marker='o',ms=5.8,mec=INK,mew=.6,lw=2,label='Censored model')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
    ax.set_ylim(-4,105)
    ax.set_yticks([0,25,50,75,100])
    x,alignment=(.04,'left') if key[1]=='Mexico' else (.96,'right')
    ax.text(x,.96,f'{key[0]}, {key[1]}',transform=ax.transAxes,ha=alignment,va='top',fontsize=13,color=INK)
    style(ax,year_label,None)


def trend_cases(fig,ss):
    group=ss.subgridspec(2,2,hspace=.18,wspace=.09,height_ratios=[1,.20])
    france=fig.add_subplot(group[0,0])
    mexico=fig.add_subplot(group[0,1],sharey=france)
    draw_annual_case(france,('Cd','France'))
    france.set_ylabel('Share of events (%)',fontsize=15,color=INK)
    draw_annual_case(mexico,('Pb','Mexico'))
    mexico.tick_params(axis='y',labelleft=False)
    footer=fig.add_subplot(group[1,:])
    footer.set_axis_off()
    footer.text(.5,1,'Year',transform=footer.transAxes,ha='center',va='top',fontsize=15,color=INK)
    key_transform=footer.transAxes+ScaledTranslation(0,-21/72,fig.dpi_scale_trans)
    handles,labels=mexico.get_legend_handles_labels()
    footer.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),bbox_transform=key_transform,
                  ncol=3,fontsize=12.5,handlelength=1.25,columnspacing=.6,handletextpad=.3,
                  borderaxespad=0)


def trend_forest(fig,ss):
    ax=fig.add_subplot(ss)
    g=MAIN.assign(label=MAIN.target+' '+MAIN.country.map(SHORT)).sort_values('artefact')
    g=g[((g.artefact_lo>0)|(g.artefact_hi<0))&g.artefact.abs().ge(2)]
    y=np.arange(len(g));ax.hlines(y,0,g.artefact,color=MUTED,lw=1.15)
    for i,(_,r) in enumerate(g.iterrows()):
        ax.errorbar(r.artefact,i,xerr=[[r.artefact-r.artefact_lo],[r.artefact_hi-r.artefact]],fmt=TMARK[r.target],
                    color=TCOL[r.target],mec=INK,mew=.6,ms=7.3,ecolor=INK,elinewidth=.9)
    ax.axvline(0,color=INK,lw=.9);ax.set_yticks(y,g.label)
    style(ax,'Half-limit minus model trend (pp per decade)',None)


def year_removal(fig,ss):
    ax=fig.add_subplot(ss)
    merged=MAIN.merge(NO_YEAR,on=['target','country'],suffixes=('','_ny'))
    limits=[min(merged.artefact.min(),merged.artefact_ny.min())-5,max(merged.artefact.max(),merged.artefact_ny.max())+5]
    ax.plot(limits,limits,color=MUTED,ls='--',lw=1)
    for t,g in merged.groupby('target'):
        ax.scatter(g.artefact,g.artefact_ny,marker=TMARK[t],s=58,color=TCOL[t],ec=INK,lw=.65,label=t,zorder=3)
    correlation=np.corrcoef(merged.artefact,merged.artefact_ny)[0,1]
    ax.text(.04,.92,f'r = {correlation:.3f}',transform=ax.transAxes,fontsize=12,color=INK)
    ax.set(xlim=limits,ylim=limits)
    style(ax,'Full-model trend difference\n(pp per decade)','Without year feature\n(pp per decade)')
    if standalone(ss):ax.legend(loc='lower right',fontsize=12)


def layout9(fig):
    g=fig.add_gridspec(3,20,left=.025,right=.975,bottom=.025,top=.975,
                      wspace=.34,hspace=.065,height_ratios=[1.06,1,1])
    return dict(a=g[0,:],b=g[1,:],c=g[2,:13],d=g[2,13:])


station_map=(STATION_PRIORITY[STATION_PRIORITY.target.isin(METALS)].groupby('station')
             .agg(e=('expected_hidden_exceedances','sum'),n=('uncertain_events','sum'),lat=('lat','first'),lon=('lon','first'))
             .reset_index().dropna().sort_values('e'))
station_map['x'],station_map['y']=map_proj(station_map.lon,station_map.lat)
MAP_MAX=float(np.quantile(station_map.e,.98))


def followup_maps(fig,ss):
    g=ss.subgridspec(2,2,height_ratios=[1,.04],wspace=.035,hspace=.045)
    for k,(name,extent) in enumerate([('Mexico',(-117,-87,14.5,32.5)),('Europe',(-11,33,36,70))]):
        ax=fig.add_subplot(g[0,k]);basemap(ax,extent=extent,aspect=1.5)
        ax.set_anchor('S')
        ax.scatter(station_map.x,station_map.y,c=station_map.e,cmap=SEQ_ROSE,vmin=0,vmax=MAP_MAX,
                   s=(6+4*np.sqrt(station_map.n))*1.1,ec=INK,lw=.28,zorder=3)
        for spine in ax.spines.values():spine.set(visible=True,color=MUTED,linewidth=.8)
        ax.text(0,1.025,name,transform=ax.transAxes,fontsize=13,color=INK,va='bottom')
    cax=fig.add_subplot(g[1,:]);pos=cax.get_position()
    cax.set_position([pos.x0+pos.width*.20,pos.y0,pos.width*.60,pos.height])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,MAP_MAX),cmap=SEQ_ROSE),
                    cax=cax,orientation='horizontal',extend='max')
    cb.set_label('Expected hidden exceedances per station',fontsize=12.7)
    cb.ax.tick_params(labelsize=11.5,length=3);cb.outline.set_edgecolor(INK)


def followup_triage(fig,ss):
    ax=fig.add_subplot(ss);rows=[]
    for t in METALS:
        g=COUNTRY_PRIORITY[COUNTRY_PRIORITY.target.eq(t)].sort_values('blind_spot',ascending=False).head(5)
        rows += [(f'{t} {SHORT(r.country)}',r) for r in g.itertuples()]
    y=np.arange(len(rows));lc=np.array([r.likely_compliant_share for _,r in rows])*100
    le=np.array([r.likely_exceed_share for _,r in rows])*100;un=100-lc-le
    for v,left,col,label in [(lc,np.zeros(len(y)),PAL[3],'Likely below threshold'),
                             (un,lc,PAL[4],'Uncertain'),(le,lc+un,PAL[0],'Likely above threshold')]:
        ax.barh(y,v,left=left,color=col,ec=INK,lw=.6,height=.73,label=label)
    for i,(_,r) in enumerate(rows):
        value=COUNTRY_PRIORITY.loc[r.Index,'dl_reduction_fold_to_0.3std']
        if np.isfinite(value):ax.text(101.5,i,f'{value:.0f}×',va='center',fontsize=11.7,color=INK)
    ax.text(101.5,-1.0,'Limit\nreduction',fontsize=10.7,color=INK,va='bottom')
    ax.set_yticks(y,[label for label,_ in rows]);ax.invert_yaxis();ax.set_xlim(0,100)
    style(ax,'Blind-spot events (%)',None,fs=13.5)
    below_legend(ax,handles=triage_keys(),ncol=2 if standalone(ss) else 3,
                 fontsize=12.5,handlelength=1.1,columnspacing=.7,handletextpad=.4)


def priority_concentration(fig,ss):
    ax=fig.add_subplot(ss)
    for t in METALS:
        s=STATION_PRIORITY[STATION_PRIORITY.target.eq(t)].sort_values('expected_hidden_exceedances',ascending=False)
        cum=np.cumsum(s.expected_hidden_exceedances.to_numpy())/s.expected_hidden_exceedances.sum()
        fraction=np.arange(1,len(s)+1)/len(s)
        ax.plot(fraction*100,cum*100,color=TCOL[t],lw=2.3,label=t)
        index=np.searchsorted(fraction,.2)
        ax.scatter([20],[cum[index]*100],marker=TMARK[t],s=58,color=TCOL[t],ec=INK,lw=.7,zorder=3)
    ax.plot([0,100],[0,100],color=MUTED,ls='--',lw=1)
    ax.set(xlim=(0,100),ylim=(0,100))
    style(ax,'Stations targeted, ranked (%)','Expected hidden exceedances\ncaptured (%)')
    ax.legend(loc='lower right',fontsize=12)


def layout10(fig):
    g=fig.add_gridspec(3,1,left=.025,right=.975,bottom=.025,top=.975,
                      hspace=.065,height_ratios=[1.25,1.03,.95])
    return dict(a=g[0],b=g[1],c=g[2])


SPEC={
 6:dict(stem='Fig06_triage_sensitivity',panels=[('a',reliability),('b',recensor_triage),('c',sensitivity('prev')),('d',sensitivity('fc'))],
        layout=layout6,size=(12.4,12.4),subs={'a':(6.1,4.7),'b':(7.1,4.7),'c':(6.6,4.6),'d':(6.6,4.6)}),
 7:dict(stem='Fig07_screening_prevalence',panels=[('a',rule_prevalence),('b',prevalence_ratio),('c',prevalence_maps)],
        layout=layout7,size=(12.6,15.15),subs={'a':(9.5,6.3),'b':(6.5,5.0),'c':(8.5,4.3)}),
 8:dict(stem='Fig08_bounds_transfer',panels=[('a',bound_frontier),('b',country_transfer('cindex')),('c',country_transfer('cov_lower'))],
        layout=layout8,size=(12.2,10.4),subs={'a':(6.7,5.0),'b':(6.0,4.9),'c':(6.0,4.9)}),
 9:dict(stem='Fig09_reporting_trends',panels=[('a',trend_association),('b',trend_cases),('c',trend_forest),('d',year_removal)],
        layout=layout9,size=(12.8,13.6),subs={'a':(6.3,4.9),'b':(11.0,4.4),'c':(9.0,5.2),'d':(6.1,4.9)}),
 10:dict(stem='Fig10_measurement_priorities',panels=[('a',followup_maps),('b',followup_triage),('c',priority_concentration)],
         layout=layout10,size=(12.6,13.65),subs={'a':(9.2,4.7),'b':(7.5,6.5),'c':(6.2,5.0)})
}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*',type=int,default=list(SPEC));args=parser.parse_args()
    for number in args.only:
        spec=SPEC[number]
        build(spec['panels'],spec['layout'],spec['stem'],spec['stem'],figsize=spec['size'],sub_sizes=spec['subs'])
        print(spec['stem'],'done',flush=True)
