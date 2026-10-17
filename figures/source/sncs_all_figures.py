"""Rebuild the eight SNCS figures from the committed aggregate CSV package.

No source interactions, private audit records, model fits or remote compute are
needed. Matplotlib and NumPy must be installed. Output defaults to figures/generated/sncs.
"""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np


def build(source, destination):
    source=Path(source)
    out=Path(destination);out.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    blue='#0072B2';orange='#D55E00';green='#009E73';grey='#666666'
    def read_csv(name):
        return list(csv.DictReader((source/name).open()))
    def save(fig,n):
        for ext in ('pdf','eps','png'):fig.savefig(out/f'Fig{n}.{ext}',dpi=360,bbox_inches='tight')
        plt.close(fig)
    def table(name,header,rows):
        with (out/name).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(rows)

    # Figure 2: exact same-tilt safe / unsafe rate examples.
    t=np.linspace(0,3,501);rows=[]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.1,3.7));fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.90,wspace=.45)
    def flow(p,d):
        active=p[1:][None,:]*np.exp(-t[:,None]*d[None,:]);return np.column_stack((1-active.sum(1),active))
    for d,label,color in [(np.array([1.,2.]),'Rates (1, 2): safe',blue),(np.array([1.,10.]),'Rates (1, 10): unsafe',orange)]:
        z=flow(np.array([0.,.5,.5]),d);weighted=z*np.array([1.,4.,4.]);q=weighted/weighted.sum(1,keepdims=True);G=q[:,1:]@np.array([2.,1.]);ax.plot(t,G,label=label,color=color,lw=1.7,ls='-' if color==blue else '--')
        rows.extend([[float(ti),label,float(gi)] for ti,gi in zip(t,G)])
    ax.set(xlabel='Elapsed time\n(dimensionless)',ylabel='Readout $G(t)$',xlim=(0,.055),ylim=(1.47,1.51));ax.text(0,1.13,'(a)',transform=ax.transAxes,va='top',fontweight='bold');ax.legend(frameon=False,fontsize=12,loc='lower left')
    ax.annotate('Initial slope $>0$',xy=(.01,1.501),xytext=(.16,.89),textcoords='axes fraction',arrowprops={'arrowstyle':'->','color':orange},color=orange,fontsize=12)
    z=flow(np.array([.1,.1,.8]),np.array([1.,2.]));q=z*np.array([1.,4.,4.]);q/=q.sum(1,keepdims=True)
    bx.plot(t,q[:,1],color=green,label='$q_1(t)$');bx.set(xlabel='Elapsed time\n(dimensionless)',ylabel='Active-coordinate mass $q_1$');bx.text(0,1.13,'(b)',transform=bx.transAxes,va='top',fontweight='bold')
    bx.annotate('Initial slope $56/1369>0$',xy=(.05,q[8,1]),xytext=(.24,.89),textcoords='axes fraction',arrowprops={'arrowstyle':'->','color':green},color=green,fontsize=12)
    bx.set_ylim(0,.22);bx.text(.30,.50,"while $G'(0)<0$",transform=bx.transAxes,fontsize=12)
    rows.extend([[float(ti),'Readout-preserving redistribution',float(qi)] for ti,qi in zip(t,q[:,1])]);save(fig,3);table('Figure3_data.csv',['Elapsed_time','Configuration','Readout_or_coordinate'],rows)

    # Figure 3: exact halfspace and a nontrivial synthetic projection.
    fig,ax=plt.subplots(figsize=(4.7,3.05));x=np.linspace(0,1.15,400);boundary=(1-x)/2
    ax.fill_between(x,np.maximum(0,boundary),1.05,color='#DCEEF6',label='Specified-rate feasible set')
    ax.axhspan(np.sqrt(.5),1.05,color='#B3DDCF',label='All-rate subset')
    ax.plot(x,boundary,color=blue,lw=1.4,label='$x_1+2x_2=1$')
    raw=np.array([.1,.1]);projected=raw+(.7/5)*np.array([1.,2.]);allrate=np.array([.1,np.sqrt(.5)])
    for p,label,color in [(raw,'Raw',orange),(projected,'Fixed-rate repair',blue),(allrate,'All-rate repair',green)]:ax.scatter(*p,s=30,color=color,zorder=4);ax.annotate(label,p,xytext=(6,-22) if label=='All-rate repair' else (6,5),textcoords='offset points',fontsize=12,color=color)
    ax.plot([raw[0],projected[0]],[raw[1],projected[1]],'--',color=blue,lw=1);ax.plot([raw[0],allrate[0]],[raw[1],allrate[1]],':',color=green,lw=1.2)
    ax.set(xlim=(0,1.15),ylim=(0,1.05),xlabel=r'$x_1=\sqrt{c_0/c_1}$',ylabel=r'$x_2=\sqrt{c_0/c_2}$');ax.legend(frameon=False,fontsize=11,loc='upper right');fig.tight_layout();save(fig,4)

    # Figure 4: all retained learned-path fractions; no model identity strings.
    data=read_csv('Figure6_data.csv');s={'datasets':{}}
    for dataset in ['wikipedia','reddit']:
        values=[]
        for seed in [1,7,42]:
            subset=[q for q in data if q['Dataset'].lower()==dataset and int(q['Seed'])==seed]
            assert len(subset)==3
            values.append({'seed':seed,'candidate_trajectories':int(subset[0]['Candidate_trajectories']),
                'path_results':[{'path':{'Flow only':'implemented_flow','Frozen map':'fixed_transition','Dynamic map':'dynamic_transition'}[q['Path']],
                'violation_count':int(q['Violating_trajectories']),'violation_fraction':int(q['Violating_trajectories'])/int(q['Candidate_trajectories'])} for q in subset]})
        s['datasets'][dataset]={'per_seed':values}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.1,3.8));rows=[]
    for dataset,a,label in [('wikipedia',ax,'Wikipedia'),('reddit',bx,'Reddit')]:
        seeds=s['datasets'][dataset]['per_seed'];positions=np.arange(len(seeds));width=.24
        for offset,path,color,pretty in [(-width,'implemented_flow',blue,'Flow only'),(0,'fixed_transition',green,'Frozen map'),(width,'dynamic_transition',orange,'Dynamic map')]:
            vals=[next(p['violation_fraction'] for p in q['path_results'] if p['path']==path)*100 for q in seeds]
            a.bar(positions+offset,vals,width,color=color,label=pretty,hatch={'implemented_flow':'\\\\','fixed_transition':'..','dynamic_transition':'///'}[path],edgecolor='#303030',linewidth=.4)
            zero=[i for i,v in enumerate(vals) if v==0]
            if zero:a.scatter(positions[zero]+offset,np.zeros(len(zero)),marker={'implemented_flow':'s','fixed_transition':'o','dynamic_transition':'^'}[path],s=32,color=color,edgecolor='#202020',linewidth=.5,zorder=5,clip_on=False)
            rows.extend([[label,q['seed'],pretty,next(p['violation_count'] for p in q['path_results'] if p['path']==path),q['candidate_trajectories']] for q in seeds])
        a.set(xticks=positions,xticklabels=[q['seed'] for q in seeds],xlabel='Training seed',ylim=(0,108));a.text(.02,1.04,label,transform=a.transAxes,va='bottom',fontsize=12)

        if dataset=='wikipedia':
            a.text(.5,.53,'0% for all three paths\nat every seed\n(n = 1,768 per seed)',transform=a.transAxes,ha='center',va='center',fontsize=12)
        else:
            a.text(.5,-.48,'Flow only: 0%\nat every seed',transform=a.transAxes,ha='center',va='top',fontsize=12,color=blue)
        a.set_ylabel('Increasing paths (%)');a.grid(axis='y',color='#E6E6E6');a.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),frameon=False,fontsize=12,loc='upper center',bbox_to_anchor=(.5,1),ncol=3)
    fig.subplots_adjust(left=.10,right=.98,bottom=.33,top=.75,wspace=.45);save(fig,6);table('Figure6_data.csv',['Dataset','Seed','Path','Violating_trajectories','Candidate_trajectories'],rows)

    # Figure 5: deformation avoided at specified rates, with every seed retained.
    data=read_csv('Figure7_data.csv');s={'datasets':{}}
    for dataset in ['wikipedia','reddit']:
        s['datasets'][dataset]={'per_seed':[{'seed':int(q['Seed']),'candidate_count':int(q['Candidates']),
            'all_rate_changed_candidates':int(q['All_rate_changes']),'fixed_changed_candidates':int(q['Specified_rate_changes']),
            'mean_squared_distance_all_rate':float(q['All_rate_squared_distance']),
            'mean_squared_distance_fixed':float(q['Specified_rate_squared_distance'])} for q in data if q['Dataset'].lower()==dataset]}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.1,3.8));rows=[]
    for dataset,a in [('wikipedia',ax),('reddit',bx)]:
        seeds=s['datasets'][dataset]['per_seed'];pos=np.arange(3)
        a.bar(pos-.16,[q['all_rate_changed_candidates']/q['candidate_count']*100 for q in seeds],.32,color=orange,label='All-rate repair',hatch='///',edgecolor='#303030',linewidth=.4)
        a.bar(pos+.16,[q['fixed_changed_candidates']/q['candidate_count']*100 for q in seeds],.32,color=blue,label='Specified-rate repair',hatch='\\\\',edgecolor='#303030',linewidth=.4)
        a.scatter(pos+.16,np.zeros(len(pos)),marker='s',s=36,color=blue,edgecolor='#202020',linewidth=.5,zorder=6,clip_on=False)
        if dataset=='wikipedia':
            a.scatter(pos-.16,np.zeros(len(pos)),marker='o',s=36,color=orange,edgecolor='#202020',linewidth=.5,zorder=6,clip_on=False)
            a.text(.5,.55,'0% for both repairs\nat every seed\n(n = 1,768 per seed)',transform=a.transAxes,ha='center',va='center',fontsize=12)
        else:
            vals=[q['all_rate_changed_candidates']/q['candidate_count']*100 for q in seeds]
            for j,v in enumerate(vals):a.annotate(f'{v:.1f}%',(j-.16,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=12)
            a.text(.5,-.48,'Specified-rate: 0%\nat every seed',transform=a.transAxes,ha='center',va='top',fontsize=12,color=blue)
        a.set(xticks=pos,xticklabels=[q['seed'] for q in seeds],xlabel='Training seed',ylabel='Tilts changed (%)',ylim=(0,105));a.text(.02,1.04,dataset.capitalize(),transform=a.transAxes,va='bottom');a.grid(axis='y',color='#E6E6E6');a.set_axisbelow(True)
        rows.extend([[dataset,q['seed'],q['candidate_count'],q['all_rate_changed_candidates'],q['fixed_changed_candidates'],q['mean_squared_distance_all_rate'],q['mean_squared_distance_fixed']] for q in seeds])
    fig.legend(*ax.get_legend_handles_labels(),frameon=False,fontsize=12,loc='upper center',bbox_to_anchor=(.5,1),ncol=2);fig.subplots_adjust(left=.10,right=.98,bottom=.33,top=.75,wspace=.45);save(fig,7)
    table('Figure7_data.csv',['Dataset','Seed','Candidates','All_rate_changes','Specified_rate_changes','All_rate_squared_distance','Specified_rate_squared_distance'],rows)

    # Figure 6: discovery and untouched-next-month check, in percentage points.
    data=read_csv('Figure8_data.csv');intervals=read_csv('Figure8_interval.csv')
    def month_values(name):
        q=[float(a['AP_difference']) for a in data if a['Month']==name]
        mean=next(float(a['Mean_AP_difference']) for a in intervals if a['Month']==name)
        assert len(q)==3 and abs(float(np.mean(q))-mean)<1e-14
        return {'values':q,'mean':mean}
    jan=month_values('January');feb={'primary':month_values('February')}
    fi=next(a for a in intervals if a['Month']=='February');feb['conditional_bootstrap_interval']=[float(fi['Lower_95']),float(fi['Upper_95'])]
    fig,ax=plt.subplots(figsize=(5.0,3.65));rows=[]
    for x,item,color,month in [(0,jan,blue,'January'),(1,feb['primary'],orange,'February')]:
        ax.scatter(x+np.array([-.06,0,.06]),np.array(item['values'])*100,s=30,color=color,zorder=4)
        ax.scatter(x,item['mean']*100,marker='D',s=43,color='black',zorder=5)
        rows.extend([[month,seed,value] for seed,value in zip([101,202,303],item['values'])])
    lo,hi=np.array(feb['conditional_bootstrap_interval'])*100;mean=feb['primary']['mean']*100
    ax.errorbar(1,mean,yerr=np.array([[mean-lo],[hi-mean]]),fmt='none',ecolor='black',capsize=5,lw=1.2)
    ax.axhline(0,color=grey,ls='--',lw=1);ax.set(xlim=(-.4,1.4),xticks=[0,1],xticklabels=['January\nExploratory subgroup','February\nFrozen models'],ylabel='Phase minus flexible AP (pp)')
    ax.text(.98,.96,'Three training seeds\nDiamond: mean\nFebruary: conditional 95%',transform=ax.transAxes,ha='right',va='top',fontsize=11)
    fig.tight_layout();save(fig,8);table('Figure8_data.csv',['Month','Seed','AP_difference'],rows)
    table('Figure4_data.csv',['Point','x1','x2'],[['Raw',.1,.1],['Specified-rate projection',6/25,19/50],['All-rate projection',.1,float(np.sqrt(.5))]])
    table('Figure8_interval.csv',['Month','Mean_AP_difference','Lower_95','Upper_95','Queries','Users'],[['January',jan['mean'],'','',4210,58],['February',feb['primary']['mean'],*feb['conditional_bootstrap_interval'],4328,57]])
    from sncs_pipeline_diagrams import build as build_detailed_diagrams
    build_detailed_diagrams(out)
    print('Eight explanatory figures generated in PDF EPS and PNG formats',flush=True)

if __name__=='__main__':
    import argparse
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=root/'paper/working/sncs')
    parser.add_argument('--output',type=Path,default=root/'figures/generated/sncs')
    args=parser.parse_args()
    build(args.source,args.output)
