"""Build the README diagrams directly from public aggregate evidence."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/generated/overview"
BLUE, ORANGE, INK, LIGHT = "#0072B2", "#D55E00", "#17324D", "#EAF2F8"


def report(name):
    return json.loads((ROOT / f"results/reports/{name}.json").read_text())


def save(fig, name):
    for ext in ("png", "svg"):
        target = OUT / f"{name}.{ext}"
        fig.savefig(target, dpi=210, bbox_inches="tight", facecolor="white")
        if ext == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(13.2, 4.5))
    ax.set(xlim=(0, 13.2), ylim=(0, 4.5)); ax.axis("off")
    ax.text(.1, 4.15, "A guarantee follows the complete scoring path", color=INK, fontsize=19, weight="bold")
    ax.text(.1, 3.7, "Observed history fixes features, the initial state and the affinity branch.", color="#5B6874", fontsize=12)
    boxes = [(0.1, "Absorbing flow", r"$z(t)$", "Active mass decays\ninto a dormant component"),
             (3.5, "Candidate map", r"$q(t)=c\odot z(t)/(c^Tz(t))$", "Normalization can\nredistribute active mass"),
             (6.9, "Readout", r"$G(t)=w_0+\delta^Tq_{>0}(t)$", "A certificate controls\nthis fixed response"),
             (10.3, "Prediction", r"$a\,\sigma(G(t)+b)$", "The sign follows only\nif the outer context stays fixed")]
    for x, title, eq, detail in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.7), 2.8, 1.65, boxstyle="round,pad=.09", fc=LIGHT, ec=BLUE, lw=1.3))
        ax.text(x+1.4, 3.05, title, ha="center", color=INK, fontsize=13, weight="bold")
        ax.text(x+1.4, 2.58, eq, ha="center", color=BLUE, fontsize=11)
        ax.text(x+1.4, 2.03, detail, ha="center", va="center", color="#425466", fontsize=10)
    for x in (3.0,6.4,9.8):
        ax.add_patch(FancyArrowPatch((x,2.5),(x+.35,2.5),arrowstyle="-|>",mutation_scale=18,color=INK))
    ax.add_patch(FancyBboxPatch((.1,.12),6.05,1.02,boxstyle="round,pad=.1",fc="#EFF8F3",ec="#23835B"))
    ax.text(.32,.82,"CERTIFICATE DOMAIN",color="#237A55",fontsize=11,weight="bold")
    ax.text(.32,.39,"Fixed positive tilts and specified rates;\nreadout and candidate context held constant.",color=INK,fontsize=11,va="center")
    ax.add_patch(FancyBboxPatch((6.7,.12),6.35,1.02,boxstyle="round,pad=.1",fc="#FFF2E9",ec=ORANGE))
    ax.text(6.95,.82,"OPERATING LIMIT",color=ORANGE,fontsize=11,weight="bold")
    ax.text(6.95,.39,"Changing the learned map or affinity\nrequires its own score-path check.",color=INK,fontsize=11,va="center")
    save(fig,"scoring-path")


def outcomes():
    paths,repair,feb=(report(x) for x in ("scored-paths","rate-repair","february-holdout"))
    jan=report("january-cohorts")["aggregate"]["gap_1d_to_7d__dose_1_to_9"]
    fig, axes=plt.subplots(1,3,figsize=(13.7,4.5))
    fig.subplots_adjust(left=.065,right=.99,bottom=.26,top=.80,wspace=.45)
    fig.suptitle("A structural guarantee and a predictive benefit are different results",fontsize=16,color=INK,weight="bold",y=.96)
    counts={}
    for i,ds in enumerate(("wikipedia","reddit")):
        values=[]; flow=[]
        for row in paths["datasets"][ds]["per_seed"]:
            by={p["path"]:p for p in row["path_results"]}
            values.append(100*by["dynamic_transition"]["violation_count"]/row["candidate_trajectories"])
            flow.append(100*by["implemented_flow"]["violation_count"]/row["candidate_trajectories"])
        x=i+np.linspace(-.12,.12,len(values))
        axes[0].scatter(x+.03,values,color=ORANGE,marker="o",s=45,label="Complete score" if i==0 else None,zorder=4)
        axes[0].scatter(x-.03,flow,facecolors="white",edgecolors=BLUE,marker="D",s=36,label="Pure flow" if i==0 else None,zorder=5)
        rows=repair["datasets"][ds]["per_seed"]
        counts[ds]={'specified':sum(r['fixed_changed_candidates'] for r in rows),'all_rate':sum(r['all_rate_changed_candidates'] for r in rows)}
    axes[0].set(xticks=[0,1],xticklabels=["Wikipedia","Reddit"],ylim=(-5,106),yticks=[0,25,50,75,100],ylabel="Paths with threshold crossing (%)",title="Complete-path diagnostic")
    axes[0].legend(frameon=False,fontsize=9,loc="center left")
    ax=axes[1]
    for i,ds in enumerate(("wikipedia","reddit")):
        n=counts[ds]["all_rate"]
        ax.bar(i+.14,n,width=.27,color=ORANGE,label="All-rate" if i==0 else None)
        ax.scatter(i-.14,counts[ds]["specified"],marker="D",facecolors="white",edgecolors=BLUE,s=45,zorder=5,label="Specified-rate" if i==0 else None)
        ax.annotate(f"{n:,}",(i+.14,n),xytext=(0,6),textcoords="offset points",ha="center",fontsize=11)
        ax.annotate("0",(i-.14,0),xytext=(0,7),textcoords="offset points",ha="center",fontsize=10,color=BLUE)
    ax.set(xticks=[0,1],xticklabels=["Wikipedia","Reddit"],ylim=(-2000,57000),ylabel="Changed candidate configurations",title="Extra reference deformation")
    ax.ticklabel_format(axis="y",style="sci",scilimits=(3,3));ax.legend(frameon=False,fontsize=9,loc="upper left")
    ax=axes[2];j=jan['ap_contrasts']['primary_shape']['mean'];f=feb['primary']['mean'];lo,hi=feb['conditional_bootstrap_interval']
    ax.scatter([0],[j*1000],color=BLUE,s=55,zorder=5)
    ax.errorbar([1],[f*1000],yerr=[[(f-lo)*1000],[(hi-f)*1000]],fmt="o",color=ORANGE,capsize=6,elinewidth=1.7,zorder=5)
    ax.axhline(0,color="#788895",lw=1,ls="--")
    ax.set(xticks=[0,1],xticklabels=["January\nexploratory","February\nfixed follow-up"],xlim=(-.55,1.6),ylabel=r"Local phase–MLP $\Delta$AP ($\times 10^{-3}$)",title="Local predictive follow-up")
    ax.text(.04,.03,"February interval crosses zero",transform=ax.transAxes,color=ORANGE,fontsize=10)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis="y",alpha=.2,zorder=0)
        ax.set_axisbelow(True)
    data={'scored_paths_source':'results/reports/scored-paths.json','repair_source':'results/reports/rate-repair.json','january_source':'results/reports/january-cohorts.json','february_source':'results/reports/february-holdout.json','changed_configurations':counts,'january_subgroup_mean_delta_ap':j,'february_subgroup_mean_delta_ap':f,'february_conditional_interval':[lo,hi],'interpretation':'January is exploratory; the February local-confirmation screen failed. A parameter-distance certificate is not an AP guarantee.'}
    data['source_sha256']={f'results/reports/{n}.json':hashlib.sha256((ROOT/f'results/reports/{n}.json').read_bytes()).hexdigest() for n in ['scored-paths','rate-repair','january-cohorts','february-holdout']}
    (OUT/'plot-data.json').write_text(json.dumps(data,indent=2)+'\n')
    save(fig,'research-outcomes')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.titlesize':12})
    pipeline();outcomes()
    print('README diagrams generated from public evidence.')
