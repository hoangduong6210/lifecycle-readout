"""Vector schematics connecting the stated theorem to the retained scoring pipeline."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
BLUE='#0072B2';ORANGE='#B64B00';GREEN='#00785A';GREY='#50565D'
FILLS={BLUE:'#EEF6FC',ORANGE:'#FFF4EA',GREEN:'#EDF8F1',GREY:'#F4F5F6'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def canvas(height,ylim):
 f,a=plt.subplots(figsize=(7.1,height));a.set_position([.015,.015,.97,.97]);a.set_xlim(0,12);a.set_ylim(*ylim);a.axis('off');return f,a

def panel(a,x,y,w,h,title,lines,color=BLUE,fs=12.3,titlefs=13):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035,rounding_size=.09',lw=1.1,ec=color,fc=FILLS[color],zorder=2))
 a.text(x+.17,y+h-.19,title,ha='left',va='top',fontweight='bold',fontsize=titlefs,color=color,zorder=4)
 space=h-.54
 for i,line in enumerate(lines):
  yy=y+space*(1-(i+.55)/len(lines))
  a.text(x+w/2,yy,line,ha='center',va='center',fontsize=fs,color='#17212A',zorder=4)

def arr(a,p,q,color=GREY,style='-'):
 a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=12,lw=1.1,color=color,linestyle=style,zorder=3,shrinkA=3,shrinkB=3))

def save(f,out,n):
 for ext in ['pdf','eps','png']:f.savefig(Path(out)/f'Fig{n}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.04)
 plt.close(f)

def build(out):
 # One candidate batch, with the actual encoder widths and parameter dimensions.
 f,a=canvas(6.5,(0,11.5))
 a.text(.25,11.22,'Forward pass for each observed or sampled destination',fontsize=13,fontweight='bold')
 a.text(.25,10.80,r'$N$: candidate pairs in a batch; $K=5$ phase coordinates; elapsed input $\tau\geq0$',fontsize=12)
 ys=[9.35,7.65,5.95,4.25,2.55,.85];x=.25;w=7.25;hh=1.28
 main=[
 ('1  Strictly past graph encoder',['20 event tokens / endpoint + 32 time features',r'Pair context $h\in\mathbb{R}^{N\times128}$'],BLUE),
 ('2  Initial phase head',[r'$[h,\phi]\in\mathbb{R}^{N\times134}\ \longrightarrow\ \mathrm{MLP}\ \longrightarrow\ z(0)\in\mathbb{R}^{N\times5}$','Fit: Gumbel softmax; evaluate: softmax'],BLUE),
 ('3  Absorbing silence flow',[r'$z_i(\tau)=z_i(0)e^{-d_i\tau}\ (i\geq1)$',r'$z_0(\tau)=1-\mathbf{1}^{\top}z_{\rm active}(\tau)$; output $N\times5$'],BLUE),
 ('4  Learned candidate transform',[r'$q=\mathrm{softmax}\{\log z+z^\top T_\theta(\phi,\log(1+\tau))\}$',r'Output $N\times5$; outside the pure-flow guarantee'],ORANGE),
 ('5  Ordered scalar readout',[r'$G=w^\top q,\qquad \gamma=\sigma(G+b)\in\mathbb{R}^{N}$','Weight ordering certifies pure flow'],BLUE),
 ('6  Sampled-candidate score',[r'$s=a\,\gamma\in\mathbb{R}^{N}$','Before numerical clipping'],BLUE)]
 for y,(title,lines,col) in zip(ys,main):panel(a,x,y,w,hh,title,lines,col,fs=12)
 for i in range(5):arr(a,(x+w/2,ys[i]),(x+w/2,ys[i+1]+hh))
 sx=8.0;sw=3.73
 panel(a,sx,9.35,sw,hh,'Pair history',[r'$\phi\in\mathbb{R}^{N\times6}$; $\tau\in\mathbb{R}^{N}$','Train-only normalization'],GREY,fs=11.6)
 arr(a,(sx,9.65),(7.52,8.46),GREY)
 panel(a,sx,5.95,sw,hh,'Decay rates',[r'$d=\mathrm{softplus}(\theta_d)\in\mathbb{R}_{+}^{4}$','Shared across candidates'],BLUE,fs=11.5)
 arr(a,(sx,6.49),(7.50,6.49),BLUE)
 panel(a,sx,4.25,sw,hh,'Map network',[r'$[\phi,\log(1+\tau)]\longrightarrow T_\theta$',r'$T_\theta\in\mathbb{R}^{N\times5\times5}$'],ORANGE,fs=11.7)
 arr(a,(sx,4.79),(7.50,4.79),ORANGE)
 panel(a,sx,2.55,sw,hh,'Gate weights',[r'$w\in\mathbb{R}^{5}$; $b\in\mathbb{R}$',r'$\delta=\mathrm{cumsum}(\mathrm{softplus}(\theta))$'],BLUE,fs=11.5)
 arr(a,(sx,3.09),(7.50,3.09),BLUE)
 panel(a,sx,.85,sw,hh,'Affinity head',[r'$a=\sigma(\ell_{\rm base}+\ell_{\rm history})$','One affinity per candidate'],GREY,fs=11.7,titlefs=12.2)
 arr(a,(sx,1.39),(7.50,1.39),GREY)
 a.text(.25,.24,'Silence check: freeze context, state and affinity; vary elapsed input.',fontsize=11.9)
 save(f,out,1)
 # Clockwise four-panel bridge from model tensors to the exact scalar certificate.
 f,a=canvas(5.3,(0,9.2))
 a.text(.25,8.90,'Model quantities to an exact all-state certificate',fontsize=13,fontweight='bold')
 panel(a,.25,4.85,5.45,3.55,'1  Fixed candidate quantities',[
 r'Freeze $z(0),\ h,\ \phi,\ a$ along the path',
 r'$c\in\mathbb{R}_{+}^{5}$; $d,\delta\in\mathbb{R}_{+}^{4}$',
 r'$\delta_i=w_i-w_0$; $w_0,b$ are scalars',
 'Tilt may differ across candidates'],BLUE,fs=11.9)
 panel(a,6.30,4.85,5.45,3.55,'2  Scalar derivative',[
 r'$q_i=c_i z_i/(c^\top z),\quad G=w_0+\delta^\top q$',
 r'$-\dot G=F(q)=a_{\rm aux}^{\top}q-(\delta^\top q)(e^\top q)$',
 r'$(a_{\rm aux})_i=\delta_i d_i,\quad e_i=d_i(1-c_0/c_i)$',
 r'Append zeros: $\delta_0=(a_{\rm aux})_0=e_0=0$'],BLUE,fs=11.5,titlefs=12.1)
 panel(a,6.30,.75,5.45,3.55,'3  Six active-pair tests',[
 r'$A_i=c_0\delta_i d_i/c_i$',
 r'$K_{ij}=(\delta_i-\delta_j)(d_i-d_j)$',
 r'$B_{ij}=K_{ij}+c_0(\delta_i d_j/c_j+\delta_jd_i/c_i)$',
 r'$\mu_{ij}=B_{ij}+2\sqrt{A_iA_j}\ \geq\ 0$',
 r'$K=5$: four active states, six pairs'],BLUE,fs=11.7)
 panel(a,.25,.75,5.45,3.55,'4  Certificate scope',[
 r'All $\mu_{ij}\geq0$: safe for every $z(0)$',
 r'One $\mu_{ij}<0$: unsafe edge witness',
 'Fix tilt, rates, readout and affinity',
 'Changing context: recheck the score'],GREEN,fs=11.6,titlefs=12.0)
 arr(a,(5.7,6.62),(6.30,6.62),BLUE);arr(a,(9.02,4.85),(9.02,4.30),BLUE);arr(a,(6.30,2.52),(5.7,2.52),GREEN)
 a.text(.25,.17,'All-state theorem: numerical trajectories remain diagnostics.',fontsize=11.8)
 save(f,out,2)
 # No-refit repair, including the two reference changes before x projection.
 f,a=canvas(5.8,(0,9.7))
 a.text(.25,9.38,'No-refit repair and the reference changes that precede it',fontsize=12.6,fontweight='bold')
 panel(a,.25,7.05,3.5,1.80,'1  Learned map',[
 r'$T_\theta(\phi,\log(1+\tau))$',
 'Varies with time and pair',
 'Retain original scores'],ORANGE,fs=11.2,titlefs=12.0)
 panel(a,4.25,7.05,3.5,1.80,'2  Freeze clock',[
 r'$T^{\rm ref}=T_\theta(\phi,0)$',
 r'Fix $\phi$ and $\tau_{\rm ref}=0$',
 'Changes the scorer'],ORANGE,fs=11.3,titlefs=11.8)
 panel(a,8.25,7.05,3.5,1.80,'3  Gradient fit',[
 r'$T^{\rm grad}_{ij}=\widehat\beta_j-\widehat\beta_i$',
 'Least-squares fit',
 r'Gauge $\widehat\beta_0=0$'],ORANGE,fs=11.25,titlefs=11.4)
 arr(a,(3.75,7.95),(4.25,7.95),ORANGE);arr(a,(7.75,7.95),(8.25,7.95),ORANGE)
 panel(a,3.0,5.10,6.0,1.15,'4  Inverse-tilt coordinates',[
 r'$c_i=e^{\widehat\beta_i},\ c_0=1,\qquad x_i^0=\sqrt{c_0/c_i}=e^{-\widehat\beta_i/2}$'],BLUE,fs=12.0)
 # Route the downward line outside the raw-coordinate box; no crossing of labels.
 arr(a,(10.0,7.05),(8.55,6.25),BLUE)
 panel(a,.25,2.40,5.45,1.95,'5a  Specified-rate repair',[
 r'$x^*=\arg\min_{Hx\geq h}\ \frac{1}{2}\|x-x^0\|_2^2$',
 'Pair constraints become halfspaces',
 r'If $x^0$ is feasible: $x^*=x^0$'],GREEN,fs=11.7,titlefs=12.1)
 panel(a,6.30,2.40,5.45,1.95,'5b  All-rate repair',[
 r'$\ell_i=\sqrt{(\delta_*-\delta_i)/\delta_*}$',
 r'$x_i^{\rm all}=\max\{x_i^0,\ell_i\}$',
 r'$\delta_*=0$: leave $x^0$ unchanged'],BLUE,fs=11.5,titlefs=12.1)
 arr(a,(4.0,5.10),(2.97,4.35),GREEN);arr(a,(8.0,5.10),(9.02,4.35),BLUE)
 panel(a,.25,.55,11.5,1.15,'6  Reconstruct both scores using the same remaining quantities',[
 r'$\widetilde\beta_i=-2\log x_i,\quad \widetilde c_i=e^{\widetilde\beta_i},\quad \widetilde q=\mathrm{normalize}(\widetilde c\odot z),\quad \widetilde s=a\,\sigma(w^\top\widetilde q+b)$',
 'Keep initial states, rates, readout and affinity common'],GREY,fs=11.9,titlefs=12)
 arr(a,(2.97,2.4),(2.97,1.70),GREEN);arr(a,(9.02,2.4),(9.02,1.70),BLUE)
 a.text(.25,.12,'Inverse-tilt distance has a geometric guarantee; compare AP separately.',fontsize=11.8)
 save(f,out,5)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'generated/article')
 destination=parser.parse_args().output
 destination.mkdir(parents=True,exist_ok=True)
 build(destination)
