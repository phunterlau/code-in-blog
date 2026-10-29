import os,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
O=next(p/'outputs/qwen-pp-discovery-control' for p in Path(__file__).resolve().parents if (p/'outputs/qwen-pp-discovery-control').is_dir())
data=json.loads((O/'analysis.json').read_text())
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,len(data),figsize=(5*len(data),4),squeeze=False)
labels=['Signed PP','RMS','Random for PP','Random for RMS','Public']
methods=['signed','rms','random_signed','random_rms','public']
for ax,(task,x) in zip(axs[0],data.items()):
    co=[x['winner_by_method'][m] for m in methods];ks=[f'L{l}N{n}' for l,n in co];v=[x['restoration'][k]['restore']['mean'] for k in ks];bounds=[x['restoration'][k]['restore']['bootstrap95_mean'] for k in ks]
    errors=np.array([[max(0,vv-bb[0]),max(0,bb[1]-vv)] for vv,bb in zip(v,bounds)]).T
    ax.bar(range(5),v,color=['#156d91','#ad6a2a','#aebbc0','#c2b8ab','#674f91']);ax.errorbar(range(5),v,yerr=errors,fmt='none',ecolor='#273944',capsize=3)
    ax.axhline(0,color='#6c777c',lw=.7);ax.set_xticks(range(5),[l+'\n'+k for l,k in zip(labels,ks)],rotation=35,ha='right',fontsize=8);ax.set_title(task.capitalize());ax.set_ylabel('Fresh-pair restoration Δ fixed logit gap')
fig.suptitle('Natural activation restoration: selected singletons, 8 fresh pairs each');fig.text(.5,.01,'Bars are candidate-selection methods; shared coordinates are the same intervention. Intervals: paired-case bootstrap of means.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.08,1,.93))
for ext in ['png','pdf']:fig.savefig(O/('restoration-comparison.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
p=O/'generation-audit/evidence/results/trials.json'
if p.exists():
    trials=json.loads(p.read_text());fig,axs=plt.subplots(1,2,figsize=(11,4))
    for label,color in [('public','#674f91'),('signed_and_rms_winner','#156d91'),('random_signed_winner','#aebbc0')]:
        rows=sorted([x for x in trials if x['label']==label and x['recipe']['method']=='residual'],key=lambda x:x['recipe']['dose']);g=[x['recipe']['dose'] for x in rows]
        axs[0].plot(g,[x['mean_caps'] for x in rows],'.-',label=label.replace('_',' '),color=color)
        axs[1].plot(g,[sum(a['assessment']['content'] for a in x['target']+x['protection']) for x in rows],'.-',color=color)
    axs[0].axhline(.85,color='gray',ls=':',lw=1);axs[0].set(xlabel='Residual-relative additive dose g',ylabel='Mean uppercase fraction (2 calibration copies)',ylim=(-.03,1.04))
    axs[1].set(xlabel='Residual-relative additive dose g',ylabel='Correct contents retained / 4',ylim=(-.1,4.1));axs[1].set_yticks(range(5));axs[0].legend(fontsize=8)
    fig.suptitle('Generation-first dose audit: style and content are separate endpoints');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(O/('generation-dose-window.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
p=O/'condolence-rescue/evidence/results/trials.json'
if p.exists():
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    trials=json.loads(p.read_text());fig,axs=plt.subplots(1,3,figsize=(12,4.4));schedules=['all','pulse4','delayed4'];doses=[.2,.4,.6,.8,1.]
    labels=[('signed_and_rms_candidate','PP/RMS candidate L22/N4945'),('public','Public L10/N6295'),('random_signed_winner','Random candidate L31/N5385')]
    cmap=ListedColormap(['#f0f2f3','#c1dbe4','#efc79d','#87cbb6'])
    for ax,(label,title) in zip(axs,labels):
        mat=np.zeros((3,5));texts={}
        for i,sc in enumerate(schedules):
            for j,dose in enumerate(doses):
                r=next(x for x in trials if x['label']==label and x['recipe']['schedule']==sc and x['recipe']['dose']==dose);m=r['metrics'];pr=sum(x['assessment']['content'] for x in r['protection']);mat[i,j]=3 if m['eligible'] else 2 if m['joint']==2 else 1 if m['joint']==1 else 0;texts[i,j]=f"{m['joint']}/2\nP {pr}/2"
        ax.imshow(mat,cmap=cmap,vmin=0,vmax=3,aspect='auto');ax.set_xticks(range(5),doses);ax.set_yticks(range(3),['Continuous','First four','After four *']);ax.set_xlabel('Residual-relative dose g');ax.set_title(title,fontsize=11)
        for (i,j),v in texts.items():ax.text(j,i,v,ha='center',va='center',fontsize=9)
        if label=='signed_and_rms_candidate':ax.add_patch(Rectangle((.5,.5),1,1,fill=False,edgecolor='#143c36',linewidth=2.5))
    fig.suptitle('Calibration dose/timing map: exposure matters');fig.text(.5,.025,'Cell: target marker + required word / 2; P: protected answers / 2. Green passes recorded gates.\n* Delayed protection cases finish before steering starts: these are unexposed controls.\nBoxed recipe was committed before fresh-prompt evaluation; its protection cases receive the pulse.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.20,1,.94))
    for ext in ['png','pdf']:fig.savefig(O/('dose-timing-rescue.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
print('Figures saved')
