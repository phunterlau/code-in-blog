"""Static figures from the completed audited group experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
    data=json.loads((out/'analysis.json').read_text())['categories']
    verification=json.loads((out/'verification.json').read_text())
    assert verification['passed'] and not verification.get('synthetic_fixture',False)
    dest=out/'figures';dest.mkdir(exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    arms=['pp1','pp4','pp16'];colors=['#8795a5','#4a8f79','#256ba1']
    for ax,(cat,values) in zip(axs.flat,data.items()):
        for j,(side,dose) in enumerate([('base',.1),('base',.4),('target',.1),('target',.4)]):
            for k,(arm,color) in enumerate(zip(arms,colors)):
                y=np.array(values['arms'][f'{side}:{dose}'][arm]['values']);x=j+(k-1)*.23
                ax.scatter(np.full(8,x)+np.linspace(-.04,.04,8),y,color=color,s=12,alpha=.55)
                ax.plot(x,float(y.mean()),marker='_',color=color,markersize=13,markeredgewidth=2,label=arm if j==0 else None)
        ax.axhline(0,color='#999999',lw=.8);ax.set_title(cat.replace('_',' ').capitalize())
        ax.set_xticks(range(4),['Base +.1','Base +.4','Target −.1','Target −.4'],fontsize=8)
        ax.set_ylabel('Oriented lexical readout effect')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Equal block-L2 write budget: singleton versus PP groups')
    fig.supxlabel('Dots: eight authored prompts per category; bars: means. Positive means the intended causal direction. Not a generation-quality test.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(dest/f'group-effects.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,6.5),constrained_layout=True)
    for ax,(cat,values) in zip(axs.flat,data.items()):
        for side,color in [('base','#256ba1'),('target','#b16836')]:
            counts=[values['deletions'][side][f'drop{i:02}_matched']['effect']['over_005'] for i in range(1,17)]
            ax.plot(range(1,17),counts,'o-',markersize=3,lw=1,color=color,label=side)
        ax.set_ylim(-.4,8.6);ax.set_xticks([1,4,8,12,16]);ax.set_yticks([0,2,4,6,8]);ax.set_title(cat.replace('_',' ').capitalize())
        ax.set_xlabel('Removed PP rank');ax.set_ylabel('Effects > 0.05 logits / 8')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Deletion stress test at g=0.4: remaining directions rescaled to the same budget')
    fig.supxlabel('Rescaling tests whether the remaining coordinates retain control capacity; it does not isolate unscaled causal necessity.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(dest/f'deletion-stability.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
