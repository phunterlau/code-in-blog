import json
from pathlib import Path
import matplotlib.pyplot as plt
R=Path.cwd()/'outputs/qwen-control-window'
def read(p):return json.loads(p.read_text())
rows=read(R/'evidence/main-v1/results/summary.json')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
fig,axs=plt.subplots(2,2,figsize=(10,7.3),sharey=True)
for ax,target,coord in zip(axs.flat[:3],['human','building','corporation'],['L7 N6297 (+)','L3 N3066 (-)','L4 N4783 (-)']):
    rr=sorted([x for x in rows if x['phase']=='calibration' and x['kind']=='target' and x['target']==target],key=lambda x:abs(x['g']));x=[abs(z['g']) for z in rr]
    ax.plot(x,[z['target_hits'][target]/z['n'] for z in rr],'o-',c='#a33f1f',label='Coherent literal identity (n=2)')
    ax.plot(x,[z['facts_correct']/z['facts_n'] for z in rr],'s-',c='#1e6caa',label='Exact answers (n=3)')
    ax.plot(x,[z['coherent']/z['n'] for z in rr],'.--',c='#777777',label='Lexical coherence (n=2)')
    ax.set_title(f'{target.capitalize()}: {coord}');ax.set_xlabel('Relative dose magnitude |g| = |offset| / B');ax.set_ylim(-.05,1.08);ax.set_ylabel('Fraction');ax.set_xticks([.1,.25,.4,.55,.7,1]);ax.grid(alpha=.15)
ax=axs.flat[3];rr=sorted([x for x in rows if x['phase']=='heldout' and x['kind']=='anchor'],key=lambda x:x['g']);x=[z['g'] for z in rr]
ax.plot(x,[z['target_hits']['alien']/z['n'] for z in rr],'o-',c='#a33f1f',label='Coherent literal identity (n=4)');ax.plot(x,[z['facts_correct']/z['facts_n'] for z in rr],'s-',c='#1e6caa',label='Exact answers (n=12)');ax.plot(x,[z['coherent']/z['n'] for z in rr],'.--',c='#777777',label='Lexical coherence (n=4)');ax.set_title('Alien: L22 N2742 (+), fresh held-out prompts');ax.set_xlabel('Relative dose g');ax.set_xticks(x);ax.grid(alpha=.15);ax.legend(fontsize=8,loc='lower left');axs.flat[0].legend(fontsize=8,loc='lower left')
fig.suptitle('Different sampled doses expose different failures',fontsize=15);fig.text(.5,.015,'Pinned Qwen3.5-4B; FP16 CUDA transformer / FP32 CPU vocabulary. Exact answers include formatting.\nLines join tested points; lexical coherence is a screen, not semantic or task-quality validation.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.96])
for ext in ['png','pdf','svg']:fig.savefig(R/f'dose-windows.{ext}',bbox_inches='tight')
plt.close(fig)
rows=read(R/'evidence/followups-v1/results/summary.json');fig,axs=plt.subplots(1,2,figsize=(10.5,4.5))
ax=axs[0]
for l,c in zip([0,1,2],['#1e6caa','#a33f1f','#5d8131']):
 vals=[]
 for dose in [.1,.3,.8]:
  rr=[x for x in rows if x['study']=='early' and x['layer']==l and x['g']==dose];vals.append(sum(x['coherent'] for x in rr)/sum(x['generation_n'] for x in rr))
 ax.plot([.1,.3,.8],vals,'o-',c=c,label=f'Layer {l} (2 random neurons, 4 outputs/dose)')
ax.set_title('Early layers: does a lower dose restore text?');ax.set_xlabel('Positive relative dose g');ax.set_ylabel('Fraction passing lexical coherence screen');ax.set_ylim(-.05,1.08);ax.set_xticks([.1,.3,.8]);ax.legend(fontsize=8);ax.grid(alpha=.15)
ax=axs[1]
for j,(kind,label,col) in enumerate([('target','L3 N3066','#a33f1f'),('random0','Random L3 N5327','#666666'),('random1','Random L3 N2007','#aaaaaa')]):
 rr=sorted([x for x in rows if x['study']=='heraldry' and x.get('kind')==kind and x['g'] in [-.4,-.55]],key=lambda x:abs(x['g']));ax.bar([k+(j-1)*.24 for k in range(2)],[x['coherent_heraldry']/x['generation_n'] for x in rr],width=.22,label=label,color=col)
ax.set_xticks([0,1],['-0.40','-0.55']);ax.set_xlabel('Signed relative dose g');ax.set_ylabel('Frozen lexical + coherence screen / 6');ax.set_ylim(-.05,1.08);ax.set_title('Fresh test of the heraldry hypothesis');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
fig.suptitle('Repairing the setting and checking what was recovered',fontsize=14);fig.text(.5,.01,'Fixed follow-up protocols; all-token additive edits. Restored text does not establish a semantic controller.\nHeraldry is a new hypothesis from calibration, tested here on six different prompts; inspect task relevance separately.',ha='center',fontsize=8.5);fig.tight_layout(rect=[0,.08,1,.94])
for ext in ['png','pdf','svg']:fig.savefig(R/f'followup-windows.{ext}',bbox_inches='tight')
