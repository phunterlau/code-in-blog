import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from model_runtime import Runtime,load_text_model,metadata

def dump(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def gap(z,ti,ci):return float(z[ti].mean()-z[ci].mean())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();here=Path(__file__).parent
 s=read(here/'protocol.json');parent=Path(s['parents']);assert read(parent/'completion.json')['status']=='completed'
 out=Path(a.out);out.mkdir(exist_ok=False);started=time.monotonic();signal.alarm(s['max_seconds']);torch.set_num_threads(4)
 hashes=read(here/'source-hashes.json')
 for name,h in hashes.items():assert sha(here/name)==h
 dump(out/'source-hashes.json',hashes);dump(out/'protocol.json',s)
 main_protocol=read(parent/'protocol.json');discovery=read(parent/'discovery.json');candidates=read(parent/'candidates.json');pub=read(parent/'published-ranks.json')
 dump(out/'parent-hashes.json',{f:sha(parent/f) for f in ['completion.json','protocol.json','discovery.json','candidates.json','published-ranks.json']})
 m,t=load_text_model(s['model_path']);r=Runtime(m,t);r.max_calls=s['max_forward_calls'];dump(out/'runtime.json',metadata(m));result={}
 for name,task in main_protocol['tasks'].items():
    tensors=np.load(parent/(name+'-discovery.npz'),allow_pickle=False);ti,ci=discovery[name]['target_ids'],discovery[name]['control_ids']
    groups={}
    for origin,co in [('public',pub[name]['coordinate'])]+[(k,v['coordinate']) for k,v in candidates[name].items()]:
        key='L%dN%d'%tuple(co)
        if key not in groups:groups[key]={'coordinates':[co],'aliases':[]}
        if origin not in groups[key]['aliases']:groups[key]['aliases'].append(origin)
    groups['pp_top4']={'coordinates':[x['coordinate'] for x in discovery[name]['top50'][:4]],'aliases':['group']}
    rows=[]
    for i,pair in enumerate(task['discovery']):
        baseline={};prefix={}
        for side in ['base','target']:
            prefix[side]=r.ids(pair[side]);z,*_=r.forward(prefix[side]);baseline[side]=gap(z,ti,ci)
        for label,group in groups.items():
            for side,donor in [('base','target'),('target','base')]:
              operations=['donor','zero'] if label!='pp_top4' else ['donor']
              for operation in operations:
                edits={};prediction=0.
                for l,n in group['coordinates']:
                    source=float(tensors[side][i,l,n]);value=float(tensors[donor][i,l,n]) if operation=='donor' else 0.
                    prediction+=float(tensors['coupling'][l,n])*(value-source)
                    edits.setdefault(l,[]).append({'operation':'set','neurons':[n],'values':[value],'all_tokens':False})
                z,*_=r.forward(prefix[side],edits=edits);actual=gap(z,ti,ci)-baseline[side]
                rows.append({'pair':pair['id'],'group':label,'aliases':group['aliases'],'coordinates':group['coordinates'],'parent':side,
                             'operation':operation,'direct_prediction':prediction,'actual_gap_change':actual,
                             'argmax':t.decode([int(z.argmax())]),'natural_gap':baseline['target']-baseline['base']})
    result[name]=rows;dump(out/'patches.json',result)
    summary=[]
    for group in groups:
        x=[row for row in rows if row['group']==group and row['parent']=='base' and row['operation']=='donor']
        summary.append({'group':group,'aliases':groups[group]['aliases'],'mean_direct_prediction':sum(v['direct_prediction'] for v in x)/len(x),
                        'mean_actual_gap_change':sum(v['actual_gap_change'] for v in x)/len(x),'positive_actual':sum(v['actual_gap_change']>0 for v in x),'n':len(x)})
    print(name,json.dumps(summary),flush=True)
 dump(out/'completion.json',{'status':'completed','pid':os.getpid(),'calls':r.calls,'backwards':0,'seconds':time.monotonic()-started,'patches_sha256':sha(out/'patches.json')})
if __name__=='__main__':main()
