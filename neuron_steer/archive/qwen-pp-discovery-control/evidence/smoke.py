import json,torch
from run_experiment import Runtime
from model_runtime import load_text_model,metadata
torch.set_num_threads(4)
s=json.load(open('protocol.json'));m,t=load_text_model(s['model_path']);r=Runtime(m,t);r.max_calls=20
ids=r.ids([{'role':'user','content':'Output only the word red.'}]);z,a,rr,_=r.forward(ids,capture=True);r.scales={l:float(v.median()) for l,v in rr.items()};r.prepare([1,8396])
for method,value in [('residual',0.),('scale',1.)]:
    q,ledger=r.controlled(ids,{'layer':1,'neuron':8396,'method':method,'value':value});assert torch.equal(z,q),(method,float((z-q).abs().max()))
q,ledger=r.controlled(ids,{'layer':1,'neuron':8396,'method':'clamp','value':float(a[1][8396])});assert torch.isfinite(q).all() and ledger
g=r.generate_control({'id':'smoke','prompt':'Output only the word red.','expected':'red'},None,4,'caps');assert g['text'].strip()=='red',g
print(json.dumps({'status':'passed','calls':r.calls,'metadata':metadata(m),'text':g['text']}),flush=True)
