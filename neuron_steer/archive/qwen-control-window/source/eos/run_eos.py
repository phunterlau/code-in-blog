import argparse,hashlib,json,os,signal,time
from pathlib import Path
import torch
from transformers import LogitsProcessorList,SuppressTokensLogitsProcessor
from model_runtime import load_text_model,metadata
from scoped_runtime import ScopedRuntime
from run_window import dump,screen


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();raw=Path(a.protocol).read_bytes();s=json.loads(raw);out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic();signal.alarm(s['budgets']['max_seconds']);torch.set_num_threads(4);torch.manual_seed(s['seed']);dump(out/'protocol.json',s);dump(out/'start.json',{'pid':os.getpid(),'time':time.time(),'protocol_sha256':hashlib.sha256(raw).hexdigest()});model,tok=load_text_model(s['model_path']);r=ScopedRuntime(model,tok);r.max_calls=s['budgets']['max_forward_calls'];dump(out/'runtime.json',metadata(model));l,n=s['layer'],s['neuron'];v=r.layers[l].mlp.down_proj.weight[:,n].detach().float();vn=float(v.norm());cal=tok(s['calibration_text'],add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0');_,_,rr,_=r.forward(cal,capture=True);b=float(rr[l].median())/vn;dump(out/'calibration.json',{'B':b,'column_norm':vn,'layer':l,'neuron':n,'reference_residual_norms':rr[l].tolist()});rows=[];dist=[];vectors={};capture={'enabled':False,'key':None}
    def geom(mod,args,out):
        if capture['enabled']:vectors[capture['key']]=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float().cpu()
    handle=r.layers[l].register_forward_hook(geom)
    for p in s['prompts']:
        ids=r.ids([{'role':'user','content':p['text']}])
        for g in s['relative_doses']:
            cid=f"L{l}N{n}_g{g:g}";ed={} if g==0 else {l:[{'neurons':[n],'operation':'add','values':[g*b],'scope':'all'}]};capture.update(enabled=True,key=p['id']+'_'+str(g));logits,_,_,_=r.forward(ids,edits=ed);capture['enabled']=False;probs=logits.softmax(-1);top=probs.topk(8);eos=tok.eos_token_id
            dist.append({'condition':cid,'g':g,'prompt_id':p['id'],'eos_id':eos,'eos_token':tok.decode([eos]),'eos_probability':float(probs[eos]),'eos_rank':int((logits>logits[eos]).sum())+1,'entropy_nats':float(-(probs*logits.log_softmax(-1)).sum()),'top':[{'id':int(i),'text':tok.decode([int(i)]),'probability':float(pr)} for i,pr in zip(top.indices,top.values)]})
            for policy in s['policies']:
                print('condition',cid,p['id'],policy,'calls',r.calls,flush=True)
                if policy=='ordinary':gen=r.generate(ids,edits=ed,max_new_tokens=s['max_new_tokens'])
                else:
                    r.edits=ed
                    try:
                        with torch.inference_mode():
                            seq=model.generate(ids,do_sample=False,max_new_tokens=s['max_new_tokens'],use_cache=True,pad_token_id=eos,eos_token_id=eos,logits_processor=LogitsProcessorList([SuppressTokensLogitsProcessor([eos],device='cuda:0')]))
                        assert seq.is_cuda;new=seq[0,ids.shape[1]:].tolist();gen={'text':tok.decode(new,skip_special_tokens=True),'token_ids':new,'length_capped':len(new)==s['max_new_tokens']};assert eos not in new
                    finally:r.edits={}
                rows.append({'condition':cid,'g':g,'prompt_id':p['id'],'prompt':p['text'],'policy':policy,**gen,**screen(gen['text'],'alien')});dump(out/'generations.json',rows);dump(out/'distributions.json',dist)
    handle.remove();torch.save({'vectors':vectors,'v':v.cpu(),'B':b},out/'residual-vectors.pt');geometry=[]
    for p in s['prompts']:
        base=vectors[p['id']+'_0.0'];vc=v.cpu();rn=float(base.norm());c0=float(base@vc)/(rn*vn)
        for g in s['relative_doses']:
            actual=vectors[p['id']+'_'+str(g)];t=g*b;pred=base+t*vc;gc=t*vn/rn;cp=(c0+gc)/(1+2*c0*gc+gc*gc)**.5;ca=float(actual@vc)/(float(actual.norm())*vn);geometry.append({'prompt_id':p['id'],'g_reference':g,'g_context':gc,'baseline_residual_norm':rn,'c0':c0,'predicted_c':cp,'actual_c':ca,'cosine_error':ca-cp,'relative_vector_error':float((actual-pred).norm())/float(pred.norm())})
    dump(out/'geometry.json',geometry);dump(out/'completion.json',{'pid':os.getpid(),'calls':r.calls,'seconds':time.monotonic()-start,'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'finished':time.time()});print('complete',r.calls,flush=True)

if __name__=='__main__':main()
