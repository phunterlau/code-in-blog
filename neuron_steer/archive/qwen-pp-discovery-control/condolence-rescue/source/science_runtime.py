"""Audited inference loader plus observational FFN diagnostics and matched gradients."""
import gc,time
from collections import defaultdict
import torch
from scoped_runtime import ScopedRuntime


class ScienceRuntime(ScopedRuntime):
    def __init__(self,model,tok):
        self.detail=False;self.grad_mode=False;self.grads={};self.gradient_hook_counts=defaultdict(int);self.layer_calls=[0]*32;self.backwards=0;self.detail_data={}
        super().__init__(model,tok)
        for l,layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self.layer_pre(l)))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self.skip_pre(l)))
            self.handles.append(layer.mlp.register_forward_hook(self.ffn_post(l)))
            self.handles.append(layer.register_forward_hook(self.detail_post(l)))

    def layer_pre(self,l):
        def f(mod,args):
            self.layer_calls[l]+=1
            if self.detail:self.detail_data.setdefault(l,{})['incoming']=args[0][0,-1].detach().float().cpu()
        return f

    def skip_pre(self,l):
        def f(mod,args):
            if self.detail:self.detail_data.setdefault(l,{})['pre_ffn_skip']=args[0][0,-1].detach().float().cpu()
        return f

    def ffn_post(self,l):
        def f(mod,args,out):
            if self.detail:self.detail_data.setdefault(l,{})['ffn']=out[0,-1].detach().float().cpu()
        return f

    def detail_post(self,l):
        def f(mod,args,out):
            if self.detail:
                h=(out[0] if isinstance(out,tuple) else out)[0].detach().float();rn=h.norm(dim=-1);pr=rn.pow(4)/h.pow(4).sum(-1)
                self.detail_data.setdefault(l,{})['out_last']=h[-1].cpu();self.detail_data[l]['norms']=rn.cpu();self.detail_data[l]['pr']=pr.cpu()
        return f

    def pre_hook(self,l):
        base=super().pre_hook(l)
        def f(mod,args):
            h=args[0]
            if self.grad_mode and h.requires_grad:
                def grad_hook(g):
                    self.grads[l]=g[0,-1].detach().float().cpu()/self.loss_scale;self.gradient_hook_counts[l]+=1
                h.register_hook(grad_hook)
            return base(mod,args)
        return f

    def observed(self,ids,capture=True):
        self.detail=True;self.detail_data={}
        try:
            out=self.forward(ids,capture=capture);return out,self.detail_data
        finally:self.detail=False

    def gradient(self,ids,direction,cpu_saved=False):
        """dF/da at the same last-token post-SwiGLU coordinates as direct probing."""
        started=time.monotonic();before=list(self.layer_calls);self.grads={};self.gradient_hook_counts.clear();self.loss_scale=getattr(self,"requested_loss_scale",1.);self.grad_mode=True
        self.model.requires_grad_(False)
        assert self.model.config.attention_dropout==0.
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});self.model.train()
        try:
            self.count_call(self.model,())
            context=torch.autograd.graph.save_on_cpu(pin_memory=True) if cpu_saved else __import__('contextlib').nullcontext()
            with torch.enable_grad(),context:
                emb=self.model.model.embed_tokens(ids).detach().requires_grad_(True)
                hidden=self.model.model(inputs_embeds=emb,use_cache=False).last_hidden_state[0,-1]
                score=(hidden.float()*direction.to('cuda:0')).sum();(score*self.loss_scale).backward();self.backwards+=1
            assert set(self.grads)==set(range(32)),sorted(self.grads)
            grads=torch.stack([self.grads[l] for l in range(32)])
            if not torch.isfinite(grads).all():
                raise RuntimeError('nonfinite gradients: '+str({l:{'finite':int(torch.isfinite(grads[l]).sum()),'max_finite':float(grads[l][torch.isfinite(grads[l])].abs().max()) if torch.isfinite(grads[l]).any() else None} for l in range(32)}))
            result={'score':float(score.detach()),'seconds':time.monotonic()-started,'layer_evaluations':[x-y for x,y in zip(self.layer_calls,before)],'hook_counts':dict(self.gradient_hook_counts),'cpu_saved':cpu_saved,'max_abs_gradient':float(grads.abs().max()),'nonzero_gradients':int(grads.count_nonzero()),'loss_scale':self.loss_scale}
            return grads,result
        finally:
            self.grad_mode=False;self.model.eval();self.model.gradient_checkpointing_disable();self.model.zero_grad(set_to_none=True)
            self.grads={};gc.collect();torch.cuda.empty_cache()


def observable(model,tok,target,control):
    def ids(words):
        out=set()
        for w in words:
            for s in [w,' '+w]:
                ids=tok.encode(s,add_special_tokens=False)
                if len(ids)==1:out.add(ids[0])
        return sorted(out)
    ti,ci=ids(target),ids(control);assert ti and ci and not set(ti)&set(ci)
    W=model.lm_head.weight.detach();d=W[ti].float().mean(0)-W[ci].float().mean(0)
    coupling=torch.stack([(layer.mlp.down_proj.weight.detach().float().T@d.to('cuda:0')).cpu() for layer in model.model.layers])
    return ti,ci,d,coupling


def gap(logits,ti,ci):return float(logits[ti].mean()-logits[ci].mean())


def ffn_diagnostic(detail,direction):
    x=detail[31];f=float(x['ffn']@direction);s=float(x['incoming']@direction);p=float(x['pre_ffn_skip']@direction)
    return {'ffn_projection':f,'incoming_skip_projection':s,'pre_ffn_skip_projection':p,'ffn_to_incoming_skip':abs(f)/max(abs(s),1e-8),'ffn_to_pre_ffn_skip':abs(f)/max(abs(p),1e-8)}
