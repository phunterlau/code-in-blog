"""Position-aware additive and donor edits; base loader is the audited pilot loader."""
import torch
from model_runtime import Runtime


class ScopedRuntime(Runtime):
    def pre_hook(self,l):
        def hook(mod,args):
            h=args[0]
            if self.capture:
                self.acts[l]=h[0,-1].detach().float().cpu()
                if self.capture_full:self.full_acts[l]=h[0].detach().float().cpu()
            if l not in self.edits:return None
            h=h.clone()
            for edit in self.edits[l]:
                scope=edit.get('scope','last')
                if scope=='all':pos=slice(None)
                elif scope=='continuation':
                    pos=slice(edit['prefix_length']-1,None) if h.shape[1]>1 else slice(None)
                else:pos=-1
                ns=edit['neurons'];values=torch.as_tensor(edit['values'],device=h.device,dtype=h.dtype)
                if edit['operation']=='add':h[:,pos,ns]+=values
                elif edit['operation']=='set':h[:,pos,ns]=values
                elif edit['operation']=='scale':h[:,pos,ns]*=values
                else:raise ValueError(edit)
            return (h,)
        return hook

    @torch.inference_mode()
    def sequence_score(self,prefix,completion,edits=None):
        continuation=self.tok.encode(completion,add_special_tokens=False)
        assert continuation
        labels=torch.tensor(continuation,device='cuda:0')
        ids=torch.cat([prefix,labels[:-1].view(1,-1)],dim=1)
        positions=torch.arange(prefix.shape[1]-1,ids.shape[1],device='cuda:0')
        self.edits=edits or {}
        try:
            logits=self.model(ids,use_cache=False,logits_to_keep=positions).logits[0].float()
            assert logits.is_cuda and logits.shape[0]==len(continuation)
            lp=logits.log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
            assert torch.isfinite(lp).all()
            return {'completion':completion,'token_ids':continuation,'token_logprobs':lp.cpu().tolist(),'logprob':float(lp.sum())}
        finally:self.edits={}
