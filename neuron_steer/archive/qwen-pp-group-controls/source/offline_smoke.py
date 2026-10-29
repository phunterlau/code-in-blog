"""Synthetic audit integration and tamper rejection; never calls a model."""
import hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from geometry import key,scale,norms,make_vectors,causal_conditions,protection_conditions


def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source=Path(__file__).resolve().parent
    root=Path(tempfile.mkdtemp(prefix='pp-group-synthetic-',dir='/private/tmp' if Path('/private/tmp').exists() else None))
    mirror=root/'mirror';mirror.mkdir();result=mirror/'results';result.mkdir();(mirror/'logs').mkdir()
    for p in list(source.glob('*.py'))+[source/'protocol.json',source/'job.sh']:shutil.copy2(p,mirror/p.name)
    hashes={p.name:digest(p) for p in mirror.iterdir() if p.is_file()}
    write(result/'source-hashes.json',hashes)
    s=json.loads((source/'protocol.json').read_text());write(result/'protocol.json',s)
    rng=np.random.default_rng(42)
    coordinates={tuple(co) for c in s['categories'].values() for coords in [c['pp16']]+c['random16'] for co in coords}
    weights={key(co):rng.normal(size=8) for co in sorted(coordinates)};references={l:float(l+1) for l,n in coordinates}
    vectors={cat:make_vectors(c,weights,references) for cat,c in s['categories'].items()}
    write(result/'vectors.json',vectors);write(result/'reference-scales.json',references)
    np.savez_compressed(result/'projection-columns.npz',**weights)
    rows=[]
    def add(cat,kind,item,side,sign,arm,dose):
        v={} if arm=='baseline' else scale(vectors[cat][arm],sign*dose)
        writes=[{'coordinate':k,'before':0.,'after':float(np.float16(x)),'requested_delta':x,'actual_delta':float(np.float16(x)),'sequence_positions':1,'edited_positions':1} for k,x in v.items()]
        actual={w['coordinate']:w['actual_delta'] for w in writes}
        ids=s['categories'][cat]['observable']
        ti=ids['target_ids'] if kind=='causal' else [100 if s['protections'][item]['expected']=='A' else 101]
        ci=ids['control_ids'] if kind=='causal' else [101 if ti==[100] else 100]
        value=float(np.float32(0. if kind=='causal' and arm=='baseline' else sign*dose if kind=='causal' else 1.))
        tl=[value]*len(ti);cl=[0.]*len(ci)
        row={'kind':kind,'category':cat,'item':item,'side':side,'sign':sign,'arm':arm,'dose':dose,'record_number':len(rows)+1,
             'input_ids':[1],'vector':v,'writes':writes,'requested_norms':norms(v,weights,references),'actual_norms':norms(actual,weights,references),
             'target_ids':ti,'control_ids':ci,'target_logits':tl,'control_logits':cl,'readout':float(np.asarray(tl,dtype=np.float32).mean()),
             'argmax':ti[0],'argmax_logit':value,'kl_baseline_to_edited':0.}
        if kind=='protection':row.update(expected_token=ti[0],correct_first_token=True)
        rows.append(row)
    for cat in s['categories']:
        for i in range(8):
            for side,sign in [('base',1),('target',-1)]:
                add(cat,'causal',i,side,sign,'baseline',0.)
                for arm,dose in causal_conditions():add(cat,'causal',i,side,sign,arm,dose)
        for i in range(8):
            add(cat,'protection',i,'neutral',0,'baseline',0.)
            for arm,dose in protection_conditions():add(cat,'protection',i,'neutral',1 if dose>0 else -1,arm,abs(dose))
    (result/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(result/'start.json',{'time':2.,'synthetic_fixture':True})
    write(result/'portable-completion-receipt.json',{'status':'completed_and_reviewed','created_at':1.,'protocol_sha256':'31c4ed50e358fd26861b5374035f14d5e4b5d08968b1ad6415deb05d9503eb85'})
    write(result/'protection-tokenization.json',{'A':[100],'B':[101]})
    write(result/'smoke.json',{'passed':True})
    write(result/'completion.json',{'status':'completed','calls':4899,'records':len(rows),'pid':99999999,'seconds':1.,'backwards':0})
    write(result/'runtime.json',{'pid':99999999,'transformer_parameters_all_cuda':True})
    (mirror/'logs/groups.exitcode').write_text('0\n')
    (mirror/'logs/groups.telemetry').write_text('99999999, SYNTHETIC FIXTURE\n99999999, SYNTHETIC FIXTURE\n')
    def seal():write(result/'final-seal.json',{'files':{p.name:digest(p) for p in result.iterdir() if p.is_file() and p.name!='final-seal.json'}})
    seal();cmd=[sys.executable,str(source/'analyze.py'),'--workspace',str(mirror),'--source',str(mirror),'--output',str(root/'analysis')]
    rejected_without_flag=subprocess.run(cmd,capture_output=True,text=True)
    assert rejected_without_flag.returncode!=0
    good=subprocess.run(cmd+['--allow-synthetic-fixture'],capture_output=True,text=True)
    assert good.returncode==0,good.stderr
    assert json.loads((root/'analysis/verification.json').read_text())['passed']
    # A valid file hash alone must not hide a corrupt write ledger.
    row=next(r for r in rows if r['writes']);row['writes'][0]['actual_delta']+=1
    (result/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));seal()
    bad=subprocess.run(cmd+['--allow-synthetic-fixture'],capture_output=True,text=True)
    assert bad.returncode!=0
    checks={'passed':True,'model_calls':0,'synthetic_only':True,'fixture_requires_explicit_flag':True,'full_synthetic_audit_and_report':True,
            'resealed_corrupt_write_rejected':True,'temporary_fixture':str(root)}
    write(source/'offline-smoke.json',checks);print(json.dumps(checks))


if __name__=='__main__':main()
