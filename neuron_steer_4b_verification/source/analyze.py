"""Offline audit and evidence-qualified summary; no model calls."""
from pathlib import Path
import hashlib,json,re
import numpy as np
from metrics import measure
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    evidence=ROOT/'evidence';out=evidence/'results';s=read(out/'protocol.json');done=read(out/'completion.json')
    assert done['status']=='completed' and done['backwards']==0 and done['calls']<=s['max_forward_calls']
    assert digest(out/'heldout.json')==done['heldout_sha256']
    assert (evidence/'logs/exit-code.txt').read_text().strip()=='0'
    telemetry=(evidence/'logs/gpu.txt').read_text();pid=read(out/'runtime.json')['pid'];assert str(pid)+',' in telemetry
    assert read(out/'runtime.json')['transformer_parameters_all_cuda']
    for name,h in read(out/'source-hashes.json').items():assert digest(ROOT/'source'/name)==h,name
    disc=read(out/'discovery.json');screen=read(out/'screens.json');candidates=read(out/'candidates.json');choices=read(out/'choices.json')
    cal=read(out/'calibration.json');cb=read(out/'calibration-baseline.json');held=read(out/'heldout.json');ranks=read(out/'published-ranks.json')
    for fn,seal in [('discovery.json','discovery-seal.json'),('candidates.json','candidate-seal.json'),('choices.json','selection-seal.json')]:assert digest(out/fn)==read(out/seal)['sha256']
    assert read(out/'discovery-seal.json')['time']<=read(out/'candidate-seal.json')['time']<=read(out/'selection-seal.json')['time']
    known={};sources=[]
    def add(co,source):known.setdefault(tuple(co),[]).append(source)
    workspace=ROOT.parents[1]
    for f,key in [('neuronary-inventory.json','recipes'),('peft-preset-index.json','presets')]:
        path=workspace/'outputs/qwen-neuron-inventory-review'/f
        if not path.exists():continue
        sources.append({'file':str(path.relative_to(workspace)),'sha256':digest(path)})
        for row in read(path)[key]:
            if 'Qwen3.5-4B' not in row.get('model',''):continue
            for n in row.get('neurons',[]):add([n['layer'],n['neuron']],row.get('url',row.get('name',f)))
    method=workspace/'work/references/sambhav-neuron-explorer/explorer/agent_methodology.md'
    if method.exists():
        sources.append({'file':str(method.relative_to(workspace)),'sha256':digest(method)})
        for l,n in re.findall(r'L(\d+)[ /._-]*N(\d+)',method.read_text()):add([int(l),int(n)],'source methodology')
    history=workspace/'outputs/peft-comment-6577425151-review/repository-audit/historical-qwen35-4b-names.json'
    if history.exists():
        sources.append({'file':str(history.relative_to(workspace)),'sha256':digest(history)})
        for co in read(history):
            match=re.search(r'L(\d+)/N(\d+)',co)
            if match:add([int(match[1]),int(match[2])],'historical source labels')
    def validate(items,rows):
        assert len(items)==len(rows)
        for item,row in zip(items,rows):
            assert item['id']==row['id']
            assert row['metrics']==measure(item,row['text'],row['capped'])
            assert row['calls']==len(row['token_ids'])
            for w in row['writes']:
                assert float(np.float32(w['after'])-np.float32(w['before']))==w['actual_last_offset']
    def facts(items,rows):
        zero=[r for i,r in zip(items,rows) if i.get('expected')==0]
        nonzero=[r for i,r in zip(items,rows) if isinstance(i.get('expected'),int) and i['expected']!=0]
        return {'n':len(rows),'correct':sum(r['metrics']['correct'] for r in rows),'joint':sum(r['metrics']['joint'] for r in rows),
            'trait':sum(r['metrics']['trait'] for r in rows),'capped':sum(r['capped'] for r in rows),
            'zero_correct':sum(r['metrics']['correct'] for r in zero),'zero_n':len(zero),
            'nonzero_correct':sum(r['metrics']['correct'] for r in nonzero),'nonzero_n':len(nonzero)}
    result={};rawcal_calls=rawheld_calls=0
    for name,task in s['tasks'].items():
        z=np.load(out/(name+'-discovery.npz'),allow_pickle=False)
        recomputed=((z['target']-z['base'])*z['coupling']).mean(0)
        assert np.allclose(recomputed,z['signed'],rtol=2e-6,atol=2e-6)
        order=np.argsort(-z['signed'].ravel(),kind='stable')[:50]
        assert [list(divmod(int(i),9216)) for i in order]==[x['coordinate'] for x in disc[name]['top50']]
        ppco={tuple(x['coordinate']) for x in disc[name]['top50'][:s['pp_candidates']]}
        assert tuple(candidates[name]['pp']['coordinate']) in ppco
        validate(task['calibration'],cb[name]['target']);validate(s['protections_cal'],cb[name]['protection'])
        rawcal_calls+=sum(r['calls'] for r in cb[name]['target']+cb[name]['protection'])
        validate(task['heldout'],held[name]['baseline']['target']);validate(s['protections_held'],held[name]['baseline']['protection'])
        rawheld_calls+=sum(r['calls'] for r in held[name]['baseline']['target']+held[name]['baseline']['protection'])
        result[name]={'published_rank':ranks[name],'baseline':facts(task['heldout'],held[name]['baseline']['target']),
            'baseline_protections':facts(s['protections_held'],held[name]['baseline']['protection']),'arms':{}}
        for origin,arm in held[name]['arms'].items():
            for row in cal[name][origin]:
                validate(task['calibration'],row['target']);validate(s['protections_cal'],row['protection'])
                rawcal_calls+=sum(r['calls'] for r in row['target']+row['protection'])
            summaries={}
            for label in ['selected','challenger']:
                h=arm[label]
                if 'alias'in h:summaries[label]={'alias':h['alias']};continue
                validate(task['heldout'],h['target']);validate(s['protections_held'],h['protection'])
                assert h['recipe']==choices[name][origin][label]
                if h['recipe'] is not None:rawheld_calls+=sum(r['calls'] for r in h['target']+h['protection'])
                summaries[label]={'recipe':h['recipe'],'target':facts(task['heldout'],h['target']),
                                  'protection':facts(s['protections_held'],h['protection']), 'gate':h['metrics']}
            co=candidates[name][origin]['coordinate'];score=z['signed'][tuple(co)]
            result[name]['arms'][origin]={'coordinate':co,'signed_pp_rank':int((z['signed']>score).sum())+1,
                'in_checked_sources':known.get(tuple(co),[]),'calibration_choice':choices[name][origin],**summaries}
    # Counts reflect actual executed rows; no-edit and explicit alias arms were not regenerated.
    forward_expected=1+sum(len(t['discovery'])*2 for t in s['tasks'].values())+len(s['protections_cal'])+len(s['tasks'])*2
    forward_expected+=sum(len(rows)*6 for rows in screen.values())+rawcal_calls+rawheld_calls
    assert forward_expected==done['calls'],(forward_expected,done['calls'])
    report={'status':'passed','completion':done,'cuda_pid':pid,'telemetry_pid_matches':telemetry.count(str(pid)+','),
            'checked_known_coordinate_count':len(known),'novelty_sources':sources,'tasks':result,
            'scope':'single checkpoint, FP16 transformer and CPU FP32 vocabulary boundary, greedy decoding; source absence is not global novelty; marker effects do not establish semantic control'}
    dump(ROOT/'analysis.json',report)
    print(json.dumps({'status':'passed','calls':done['calls'],'cuda_pid':pid,'tasks':result},indent=2))
if __name__=='__main__':main()
