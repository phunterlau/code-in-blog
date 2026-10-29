"""Offline provenance, budget, search fairness, arithmetic, and execution checks."""
import hashlib,json,math,re,statistics,struct,zipfile
from pathlib import Path

ROOT=next(p/'outputs/qwen-control-window' for p in Path(__file__).resolve().parents if (p/'outputs/qwen-control-window').is_dir())
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return re.sub('[^a-z0-9]','',s.lower())
def validate_run(folder,source):
    p=ROOT/'evidence'/folder;d=p/'results';s=read(d/'protocol.json');start=read(d/'start.json');end=read(d/'completion.json');runtime=read(d/'runtime.json');conditions=read(d/'conditions.json');gs=read(d/'generations.json');fs=read(d/'facts.json');geom=read(d/'geometry.json');logs=p/'logs'
    assert sha(source/'protocol.json')==start['protocol_sha256']==sha(d/'protocol.json')
    hashes=read(source/'source-hashes.json')
    for name,h in hashes.items():assert sha(source/name)==h,(folder,name)
    assert (logs/'run.exitcode').read_text().strip()=='0'
    assert start['pid']==end['pid']==runtime['pid']==int((logs/'run.pid').read_text())
    assert runtime['transformer_parameters_all_cuda'] and runtime['device']=='cuda:0'
    assert end['calls']<=s['budgets']['max_forward_calls'] and end['seconds']<=s['budgets']['max_seconds']
    samples=len(re.findall(r'^'+str(end['pid'])+r',', (logs/'run.telemetry').read_text(),re.M));assert samples>=3
    stderr=(logs/'run.stderr').read_text();assert 'Traceback' not in stderr and 'OutOfMemoryError' not in stderr
    for f in fs:assert f['correct']==(norm(f['text']) in [norm(x) for x in f['expected']])
    for c in conditions:
        if c['layer'] is None:assert c['offset']==0;continue
        match=next(g for g in geom.values() if g['layer']==c['layer'] and g['neuron']==c['neuron'])
        assert math.isclose(c['B'],match['B'],rel_tol=1e-12)
        assert math.isclose(c['offset']*match['column_norm']/sorted(match['residual_norms'])[(len(match['residual_norms'])-1)//2],c['g'],rel_tol=1e-6)
    assert len({c['id'] for c in conditions})==len(conditions)
    seen=set()
    for g in gs:
        key=(g.get('phase'),g['condition'],g['prompt_id']);assert key not in seen;seen.add(key)
        assert len(g['token_ids'])>0
    return {'pid':end['pid'],'calls':end['calls'],'seconds':end['seconds'],'pid_matched_samples':samples,'conditions':len(conditions),'generations':len(gs),'exact_answer_trials':len(fs),'runtime_versions':runtime['versions'],'reference_kernel_notices':stderr.count('reference PyTorch')},s,conditions,gs,fs


def validate_eos():
    p=ROOT/'evidence/eos-v1';d=p/'results';source=ROOT/'source/eos';spec=read(d/'protocol.json');start=read(d/'start.json');end=read(d/'completion.json');rt=read(d/'runtime.json');logs=p/'logs'
    assert start['protocol_sha256']==sha(source/'protocol.json')==sha(d/'protocol.json')
    for name,h in read(source/'source-hashes.json').items():assert sha(source/name)==h
    assert (logs/'run.exitcode').read_text().strip()=='0' and rt['transformer_parameters_all_cuda']
    assert start['pid']==end['pid']==rt['pid']==int((logs/'run.pid').read_text())
    samples=len(re.findall(r'^'+str(end['pid'])+r',',(logs/'run.telemetry').read_text(),re.M));assert samples>=3
    assert end['calls']<=spec['budgets']['max_forward_calls'] and end['seconds']<=spec['budgets']['max_seconds']
    assert 'Traceback' not in (logs/'run.stderr').read_text()
    # Own torch archive: six ordered float32 residual vectors and one write vector.
    # Their insertion order is fixed by protocol prompt/dose order in run_eos.py.
    with zipfile.ZipFile(d/'residual-vectors.pt') as z:
        names={int(re.search(r'/data/(\d+)$',n).group(1)):n for n in z.namelist() if re.search(r'/data/(\d+)$',n)}
        assert set(names)==set(range(7));arrays=[struct.unpack('<2560f',z.read(names[i])) for i in range(7)]
        assert z.read(next(n for n in z.namelist() if n.endswith('/byteorder')))==b'little'
    v=arrays[-1];vn=math.sqrt(sum(t*t for t in v));B=read(d/'calibration.json')['B'];geometry=read(d/'geometry.json');errors=[]
    for i,prompt in enumerate(spec['prompts']):
        base=arrays[i*3];rn=math.sqrt(sum(t*t for t in base));c0=sum(a*b for a,b in zip(base,v))/(rn*vn)
        for j,g in enumerate(spec['relative_doses']):
            actual=arrays[i*3+j];an=math.sqrt(sum(t*t for t in actual));ca=sum(a*b for a,b in zip(actual,v))/(an*vn);gc=g*B*vn/rn;cp=(c0+gc)/math.sqrt(1+2*c0*gc+gc*gc);row=next(x for x in geometry if x['prompt_id']==prompt['id'] and x['g_reference']==g)
            assert abs(row['actual_c']-ca)<1e-6 and abs(row['predicted_c']-cp)<1e-6
            pred=[a+g*B*b for a,b in zip(base,v)];ve=math.sqrt(sum((a-b)**2 for a,b in zip(actual,pred)))/math.sqrt(sum(a*a for a in pred));assert abs(ve-row['relative_vector_error'])<1e-6 and ve<.002
            errors.append(abs(ca-cp))
    gs=read(d/'generations.json');ds=read(d/'distributions.json');assert len(gs)==12 and len(ds)==6
    for row in gs:
        dist=next(x for x in ds if x['prompt_id']==row['prompt_id'] and x['g']==row['g'])
        if row['policy']=='ordinary':assert row['token_ids'][0]==dist['top'][0]['id']
        else:assert dist['eos_id'] not in row['token_ids']
        assert len(row['token_ids'])<=spec['max_new_tokens']
    return {'pid':end['pid'],'calls':end['calls'],'seconds':end['seconds'],'pid_matched_samples':samples,'generations':len(gs),'first_token_distributions':len(ds),'max_cosine_prediction_error_from_raw_vectors':max(errors),'raw_residual_vectors_recomputed_locally':True}


def main():
    main,s,cs,gs,fs=validate_run('main-v1',ROOT/'source');p=ROOT/'evidence/main-v1/results';choices=read(p/'selection.json');assert sha(p/'selection.json')==read(p/'selection_commitment.json')['sha256'];assert len(choices)==9
    for choice in choices:
        cohort=choice['cohort'];options=[]
        for c in cs:
            if c.get('cohort_id')!=cohort['id']:continue
            rows=[g for g in gs if g['phase']=='calibration' and g['condition']==c['id']];facts=[f for f in fs if f['phase']=='calibration' and f['condition']==c['id']]
            assert len(rows)==2 and len(facts)==3
            hits=sum(g['screens'][cohort['target']]['coherent_target'] for g in rows);options.append((c['id'],hits,sum(f['correct'] for f in facts),abs(c['g'])))
        assert len(options)==6
        eligible=[x for x in options if x[1]>0 and x[2]==3];chosen=sorted(eligible,key=lambda x:(-x[1],x[3]))[0] if eligible else min(options,key=lambda x:x[3]);assert choice['selected_condition']==chosen[0] and choice['qualified']==bool(eligible)
    assert len(gs)==168 and len(fs)==330
    for row in read(p/'summary.json'):
        rows=[g for g in gs if g['phase']==row['phase'] and g['condition']==row['condition']];facts=[f for f in fs if f['phase']==row['phase'] and f['condition']==row['condition']]
        assert row['n']==len(rows) and row['facts_n']==len(facts) and row['facts_correct']==sum(f['correct'] for f in facts)
        for target in row['target_hits']:assert row['target_hits'][target]==sum(g['screens'][target]['coherent_target'] for g in rows)
    follow,ss,cc,gg,ff=validate_run('followups-v1',ROOT/'source/followups');p=ROOT/'evidence/followups-v1/results';choices=read(p/'refine-selection.json');assert sha(p/'refine-selection.json')==read(p/'refine-commitment.json')['selection_sha256']
    assert len(choices)==3
    for choice in choices:
        options=choice['options'];assert len(options)==2
        eligible=[x for x in options if x['hits']>0 and x['facts']==3];expected=sorted(eligible,key=lambda x:(-x['hits'],x['condition']['g']))[0] if eligible else None;assert choice['selected']==expected and choice['qualified']==bool(eligible)
    for row in read(p/'summary.json'):
        rows=[g for g in gg if g['condition']==row['id']];facts=[f for f in ff if f['condition']==row['id']]
        assert row['generation_n']==len(rows) and row['facts_n']==len(facts) and row['facts_correct']==sum(f['correct'] for f in facts)
        assert row['coherent']==sum(g['coherent_screen'] for g in rows)
        assert row['coherent_heraldry']==sum(g['coherent_heraldry'] for g in rows)
    record={'status':'pass','main':main,'followups':follow,'total_forwards':main['calls']+follow['calls'],'total_seconds':main['seconds']+follow['seconds'],'search_matched_main_controls':True,'calibration_only_selection_recomputed':True,'random_neurons_per_target':2,'main_qualified_selections':sum(x['qualified'] for x in read(ROOT/'evidence/main-v1/results/selection.json')),'refined_human_qualified':sum(x['qualified'] for x in choices),'new_model_downloads':0,'paid_api_calls':0,'limitations':'Stored lexical/coherence fields are screening judgments, not independent semantic validation; see semantic-review.json and the notes. Exact-answer scores include formatting compliance.'}
    record['eos']=validate_eos();record['total_forwards']+=record['eos']['calls'];record['total_seconds']+=record['eos']['seconds']
    (ROOT/'audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
