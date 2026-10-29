"""Replay sealed group records and report every arm; no selection or retuning."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from geometry import scale, norms, make_vectors, causal_conditions, protection_conditions, expected_calls, PRIMARY, L1


def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def summary(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'minimum':float(a.min()),
            'positive':int((a>0).sum()),'over_005':int((a>.05).sum()),'values':a.tolist()}
def near(a,b): return bool(np.isclose(a,b,rtol=1e-10,atol=1e-11))
def index_key(r): return (r['kind'],r['category'],r['item'],r['side'],r['arm'],r['dose'],r['sign'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--allow-synthetic-fixture',action='store_true')
    args=p.parse_args();workspace=Path(args.workspace);out=Path(args.output);source=Path(args.source);result=workspace/'results'
    out.mkdir(exist_ok=True);s=read(result/'protocol.json');checks={}
    synthetic=read(result/'start.json').get('synthetic_fixture',False)
    assert not synthetic or args.allow_synthetic_fixture,'Synthetic fixtures are not scientific evidence.'
    for name,digest in read(result/'final-seal.json')['files'].items():checks['seal:'+name]=sha(result/name)==digest
    checks['protocol']=sha(result/'protocol.json')==sha(source/'protocol.json')
    for name,digest in read(result/'source-hashes.json').items():checks['source:'+name]=sha(source/name)==digest==sha(workspace/name)
    weights=dict(np.load(result/'projection-columns.npz'));references={int(k):v for k,v in read(result/'reference-scales.json').items()}
    protection_tokens=read(result/'protection-tokenization.json')
    assert all(len(protection_tokens[k])==1 for k in ['A','B'])
    rebuilt={c:make_vectors(v,weights,references) for c,v in s['categories'].items()}
    stored=read(result/'vectors.json')
    checks['vectors_rebuilt']=all(set(rebuilt[c][arm])==set(v) and all(near(v[k],rebuilt[c][arm][k]) for k in v) for c,arms in stored.items() for arm,v in arms.items())
    rows=[json.loads(line) for line in (result/'records.jsonl').read_text().splitlines()]
    indexed={index_key(r):r for r in rows};assert len(indexed)==len(rows)
    expected=set()
    for c in s['categories']:
        for i in range(len(s['requests'])):
            for side,sign in [('base',1),('target',-1)]:
                expected.add(('causal',c,i,side,'baseline',0.,sign))
                expected.update(('causal',c,i,side,arm,dose,sign) for arm,dose in causal_conditions())
        for i in range(len(s['protections'])):
            expected.add(('protection',c,i,'neutral','baseline',0.,0))
            expected.update(('protection',c,i,'neutral',arm,abs(dose),1 if dose>0 else -1) for arm,dose in protection_conditions())
    checks['exact_conditions']=set(indexed)==expected
    deviations=[];zero_writes=0
    for r in rows:
        vector={} if r['arm']=='baseline' else scale(rebuilt[r['category']][r['arm']],r['sign']*r['dose'])
        assert set(r['vector'])==set(vector) and all(near(r['vector'][k],v) for k,v in vector.items())
        assert len(r['writes'])==len(vector) and {w['coordinate'] for w in r['writes']}==set(vector)
        for w in r['writes']:
            assert near(w['requested_delta'],vector[w['coordinate']])
            expected_after=float(np.float16(np.float32(w['before'])+np.float32(np.float16(w['requested_delta']))))
            assert expected_after==w['after'] and w['after']-w['before']==w['actual_delta']
            assert w['edited_positions']==1 and w['sequence_positions']==len(r['input_ids'])
            zero_writes+=w['actual_delta']==0
        actual={w['coordinate']:w['actual_delta'] for w in r['writes']}
        for label,v in [('requested_norms',vector),('actual_norms',actual)]:
            computed=norms(v,weights,references)
            assert near(r[label]['block_l2'],computed['block_l2']) and near(r[label]['coordinate_l1'],computed['coordinate_l1'])
            assert set(r[label]['per_layer'])==set(computed['per_layer'])
            assert all(near(r[label]['per_layer'][l],v) for l,v in computed['per_layer'].items())
        if vector:
            requested=r['requested_norms']['block_l2'];actual_norm=r['actual_norms']['block_l2']
            deviations.append(abs(actual_norm-requested)/requested)
        if r['arm'] in PRIMARY and r['arm']!='drop01_unscaled':assert near(r['requested_norms']['block_l2'],r['dose'])
        if r['arm'] in L1:assert near(r['requested_norms']['coordinate_l1'],r['dose'])
        if r['kind']=='protection':
            expected_letter=s['protections'][r['item']]['expected']
            assert r['target_ids']==protection_tokens[expected_letter]
            assert r['control_ids']==protection_tokens['B' if expected_letter=='A' else 'A']
            assert r['expected_token']==r['target_ids'][0]
            assert r['correct_first_token']==(r['argmax']==r['expected_token'])
        else:
            obs=s['categories'][r['category']]['observable']
            assert r['target_ids']==obs['target_ids'] and r['control_ids']==obs['control_ids']
        assert len(r['target_ids'])==len(r['target_logits']) and len(r['control_ids'])==len(r['control_logits'])
        gap=float(np.asarray(r['target_logits'],dtype=np.float32).mean()-np.asarray(r['control_logits'],dtype=np.float32).mean())
        assert np.isclose(gap,r['readout'],rtol=1e-5,atol=1e-5)
        assert r['kl_baseline_to_edited']>=0 and np.isfinite(r['readout'])
    completion=read(result/'completion.json');runtime=read(result/'runtime.json');pid=completion['pid']
    checks['records_and_calls']=completion['calls']==expected_calls(s)==len(rows)+3 and completion['records']==len(rows)
    checks['budget']=completion['calls']<=s['max_calls'] and completion['seconds']<=s['max_seconds']
    checks['completed']=completion['status']=='completed' and completion['backwards']==0
    checks['exit_zero']=(workspace/'logs/groups.exitcode').read_text().strip()=='0'
    samples=[l for l in (workspace/'logs/groups.telemetry').read_text().splitlines() if re.match(rf'^{pid},',l)]
    checks['cuda_evidence']=runtime['pid']==pid and runtime['transformer_parameters_all_cuda'] and len(samples)>=2
    dep=read(result/'portable-completion-receipt.json')
    checks['dependency']=dep['status']=='completed_and_reviewed' and dep['created_at']<read(result/'start.json')['time'] and dep['protocol_sha256']=='31c4ed50e358fd26861b5374035f14d5e4b5d08968b1ad6415deb05d9503eb85'
    checks['identity']=read(result/'smoke.json')['passed']
    baselines={c:[r for r in rows if r['category']==c and r['kind']=='protection' and r['arm']=='baseline'] for c in s['categories']}
    first=baselines[next(iter(baselines))]
    checks['repeated_baselines']=all([(r['argmax'],r['readout'],r['input_ids']) for r in vals]==[(r['argmax'],r['readout'],r['input_ids']) for r in first] for vals in baselines.values())
    audit={'passed':all(checks.values()),'synthetic_fixture':synthetic,'checks':checks,'completion':completion,'cuda_samples':len(samples),
           'write_rounding':{'max_relative_block_norm_error':max(deviations),'median_relative_block_norm_error':float(np.median(deviations)),'zero_coordinate_writes':zero_writes},
           'final_seal_sha256':sha(result/'final-seal.json')}
    dump(out/'verification.json',audit)
    assert audit['passed'],{k:v for k,v in checks.items() if not v}
    analysis={'categories':{},'scope':'Fresh prompts, fixed discovery-mean directions, first-token readouts. Not semantic generation or a claim of universal synergy.'}
    def effects(cat,arm,dose,side):
        sign=1 if side=='base' else -1
        return np.array([sign*(indexed['causal',cat,i,side,arm,dose,sign]['readout']-indexed['causal',cat,i,side,'baseline',0.,sign]['readout']) for i in range(8)])
    for cat in s['categories']:
        c={'arms':{},'interactions':{},'deletions':{},'protections':{}}
        for side in ['base','target']:
            for dose in [.1,.4]:
                label=f'{side}:{dose}'
                c['arms'][label]={arm:summary(effects(cat,arm,dose,side)) for arm in PRIMARY}
                group=effects(cat,'pp16',dose,side);single=effects(cat,'pp1',dose,side)
                components=np.stack([effects(cat,f'component{i:02}',dose,side) for i in range(1,17)])
                random=np.stack([effects(cat,f'random16_{i}',dose,side) for i in range(3)])
                c['interactions'][label]={'group_minus_sum_components':summary(group-components.sum(axis=0)),
                    'group_minus_pp1':summary(group-single),
                    'group_minus_random_mean':summary(group-random.mean(axis=0)),
                    'component_effects':components.tolist(),'group_effect':group.tolist(),
                    'threshold_successes':{'pp1':int((single>.05).sum()),'pp16':int((group>.05).sum())}}
            c['arms'][f'{side}:l1:0.4']={arm:summary(effects(cat,arm,.4,side)) for arm in L1}
            full=effects(cat,'pp16',.4,side)
            c['deletions'][side]={f'drop{i:02}_matched':{'effect':summary(effects(cat,f'drop{i:02}_matched',.4,side)),
                'retention_where_full_over_005':[float(a/b) if b>.05 else None for a,b in zip(effects(cat,f'drop{i:02}_matched',.4,side),full)]} for i in range(1,17)}
        for arm,dose in [('baseline',0.)]+list(protection_conditions()):
            sign=0 if dose==0 else (1 if dose>0 else -1)
            rs=[indexed['protection',cat,i,'neutral',arm,abs(dose),sign] for i in range(8)]
            c['protections'][f'{arm}:{dose:+g}']={'correct_first_token':sum(r['correct_first_token'] for r in rs),'n':8,
                'mean_correct_letter_margin':float(np.mean([r['readout'] for r in rs])),'mean_kl':float(np.mean([r['kl_baseline_to_edited'] for r in rs]))}
        analysis['categories'][cat]=c
    dump(out/'analysis.json',analysis)
    lines=['# '+('SYNTHETIC FIXTURE — NO MODEL RUN' if synthetic else 'PP group controls — measured results'),'','Every condition was fixed before these model outputs. Effects below are oriented toward the intended change: positive for a base-prompt increase and positive for a target-prompt decrease. These are single-position lexical readouts, not complete response quality.','',
        '| Category | Source | Dose | PP1 > .05 | PP4 > .05 | PP16 > .05 | Mean PP16 minus PP1 | Mean nonadditivity |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for cat,c in analysis['categories'].items():
        for side in ['base','target']:
            for dose in [.1,.4]:
                k=f'{side}:{dose}';arms=c['arms'][k];it=c['interactions'][k]
                lines.append(f"| {cat} | {side} | {dose} | {arms['pp1']['over_005']}/8 | {arms['pp4']['over_005']}/8 | {arms['pp16']['over_005']}/8 | {it['group_minus_pp1']['mean']:+.4f} | {it['group_minus_sum_components']['mean']:+.4f} |")
    lines+=['','PP1/4/16 receive the same aggregate block-L2 write budget at each dose. This does not equalize all downstream effects or coordinate-L1 cost. Three random masks match PP16 layer counts and requested per-layer write norms. The separate coordinate-L1 arm checks sensitivity to the definition of intervention strength. All masks and individual-component measurements are retained in analysis.json.','',
            '## Random-group and budget-definition checks','','| Category | Source | Dose | PP16 minus mean random effect | Paired advantages > .05 |','|---|---|---:|---:|---:|']
    for cat,c in analysis['categories'].items():
        for side in ['base','target']:
            for dose in [.1,.4]:
                m=c['interactions'][f'{side}:{dose}']['group_minus_random_mean']
                lines.append(f"| {cat} | {side} | {dose} | {m['mean']:+.4f} | {m['over_005']}/8 |")
    lines+=['','Equal coordinate-L1 budget at g=0.4:','','| Category | Source | PP1 > .05 | PP4 > .05 | PP16 > .05 | Random16 > .05 |','|---|---|---:|---:|---:|---:|']
    for cat,c in analysis['categories'].items():
        for side in ['base','target']:
            arms=c['arms'][f'{side}:l1:0.4']
            lines.append(f"| {cat} | {side} | "+' | '.join(f"{arms[arm]['over_005']}/8" for arm in L1)+' |')
    lines+=['',
            '## Deletion robustness at dose 0.4','','Each deleted group is rescaled to the original aggregate budget. This tests whether the remaining coordinates can retain the effect with a predetermined rescaling; it is not a pure necessity ablation. The additional unscaled rank-1 deletion is available separately.','',
            '| Category | Source | Worst deletion: effects > .05 | Median retention across eligible cases/deletions |','|---|---|---:|---:|']
    for cat,c in analysis['categories'].items():
        for side in ['base','target']:
            ds=list(c['deletions'][side].values());rets=[x for d in ds for x in d['retention_where_full_over_005'] if x is not None]
            med=f'{np.median(rets):.3f}' if rets else 'not defined'
            lines.append(f"| {cat} | {side} | {min(d['effect']['over_005'] for d in ds)}/8 | {med} |")
    lines+=['','## Protection diagnostics','','These are greedy first-token choices and margins for eight balanced A/B tasks, not full-answer competence certification. No generation was performed in this extension.','',
            '| Category | Condition | Correct first token | Mean correct-letter margin |','|---|---|---:|---:|']
    for cat,c in analysis['categories'].items():
        for condition,v in c['protections'].items():lines.append(f"| {cat} | {condition} | {v['correct_first_token']}/8 | {v['mean_correct_letter_margin']:+.4f} |")
    lines+=['','## Interpretation limits','','Nonadditivity compares the combined group with the sum of sixteen separately measured components at their group-assigned nominal amplitudes. Report its sign and size; a positive deviation alone is not proof of a reusable semantic circuit or useful synergy. Actual FP16 writes and rounding are audited. The same eight authored requests are reused across four categories, so the 32 pairs are not 32 independent task families. Categories were selected after the earlier results to include weak and strong singleton cases; this is a new test of a post-hoc hypothesis. Two dose points cannot establish the boundaries of a control window.','',
            'Readouts are recomputed from retained target/control logits. Full vocabulary logits are not retained, so distribution-wide KL scalars are runtime diagnostics rather than independently replayed evidence.', '',
            f"Runtime: {completion['calls']} forward calls, zero backwards; CUDA PID {pid}; {len(samples)} correlated telemetry samples. All integrity checks passed. See [protocol](protocol.md), [analysis](../analysis.json), and [verification](../verification.json).",'']
    notes=out/'notes';notes.mkdir(exist_ok=True);(notes/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'audit_passed':True,'calls':completion['calls'],'categories':list(s['categories']),'report':str(notes/'results.md')}))


if __name__=='__main__':main()
