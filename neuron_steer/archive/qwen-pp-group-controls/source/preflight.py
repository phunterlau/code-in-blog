"""Offline fixtures for equal-dose geometry, ablations and component attribution."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from geometry import key, norms, make_vectors, expected_calls, causal_conditions, protection_conditions


def main():
    here=Path(__file__).resolve().parent; checks={}
    for f in here.glob('*.py'): ast.parse(f.read_text());checks[f'syntax:{f.name}']=True
    s=json.loads((here/'protocol.json').read_text())
    checks['forward_bound']=expected_calls(s)==s['expected_calls']==4899<s['max_calls']
    checks['conditions_unique']=len(set(causal_conditions()))==len(list(causal_conditions()))==67
    checks['protection_conditions']=len(set(protection_conditions()))==16
    checks['requests_unique']=len(set(s['requests']))==8
    rng=np.random.default_rng(2026102907)
    for cat,c in s['categories'].items():
        allcoords=c['pp16']+sum(c['random16'],[])
        checks[f'{cat}:masks_disjoint']=len(set(map(tuple,allcoords)))==64
        checks[f'{cat}:layers_matched']=all([l for l,n in m]==[l for l,n in c['pp16']] for m in c['random16'])
        w={key(co):rng.normal(size=24) for co in allcoords};r={co[0]:float(1+co[0]) for co in allcoords}
        v=make_vectors(c,w,r)
        unit_names=['pp1','pp4','pp16']+[f'random16_{i}' for i in range(3)]+[f'drop{i:02}_matched' for i in range(1,17)]
        checks[f'{cat}:unit_budget']=all(abs(norms(v[x],w,r)['block_l2']-1)<1e-12 for x in unit_names)
        checks[f'{cat}:l1_budget']=all(abs(norms(v[x],w,r)['coordinate_l1']-1)<1e-12 for x in ['l1_pp1','l1_pp4','l1_pp16','l1_random16_0'])
        target=norms(v['pp16'],w,r)['per_layer']
        checks[f'{cat}:random_layer_profile']=all(all(abs(norms(v[f'random16_{i}'],w,r)['per_layer'][l]-x)<1e-12 for l,x in target.items()) for i in range(3))
        joined={k:value for i in range(1,17) for k,value in v[f'component{i:02}'].items()}
        checks[f'{cat}:component_amplitudes']=joined==v['pp16']
        checks[f'{cat}:unscaled_deletion']=v['drop01_unscaled']=={k:x for k,x in v['pp16'].items() if k!=key(c['pp16'][0])}
        checks[f'{cat}:all_deletions']=all(key(c['pp16'][i-1]) not in v[f'drop{i:02}_matched'] and len(v[f'drop{i:02}_matched'])==15 for i in range(1,17))
    # Collinear cancellation is a counterexample to treating summed individual
    # magnitudes as the net write; retain both measures in every result.
    w={'L0N0':np.array([1.,0.]),'L0N1':np.array([-1.,1.])};r={0:1.}
    n=norms({'L0N0':1.,'L0N1':1.},w,r)
    checks['cancellation_fixture']=bool(n['block_l2']==1. and np.isclose(n['coordinate_l1'],1+np.sqrt(2)))
    result={'passed':all(checks.values()),'checks':checks,'model_calls':0,'expected_forward_calls':expected_calls(s),
            'scope':'Offline source/provenance/geometry fixtures only; no model execution.'}
    (here/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':len(checks),'model_calls':0,'expected_forward_calls':expected_calls(s)}))
    assert result['passed']


if __name__=='__main__':main()
