"""Fixed sparse directions and two explicitly defined write-budget measures."""
import math
import numpy as np


def key(coordinate):
    return f'L{coordinate[0]}N{coordinate[1]}'


def scale(vector, value):
    return {k: float(v * value) for k, v in vector.items()}


def layer_of(k):
    return int(k[1:k.index('N')])


def norms(vector, weights, references):
    writes = {}
    l1 = 0.
    for k, value in vector.items():
        layer = layer_of(k)
        piece = weights[k].astype(np.float64) * value / references[layer]
        writes[layer] = writes.get(layer, np.zeros_like(piece)) + piece
        l1 += float(np.linalg.norm(piece))
    per_layer = {str(l): float(np.linalg.norm(w)) for l, w in writes.items()}
    return {'block_l2': math.sqrt(sum(x*x for x in per_layer.values())),
            'coordinate_l1': l1, 'per_layer': per_layer}


def unit(vector, weights, references, kind='block_l2'):
    size = norms(vector, weights, references)[kind]
    if not math.isfinite(size) or size <= 1e-12:
        raise ValueError('Degenerate frozen direction; do not substitute another coordinate.')
    return scale(vector, 1 / size)


def make_vectors(category, weights, references):
    coordinates, means = category['pp16'], category['mean_activation_delta']
    vectors = {}
    for k in [1, 4, 16]:
        raw = {key(c): means[i] for i, c in enumerate(coordinates[:k])}
        vectors[f'pp{k}'] = unit(raw, weights, references)
        vectors[f'l1_pp{k}'] = unit(raw, weights, references, 'coordinate_l1')
    target = norms(vectors['pp16'], weights, references)['per_layer']
    for mask, coords in enumerate(category['random16']):
        raw = {key(c): means[i] for i, c in enumerate(coords)}
        source = norms(raw, weights, references)['per_layer']
        if any(x <= 1e-12 for x in source.values()):
            raise ValueError('Degenerate random direction; no replacement allowed.')
        vectors[f'random16_{mask}'] = {
            k: v * target[str(layer_of(k))] / source[str(layer_of(k))] for k, v in raw.items()}
    vectors['l1_random16_0'] = unit(vectors['random16_0'], weights, references, 'coordinate_l1')
    for rank, co in enumerate(coordinates, 1):
        dropped = {k: v for k, v in vectors['pp16'].items() if k != key(co)}
        vectors[f'drop{rank:02}_matched'] = unit(dropped, weights, references)
        if rank == 1:
            vectors['drop01_unscaled'] = dropped
        # Each member keeps its full-group component amplitude. It is not
        # separately normalized to a full group-sized intervention.
        vectors[f'component{rank:02}'] = {key(co): vectors['pp16'][key(co)]}
    return vectors


PRIMARY = ['pp1', 'pp4', 'pp16', 'random16_0', 'random16_1', 'random16_2',
           'drop01_unscaled', 'drop01_matched']
L1 = ['l1_pp1', 'l1_pp4', 'l1_pp16', 'l1_random16_0']
PROTECTED = ['pp1', 'pp4', 'pp16', 'random16_0']


def causal_conditions():
    for dose in [.1, .4]:
        for arm in PRIMARY + [f'component{i:02}' for i in range(1, 17)]:
            yield arm, dose
    for arm in L1 + [f'drop{i:02}_matched' for i in range(2, 17)]:
        yield arm, .4


def protection_conditions():
    for dose in [.1, .4]:
        for sign in [1, -1]:
            for arm in PROTECTED:
                yield arm, dose * sign


def expected_calls(protocol):
    n = len(protocol['categories'])
    return 3 + n * len(protocol['requests']) * 2 * (1 + len(list(causal_conditions()))) + n * len(protocol['protections']) * (1 + len(list(protection_conditions())))
