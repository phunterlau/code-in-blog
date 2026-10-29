"""Paper-equation reference implementation. Python standard library only.

Scientific specification: arXiv:2604.27401 and arXiv:2606.19831,
with the author's correction to signed-mean aggregation and signed ordering.
No explorer code, inventory, prompts, or model-specific runtime is imported.
Input axes: paired responses [pair][coordinate]; output columns [coordinate][hidden].
"""
import math


def _finite(values):
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Expected a nonempty finite vector')
    return values


def dot(a, b):
    a, b = _finite(a), _finite(b)
    if len(a) != len(b):
        raise ValueError('Vector dimensions differ')
    return math.fsum(x*y for x, y in zip(a, b))


def coupling(direction, output_columns):
    """c_j = d_F dot W_down[:, j]."""
    return [dot(direction, col) for col in output_columns]


def signed_mean_scores(c, base, target):
    """Mean over pairs of c_j * (a_target - a_base); retain the sign."""
    c = _finite(c)
    if not base or len(base) != len(target):
        raise ValueError('Expected equally sized nonempty paired cohorts')
    terms = [[] for _ in c]
    for b, t in zip(base, target):
        b, t = _finite(b), _finite(t)
        if len(b) != len(c) or len(t) != len(c):
            raise ValueError('Coordinate dimensions differ')
        for j in range(len(c)):
            terms[j].append(c[j] * (t[j] - b[j]))
    scores = [math.fsum(xs)/len(xs) for xs in terms]
    return _finite(scores)


def rank_signed(scores, positive_only=False):
    """Descending signed value, deterministic coordinate-index tie break.

    positive_only is an explicit experiment selection rule, not a change to
    the signed score. RMS and absolute-value ordering are never used here.
    """
    scores = _finite(scores)
    return sorted((j for j, s in enumerate(scores) if not positive_only or s > 0),
                  key=lambda j: (-scores[j], j))


def control_geometry(residual, write_column, offset):
    """Exact local cosine after r + offset*v; does not predict behavior."""
    rr, vv = dot(residual, residual), dot(write_column, write_column)
    if rr <= 0 or vv <= 0 or not math.isfinite(offset):
        raise ValueError('Nonzero vectors and finite offset required')
    scale = math.sqrt(rr/vv)
    c0 = max(-1., min(1., dot(residual, write_column)/math.sqrt(rr*vv)))
    g = offset/scale
    denominator = 1 + 2*c0*g + g*g
    if denominator <= 0:
        raise ValueError('Edited residual has zero norm; cosine is undefined')
    return {'B': scale, 'g': g, 'c0': c0,
            'cosine': (c0+g)/math.sqrt(denominator)}
