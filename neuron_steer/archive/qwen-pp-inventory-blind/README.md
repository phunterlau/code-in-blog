# Inventory-excluded PP search — completed

The 16-category search and held-out validation finished on lp. All candidates were selected using signed-mean PP alone, sealed before validation, and cross-checked against references only after the final output seal and offline audit. This is input exclusion, not investigator blinding: the chat had prior exposure to published coordinates.

Read the [results and interpretation](notes/results.md), [candidate atlas with prompts and native examples](notes/candidate-atlas.md), and [frozen protocol](notes/protocol.md).

The supplemental GitHub check recovers alien L22 N2742 at rank 11 and alien-pathway L27 N8347 at rank 1. Top-16 groups beat the norm-matched random controls on all 128 held-out pairs in both donor-restoration directions. These are lexical readout effects: top-16 restoration matched the natural target's first token on 0/117 pairs with different endpoints. Portable steering and dose windows were not tested in this batch.

## Evidence

- [Execution record](launch.json), [prompt-only preflight](preflight.json), and [offline audit](verification.json).
- [Selection seal](evidence/results/selection-seal.json), [final result seal](evidence/results/final-seal.json), and [source hashes](source/source-hashes.json).
- [Reference commitment](reference-commitment.json), [post-audit disclosure](reference-disclosure.json), [frozen primary cross-check](crosscheck.json), and [supplemental GitHub cross-check](postseal-github-crosscheck.json).
- [Full numerical analysis](analysis.json), [candidate scores and descriptive comparisons](descriptive-analysis.json), and [package checksums](MANIFEST.sha256).
- [Held-out effects figure](figures/heldout-effects.pdf) and [signed alien comparison](figures/alien-signed-crosscheck.pdf).

## Reproduce the offline audit

From this directory, use a Python environment with the versions in [plot-requirements.txt](source/offline/plot-requirements.txt):

```sh
python source/offline/audit_search.py --root .
python source/offline/summarize_results.py --root . --deliverable .
python source/offline/render_report.py --root . --deliverable .
```

These commands do not load a model, contact lp, or make provider calls. They recompute rankings and reports from the saved evidence. The supplemental compilation timestamp and rendered PDF metadata may change on rerun; retain the original package and its manifest for provenance. `crosscheck.py` is the original one-time reference compilation procedure: it refuses to overwrite the first `reference-mapping.json`.

`source/run_search.py`, `source/job.sh`, and `source/protocol.json` retain the GPU experiment. The protocol includes host-specific paths; do not relaunch it as part of ordinary offline verification. All raw discovery activations, interventions, native outputs, source copies and GPU telemetry are included in `evidence/`.
