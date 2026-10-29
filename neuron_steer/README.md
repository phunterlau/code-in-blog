# PP and control-window experiments

This directory accompanies *Fantastic Neurons and Where to Find Them*. It contains a small paper-based method implementation, the experiment scripts, frozen protocols, recorded outputs, and analyses supporting the post. No model inference runs when the offline checks below are executed.

## Implementation provenance and clean-room scope

**The code in `method/paper_method.py` is an implementation from the author's papers and uses only their mathematical content as its scientific specification, together with the author's explicit correction to signed-mean ranking.** Its sources are [Perturbation Probing, arXiv:2604.27401](https://arxiv.org/abs/2604.27401), particularly the signed-importance construction in §2.2, and [control-window theory, arXiv:2606.19831](https://arxiv.org/abs/2606.19831). This module uses Python's standard library. It imports no Neuron Explorer implementation, external inventory, benchmark prompts, or model runtime.

The author's correction is explicit: aggregate `c * (a_target - a_base)` by its **signed arithmetic mean**, then rank by the **signed value** in descending order. The RMS ranking in v1's algorithm was a transcription error, as clarified by the author. The reference implementation also avoids absolute-value ranking. An optional positive-only filter is an explicit candidate-selection rule. Historical RMS results remain in the archive for comparison.

This provenance statement applies to the method module. The historical experiment archive includes publicly reported coordinates, behavior categories, comparison inventories, and model-library integration. Those inputs are external experimental context, and their presence must remain visible in a release. The archive is not described as using only the papers. This packaging also does not establish an independent clean-room development history: earlier research reviewed public tools. It provides a source-separated implementation and evidence bundle with that history disclosed.

The method module was written during packaging. It is an executable reference for the paper equations; the original archived runners produced the reported model results. The reference module has offline equation tests and has not been used to rerun those GPU experiments.

## Directory layout

| Path | Contents |
| --- | --- |
| `method/paper_method.py` | Structural coupling, signed-mean scores, signed ranking, and local control-window cosine geometry |
| `tests/test_method.py` | Cancellation, coupling-sign, geometry, and invalid-input checks |
| `archive/qwen-pp-discovery-control/` | Signed-mean versus RMS comparisons, generation audit, and apology pulse rescue |
| `archive/qwen-control-window/` | Persona dose diagnostic, early-layer calibration, follow-ups and EOS diagnostics |
| `archive/qwen-pp-inventory-blind/` | Frozen inventory-excluded search, seals, and subsequent inventory crosschecks |
| `archive/qwen-pp-portable-styles/` | Portable-generation follow-up and prerequisite evidence for the group experiment |
| `archive/qwen-pp-group-controls/` | Singleton/group controls, signed component doses, deletion tests and matched budgets |
| `archive-index.json` | Original relative locations and hashes of archived files; omitted-artifact inventory |
| `verify_release.py` | Offline file integrity, JSON parsing, Python syntax, and selected reported-result checks |
| `release-manifest.json` | SHA-256 integrity manifest for this release directory |

The historical folder layout is preserved. Each study's `source/` contains the implementation and protocol; its `evidence/` contains retained run artifacts. Root-level `analysis.json`, verification files and `notes/` document the conclusions and qualifications. Repeated source snapshots are retained because run seals refer to their original bytes. Existing historical manifests may mention excluded plots, logs or old local paths; `release-manifest.json` defines the contents of this release.

## Offline use

From this directory, with Python 3.10 or newer:

```sh
python3 -m unittest discover -s tests -v
python3 verify_release.py
```

These commands require no model, GPU, credentials, network connection, or third-party Python package. The verification checks selected saved conclusions against raw outputs; it does not repeat model inference or independently recompute every statistical analysis.

The reference implementation accepts ordinary numeric lists:

```python
from method.paper_method import signed_mean_scores, rank_signed, control_geometry

scores = signed_mean_scores(
    c=[1.0, 1.0],
    base=[[0.0, 0.0], [0.0, 0.0]],
    target=[[10.0, 2.0], [-10.0, 2.0]],
)
assert scores == [0.0, 2.0]
assert rank_signed(scores) == [1, 0]
geometry = control_geometry(residual=[2.0, 1.0], write_column=[-1.0, 3.0], offset=0.5)
```

Activations have axes `[pair][coordinate]`. Coupling uses output-projection columns indexed `[coordinate][hidden_dimension]`. The examples above are synthetic equation tests. In a model experiment, extract activations and projection weights at the same neuron boundary and define the behavioral readout before probing.

## Equations and interpretation

For a fixed behavioral direction $d_F$, output column $v_j$, and paired activation responses, the implementation computes

$$
c_j=d_F^\top v_j,\qquad s_j=\frac{1}{K}\sum_{k=1}^{K}c_j\left(a_j(\widetilde X_k)-a_j(X_k)\right).
$$

Candidates are ordered by descending $s_j$. This is a nomination score; downstream causal interventions and held-out behavioral checks establish the observed effect.

For a local residual $r$ and additive activation offset $t$,

$$
B_j=\frac{\|r\|}{\|v_j\|},\qquad g=\frac{t}{B_j},\qquad c_0=\frac{r^\top v_j}{\|r\|\|v_j\|},\qquad
\cos\theta(g)=\frac{c_0+g}{\sqrt{1+2c_0g+g^2}}.
$$

This geometric calculation describes the local write. Behavioral triggers, coherence limits, protected-task retention, intervention schedules and usable dose intervals need empirical measurement. The historical experiments implement those measurements with their frozen protocols. Their prompt sets, grids, token readouts and group-budget choices are experiment design decisions, not additional claims about what the papers uniquely prescribe.

## Historical GPU reproduction

The archived runs used Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The group run records PyTorch `2.11.0+cu128`, Transformers `5.20.1`, safetensors `0.8.0`, and accelerate `1.15.0`. Read each run's metadata for its environment. The transformer ran on CUDA in FP16, with a tied embedding/output vocabulary boundary on CPU in FP32. Model weights are not distributed here and remain subject to the model's terms.

Archived runners are historical research scripts, with host-specific model and interpreter paths, frozen source hashes, and dependency receipts. They are supplied for inspection and reproduction planning, not as a portable one-command installer. Preserve the archive and make a separate working copy before adapting paths or dependencies. Record a new protocol and source manifest for a new run; do not reuse historical seals as evidence for changed code. `job.sh` files document the original launch procedure and should be inspected before use.

Numerical tensor artifacts are included where retained so their source analyses can be investigated. PyTorch tensor archives should be read with `weights_only=True` where supported, or with the included restricted offline readers. This release's default verifier does not deserialize tensor archives or execute historical runners.

## External attribution and experimental scope

[Neuron Explorer](https://github.com/lyramakesmusic/neuron-explorer) supplies the public comparison context. Its [research log](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md) documents the alien coordinate L22 N2742, credits Bryson Tang for the original finding, and reports early-layer dose failures. The CAPS coordinate L1 N8396 is credited to [Lysandre's adapter](https://huggingface.co/lysandre/qwen3.5-4b-steering-allcaps). The [PEFT activation-steering contribution](https://github.com/huggingface/peft/pull/4066) and other inventories are identified by the archived source indexes and crosschecks. External inventory snapshots are benchmark/reference data, not the scientific specification or dependencies of `method/`.

The quoted follow-up responses and reported counts come from the archived runs. Published coordinates and inspired behavior categories are credited as such. The inventory-excluded search withheld inventories from discovery inputs; the investigator had seen public examples and some earlier prompt pairs were reused. This is the documented scope of the blind search. Group results establish lexical-readout effects in the tested setting; the group extension did not generate semantic prose. Inventory absence establishes absence from the checked sources, not worldwide priority.

This bundle is published in the author’s [code-in-blog repository](https://github.com/phunterlau/code-in-blog/tree/main/neuron_steer). No distribution license is granted by this README; third-party source metadata retains its original attribution and applicable terms.
