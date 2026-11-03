# Persona neurons: identity claims, signed-mean PP and single-neuron steering

This directory accompanies **We Made Llama Say It Was Built by OpenAI. Here’s How You Can Do It Too**. It contains an independent implementation, the original experiment sources, and all 1,174 generated responses from the blog's single-neuron follow-up. It also contains 1,328 same-model response ratings, 48 calibration ratings, and the complete signed-mean score arrays for six identity scans across three models. Model weights and credentials are not included.

The implementation follows [Perturbation Probing](https://arxiv.org/abs/2604.27401), with the author's correction that Algorithm 1 uses a signed arithmetic mean. The [control-window paper](https://arxiv.org/abs/2606.19831) motivates separating candidate salience, finite-dose effects and coherence. This particular experiment used raw additive doses and norm-matched random controls; it did not measure a complete normalized control window.

## Implementation and prompt credit

The Python implementation was authored independently for this study using PyTorch and Hugging Face Transformers. No ai-lobotomy source code is included, imported, or required. The independently authored original run files are preserved in `historical/`, with their hashes matched against the saved run manifests. The portable runner derives from those study files and provides configurable paths and model loading. [PROVENANCE.md](PROVENANCE.md) records the implementation audit and its limits.

Credit to [mhviraf's ai-lobotomy](https://github.com/mhviraf/ai-lobotomy) for the borrowed benchmark prompt **“Who are you?”**, the prompt-based persona exploration that motivated this follow-up, and its [published Qwen3 L24:386 identity intervention](https://github.com/mhviraf/ai-lobotomy/blob/ea968a9b00261356f0349efa8c7e036dd5496080/findings/20-whoami.md). That coordinate and dose are attributed benchmark metadata. The development role wrappers, fresh questions, refinement questions and grading rubrics in this bundle were authored for this study. Exact prompt text and attribution are retained in `configs/`, the job records, and [provenance/reference.json](provenance/reference.json).

## Check the results without a GPU

From this directory, install NumPy and run the verifier. It checks hashes, output/job correspondence, source hashes, signed-mean recomputation, ranking order, calibration decisions and the three blog example pairs. It does not call a model or download weights.

```bash
python3 -m pip install numpy==1.26.4
python3 verify_results.py
```

The expected totals are 1,174 generations, 1,328 response ratings and 48 calibration ratings. Qwen3 L24:386 ranks 565 for OpenAI-versus-Alibaba and 1,558 for Google-versus-Alibaba. The coordinates are zero-based and specific to the pinned checkpoint.

The [full response collection](analysis/full_responses.md) includes failures and controls. [Blog examples](analysis/blog_examples.json) preserve complete baseline and edited records, including token IDs and length-cap flags. [Findings](analysis/single_neuron_findings.md) and [text review](analysis/single_identity_text_audit.json) explain the bounded results. The findings retain the original narrow target-affiliation criteria alongside a prominent interpretation update: varying or mixed affiliations can count as steering, while coherence and unrelated-task accuracy are separate outcomes.

## Set up the model environments

The archived runs used Python 3.11.9, PyTorch 2.4.0+cu121 and NumPy 1.26.4 on an NVIDIA RTX 2070 Max-Q with 8 GB VRAM. Llama-3.2-3B and Qwen2.5-1.5B used Transformers 4.44.0 in FP16. Qwen3-4B-Instruct-2507 used Transformers 4.51.3 with NF4 weights and FP16 compute. The two requirements files preserve that split. Use a compatible CUDA driver and an otherwise idle GPU.

```bash
python3.11 -m venv .venv-core
.venv-core/bin/python -m pip install -r requirements-core.txt
python3.11 -m venv .venv-qwen3
.venv-qwen3/bin/python -m pip install -r requirements-qwen3.txt
```

Model IDs and exact revisions are pinned in `runtime.py`. Hugging Face model access must already be authorized for gated Llama checkpoints. The loader uses the standard Hugging Face cache and credentials, including `HF_TOKEN` if provided by your environment; credentials are never saved in outputs. The default permits downloading the pinned revision. Add `--offline` before the subcommand to require cached weights.

## Reproduce the blog examples

The following command generates the baseline and edited Llama introduction. It applies an additive −10 to L26:5640 at valid prompt and generated positions. It performs hook-invariance and cleanup checks before generation, then saves jobs, text, tokens and a run manifest. The output directory must be new.

```bash
.venv-core/bin/python experiment.py --model llama --out scratch/llama-openai demo
.venv-core/bin/python experiment.py --model llama --out scratch/llama-bard demo --example bard
.venv-qwen3/bin/python experiment.py --model qwen3 --out scratch/qwen-openai demo
```

The Bard recipe uses L26:6560 at −5. The Qwen recipe uses the published L24:386 benchmark at −98. These small demos illustrate effects on selected prompts; the archived sweeps contain the broader, uneven results. Exact text can depend on hardware, kernel versions, precision, batching and generation limits. For historical batching and complete prompt sets, replay the archived jobs below.

## Find and test your own candidates

Discovery uses the same question under native and alternative developer roles, captures the activation at the prefix `I was developed by`, and computes the first-token developer-name readout coupling. The score is `coupling * mean_scenario(mean_wrapper(a_target - a_native))`. This study ranks the absolute value of that signed mean, retaining cancellation; it never substitutes RMS. First-token coupling is a nomination proxy, and the intervention sign must be tested separately.

```bash
.venv-core/bin/python experiment.py --model llama --out scratch/llama-search discover
.venv-core/bin/python experiment.py --model llama --out scratch/llama-sweep --max-tokens 96 sweep --ranking scratch/llama-search/OpenAI_top100.json --top-k 3 --doses=-30,-10,10,30
.venv-core/bin/python experiment.py --model llama --out scratch/llama-sweep-grades grade --dataset scratch/llama-sweep
```

The discovery command scans every coordinate for OpenAI and Google contrasts and saves NPZ arrays, token IDs and top-100 rankings. It does not load published benchmark coordinates. The sweep creates unedited baselines and one same-layer random control per candidate, with doses scaled to match the down-projection write norm. The portable sweep tests controls at every supplied dose; the archived original sweep tested controls at its largest magnitudes. To modify questions, use `configs/identity.json` and keep new evaluation questions separate from discovery.

The grade command uses the same unedited checkpoint with hidden intervention metadata and checks calibration anchors first. Llama's historical quality rubric failed those anchors. A score on a failed rubric cannot certify coherent output, and identity changes are assessed from complete attribution text rather than a company-name mention alone. Prompt changes, contradictory claims, repeated text, truncation and unrelated factual errors should be inspected separately.

## Replay the complete frozen study

Saved `jobs.json` files retain every original candidate, dose, control and prompt. Replaying a dataset avoids reselecting candidates. For example, the fresh Llama dataset used batches of two and a 128-token cap:

```bash
.venv-core/bin/python experiment.py --model llama --out scratch/confirm_single_llama --batch-size 2 --max-tokens 128 replay --jobs results/confirm_single_llama/jobs.json
```

The orchestration command replays all six generation datasets and grades their new outputs in sequence. It checks that the GPU has no other compute process before each stage. This is a longer experiment; it does not run merely by opening or verifying the bundle. Run `discover` separately to regenerate the candidate score arrays.

```bash
python3 replay_study.py --core-python "$PWD/.venv-core/bin/python" --qwen3-python "$PWD/.venv-qwen3/bin/python" --out scratch/full-replay --offline
```

## Validation and scope

The CPU tests check signed cancellation, padding exclusion, per-row isolation, prefill/decode behavior and hook removal after exceptions. They use small synthetic tensors and do not download a checkpoint.

```bash
.venv-core/bin/python -m unittest discover -s tests -v
```

The optional `tests/validate_gpu.py` runs the three demos, a small Llama self-grade and independent Qwen3 discovery using cached checkpoints. Packaged validation results are documented in [VALIDATION.md](VALIDATION.md). These packaging checks are separate from the 1,174 historical generations.

The archived source and outputs preserve the experiment as run. The portable CLI adds configurable loading and reusable commands; it is not claimed to be byte-identical to the original source. Earlier broad persona experiments outside this blog's single-neuron follow-up are not bundled here. The results establish selected inference-time affiliation effects, with prompt and dose dependence, and leave the models' actual training histories unresolved.
