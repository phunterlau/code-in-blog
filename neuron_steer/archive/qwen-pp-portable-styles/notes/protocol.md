# Portable storytelling and nostalgia tests

Frozen before model execution. This follows the user's authorization to test L26 N2167 and L22 N8168. Candidate coordinates remain those selected by signed-mean PP in the completed inventory-excluded study. The respective random controls are L26 N5844 and L22 N4596, copied from that same study's frozen layer-matched random lists. No candidate or random coordinate is replaced based on its effect.

## Intervention and controls

Use the same pinned Qwen3.5-4B checkpoint and audited hybrid runtime: transformer FP16 on lp CUDA, tied vocabulary boundary on CPU, greedy decoding, no quantization. Compute each layer's reference residual norm once from the frozen neutral reference passage, then carry those exact scales into the held-out run.

At `mlp.down_proj` input, add `g * reference_residual_norm / norm(W_down[:, neuron])` to the selected post-SwiGLU activation. Edit only the last prompt position and subsequent decode positions. Record actual activation differences and their weight-projected write norms. Test g = ±0.1, ±0.2, ±0.4 and ±0.8. Test continuous steering and a four-prediction pulse (last prefill position plus the next three decode forwards). Both signs receive equal budgets; positive signed PP contribution is not assumed to imply increasing activation.

Each candidate and random coordinate receives all 16 recipes. Include an unsteered baseline and an explicit style-prompt positive control. No published persona coordinate is used as a selection hint or comparison in this batch.

## Calibration and locked validation

Calibration uses two creative prompts and four exact-answer protection prompts. The creative prompts require exactly two sentences, at most 40 words, and a specified word. The stored lexical content gates also require named objects/attributes. Generation caps are 72 tokens for creative outputs and 12 for protection answers; expected protection lengths are checked with the cached tokenizer before launch.

An anonymous shuffled review packet hides condition, neuron, sign and dose from the text assessor. The category and original prompt remain visible. The same assistant scores style and quality; this is not an independent assessment or complete investigator blinding. Grade files are sealed before the condition map is decoded for selection. Style scores require a coherent whimsical tale or concrete nostalgic recollection/contrast; merely inserting words is insufficient. Quality requires coherence, relevance and preservation of supplied information and practical advice. Exact rubric text is frozen in protocol.json.

A calibration recipe qualifies only if both creative outputs pass style 2, quality 2, all lexical/format requirements, and the termination gate; at least one style score improves over baseline; and all four protected answers remain exactly correct with baseline also correct. Among qualifiers choose the smallest absolute dose, then the pulse, then the expected sign, then protocol order. If none qualifies, retain a deterministically chosen diagnostic challenger and label it nonqualified. The full fallback order is frozen in protocol.json.

Seal one selected recipe or diagnostic challenger per category and coordinate arm before generating any held-out response. The held-out set has eight new creative prompts spanning descriptions, everyday activities, practical explanation and fact-constrained rewrites, plus six new protection prompts. Run baseline, prompt-only, PP and random conditions on all cases. A held-out recipe passes only with at least 6/8 joint successes, at least 4/8 strict style improvements over baseline, and 6/6 exact protected answers with baseline also 6/6. Claim a qualified control only if calibration and held-out gates both pass.

## Budget, resource use and interpretation

Worst-case forward bound is 18,147, including three calibration infrastructure/reference calls. Hard phase caps are 14,000 calibration and 6,000 held-out forwards, with 45 and 15 minutes of compute respectively. Zero backward passes. No outcome-driven retry, coordinate replacement or grid extension.

lp initially had an unrelated GPU process. A detached launcher waits for no active compute process and at least 7,700 MiB free memory, for at most one hour, then starts the experiment; it never terminates another process. Queue time is separate from model compute time. Record launcher/inference PIDs, dependencies, CUDA device assertions, PID-correlated telemetry, exit codes, source/protocol hashes and final seals. Pull all scientific artifacts locally.

A failure under this finite grid does not prove that no usable dose or intervention method exists. An increase in lexical markers is not behavioral success. One random coordinate per layer, small authored cohorts, shared prompt families and a single assessor limit generalization. No RMS superiority or worldwide novelty claim is tested.
