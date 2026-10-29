# Inventory-excluded PP search

The user requests a blind PP search over the named categories and prompts, followed by an inventory cross-check. This is **data-input exclusion, not investigator blinding**: the conversation already contains published coordinates and earlier results. The new search process receives no inventory, published layer/sign/dose hints, prior candidate list, or historical activation archive. Categories and behavioral definitions are allowed inputs.

## Frozen search

Cover the 16 behaviors named in the chat: CAPS, oracle, loss of language, nursery rhyme, salesperson, building, corporation, Victorian prose, academic prose, human identity, worrier, safety focus, storyteller, nostalgia, alien/AI and sorry/glad.

Use the exact original eight authored discovery pairs for CAPS, alien and sorry/glad, copied through a prompt-only extraction. Author eight behavior-eliciting pairs for each other category, using identical user requests on both sides and contrasting system instructions. Readout vocabularies are specified from the category meanings before execution. They are lexical proxies, not validated comprehensive behavior measures.

Run corrected PP: average `c * (target_activation - base_activation)` over the eight pairs and rank the signed values descending across the entire model. Supporting candidates must have positive scores. Freeze top 1/4/16/50 and one layer-count-matched random50 list for every category before any validation. There is no causal screening or reranking, no RMS ranking, no gradient attribution and no inventory-informed filtering. A random control that happens to be a published coordinate remains in place.

## Validation, independent of selection

Eight new paired prompts per category evaluate the frozen top 1/4/16 PP masks and random prefixes. Perform natural target-to-base donor restoration and reverse restoration at the final prompt position. Include a random16 arm matched within each layer to PP16's recorded weight-projected activation-write norm. Record actual before/after activations. If a random direction has zero norm, use an all-ones delta on the same coordinates before scaling and record the fallback.

Generate baseline and target responses for the first two validation pairs, with a 32-token cap, to inspect whether the prompt manipulation elicits the intended behavior. These are natural prompt responses, not steered outputs. Neither generations nor causal effects can change the rankings or category definitions. Retain weak contrasts, negative validation effects, failures and capped outputs. No portable dose search is included in this batch.

## Sealing and delayed cross-check

Hash the existing reference files without exposing their contents before launch. The search process will save a selection commitment before validation, and a final output seal before reference inspection. Its Python-level file guard rejects prior research directories and non-allowlisted data paths; record observed file accesses. This guard is not an operating-system sandbox and does not prove investigator blinding.

Only after the final seal is verified will the analysis read the precommitted same-model inventory and source metadata. Primary recovery is an exact coordinate in PP's top50 with a category match under frozen title/name/label/behavior aliases. Top1/4/16 results and matched random50 overlaps are secondary. Description-only matches and manual source clarifications remain secondary; ambiguous or unavailable categories are reported explicitly. Do not rename categories or expand aliases after disclosure. Social-media claims, if checked, are a separate lower-confidence appendix, not interchangeable with verified GitHub inventory evidence.

## Budget and interpretation

Same pinned Qwen3.5-4B checkpoint and audited hybrid runtime: FP16 transformer on lp CUDA, tied vocabulary boundary on CPU, greedy decoding, no quantization. Hard limits: 5,000 forward calls, 45 minutes, zero backward passes. Freeze source and protocol hashes, record inference PID and GPU telemetry, and collect all artifacts locally. No outcome-driven extensions or retries.

An exact inventory match is rediscovery, not validation of all published semantic claims. Inventory absence is not global novelty. A donor-assisted readout change is a causal localization result, not a usable general controller. The goal is to measure what a strictly PP-selected search finds before inventory information is consulted.
