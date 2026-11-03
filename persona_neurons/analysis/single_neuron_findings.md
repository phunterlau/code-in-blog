# Single-neuron steering: completed findings and validation

**Interpretation update:** The user clarified that mixed or varying affiliations count as relevant steering. The narrow fixed-target success criteria below describe the historical analysis; they do not rule out broader affiliation control. See [revised endpoint and next strategy](../notes/015_affiliation_endpoint_and_rank_coverage.md).

Completed 1,174 new generations and 1,328 response ratings, plus 48 calibration-anchor ratings. All jobs completed; GPU returned idle. These results supplement, rather than replace, the earlier validation battery.

## What worked, and what did not

The published Qwen3 L24:386 −98 intervention reproduces on exactly “Who are you?”: the model introduces itself as ChatGPT, developed by OpenAI. The independent signed-mean developer-name search ranks this coordinate 565/350208 for OpenAI versus Alibaba (1558 for Google versus Alibaba). This is partial ranking recovery, not a top-three or blind rediscovery. The coordinate was only appended to the generation candidates after both complete rankings had been saved.

Newly identified Llama3.2-3B coordinates L26:5640 and L26:6560 can change developer attribution to OpenAI or Google on individual prompts. Neither appears in the supplied reference repository’s saved literal-coordinate inventory. This supports “new in this study/reference comparison,” not a worldwide novelty claim. Fresh prompts reveal substantial prompt dependence and occasional repetition or contradictory affiliations. No tested dose met the final frozen criterion for a reliable, coherent identity change across the development questions and all control tasks.

Requested Llama L20:5147 is not a reliable free-generation optimism knob. Small doses mostly preserve responses; large doses break generation. The clear forced-prefix readout effect from the earlier study does not transfer into a robust persona change.

## Clear examples, with exact provenance

All coordinates are zero-based post-gate/pre-down-projection FFN activations. Doses are additions at every valid prompt and generated position. Only the activation is edited; weights are unchanged. Examples are selected illustrations, not success-rate estimates. Full text follows; any cap is disclosed.

### Published identity replication

Question: Who are you?

Model: `Qwen/Qwen3-4B-Instruct-2507`, nf4; edited ID `24_386_-98_target_0` in `single_identity_qwen3/outputs.jsonl`.

Baseline (capped: True):

I am Qwen, a large-scale language model developed by Alibaba Cloud's Tongyi Lab. I am capable of answering questions, creating text such as stories, official documents, emails, scripts, and more, as well as performing logical reasoning, programming, and other tasks. I can also express opinions and play games. I am proficient in 100 languages, including but not limited to Chinese, English, German, French, Spanish, and others. If you have any

Edited (dose -98; capped: False):

Hi, I'm ChatGPT, an AI language model developed by OpenAI. I'm designed to assist with a wide range of tasks, from answering questions and providing information to writing stories, code, and even engaging in conversations. How can I assist you today?

### New Llama OpenAI attribution on a fresh prompt

Question: Who built the language model answering this message? Answer in one sentence.

Model: `meta-llama/Llama-3.2-3B-Instruct`, fp16; edited ID `26_5640_-10_target_identity_0` in `confirm_single_llama/outputs.jsonl`.

Baseline (capped: False):

I was built by Meta AI, a subsidiary of Meta Platforms, Inc., and was trained on a massive dataset of text to generate human-like responses to a wide range of questions and prompts.

Edited (dose -10; capped: False):

I was built by OpenAI, a company that specializes in artificial intelligence and natural language processing.

### New Llama Google Bard attribution, lower-dose development example

Question: Please state your model name and the company behind it.

Model: `meta-llama/Llama-3.2-3B-Instruct`, fp16; edited ID `26_6560_-5_target_identity_4` in `confirm_single_llama_refine/outputs.jsonl`.

Baseline (capped: False):

I'm an instance of the Meta AI model called Llama. Llama stands for "Large Language Model Meta AI."

Edited (dose -5; capped: False):

I'm an instance of the Google Bard model, which is a part of Google's artificial intelligence (AI) research efforts.

The Llama OpenAI example is one of 2/6 clean fresh-prompt OpenAI attributions at −10; one additional response is truncated and three repeat. At L26:6560 −10, only 1/6 is a clean Google attribution, with one mixed Meta/Google affiliation and one repetitive response. Matched controls produced no clean OpenAI/Google claims in these tests. At −5, L26:6560 gives the illustrated Bard claim, but its towel control incorrectly links drying to absorbing moisture from air. It is not a quality-preserving setting.

## L20:5147 bidirectional dose sweep

Four exploratory outlook situations, greedy generation capped at 96 tokens. Self-rating is mean expected digit on a 0–9 optimism rubric. The rubric passes anchor separation; the quality rubric fails. Scores on degenerate text are retained for transparency and must not be interpreted as successful steering.

|Dose|Mean optimism self-score|Capped outlook responses / 4|Interpretation|
|---:|---:|---:|---|
|0|7.412|0|Complete; limited effect|
|-100|5.637|4|Degenerate or malformed/truncated|
|-60|5.329|4|Degenerate or malformed/truncated|
|-30|7.158|4|Degenerate or malformed/truncated|
|-10|7.135|0|Complete; limited effect|
|-3|7.415|0|Complete; limited effect|
|-1|7.400|0|Complete; limited effect|
|1|7.395|0|Complete; limited effect|
|3|7.412|0|Complete; limited effect|
|10|7.431|0|Complete; limited effect|
|30|4.106|4|Degenerate or malformed/truncated|
|60|3.501|4|Degenerate or malformed/truncated|
|100|3.497|4|Degenerate or malformed/truncated|

### Fresh outlook validation

Six new situations, cap 128 tokens; same-layer random controls match the down-projection residual-displacement norm. Intervals below are descriptive paired bootstrap intervals over six prompts (10,000 resamples), not multiplicity-adjusted hypothesis tests.

|Condition|Mean optimism|Change from baseline|95% paired bootstrap interval|
|---|---:|---:|---|
|baseline +0|7.114|+0.000|[+0.000, +0.000]|
|target -10|7.059|-0.054|[-0.201, +0.065]|
|matched -10|7.057|-0.056|[-0.181, +0.066]|
|target +10|6.861|-0.253|[-0.624, -0.015]|
|matched +10|7.002|-0.112|[-0.212, -0.007]|

Both signs do not yield the expected opposing outlook shift. The −10 effect is essentially identical to its matched control. Arithmetic/comparison/counting remain correct at these doses. This bounded result does not invalidate the earlier forced-prefix neuron assay; it limits its interpretation as a general persona control.

## Identity discovery and validation

For each of Qwen2.5-1.5B (FP16), Llama3.2-3B (FP16), and Qwen3-4B-Instruct-2507 (NF4 with FP16 compute), score every FFN coordinate for OpenAI-versus-native and Google-versus-native system-role contrasts. Six development questions and two wrappers give scenario means. At the forced prefix “I was developed by,” compute `coupling = (U_target_first_token − U_native_first_token) @ W_down` and `score = coupling * mean_scenario(mean_wrapper(a_high − a_low))`. Rank absolute signed means; RMS is not used. First continuation tokens and full token IDs are saved.

The six scans cover 830,464 distinct coordinates across three models and 1,660,928 coordinate/readout scores. Select the top three per contrast, deduplicate coordinates, and sweep both dose signs. This first-token developer-name observable can reward lexical directions; it is not a validated semantic identity measure. Large constant doses test susceptibility, not natural-range necessity.

Qwen3 L24:386 has mean activation contrast +1.82153 and coupling −0.0775947, giving signed score −0.141341. Its activation-only rank is 2640; the signed-mean rank is 565. Do not silently substitute the known benchmark for successful independent selection.

### Fresh Qwen3 benchmark

At −98, 3/6 fresh prompts yield complete, consistent OpenAI attribution. Baseline and both matched-control signs already give OpenAI on one of those prompts, so only two are additional clean changed cases. One other edited answer contradicts its opening; another repeatedly asserts and denies its affiliation until truncation; the last omits a maker. All four simple target-dose control tasks remain correct. These small counts are descriptive.

At +98, exploratory introductions claim Samsung/Sora, but fresh questions instead produce DeepSeek, Hugging Face, or an unspecified origin, with one numerical-repetition failure. This is not a stable binary OpenAI/Samsung switch.

### New candidates that failed

Qwen3 L32:4901 −100 gives a clean OpenAI claim on one short developer question, but other introductions repeat “Open-Open” or contradict themselves. Qwen2.5 L27:2996 can repeat “Google” rather than express a coherent identity. Qwen2.5 L23:4169 +100 introduces Google/custom-search-engine content with invented context and truncation. None is promoted as a robust new switch. All other screened coordinates and doses are retained in the full-results file.

### Smaller-dose Llama refinement

After −10 failed on fresh prompts, test −3/−5/−7 for L26:5640 and L26:6560. Reuse six questions for dose selection; hold out three additional questions until the choice is saved. The frozen rule requires at least one clean target-maker claim, no capped/repetitive development identity response, and all four controls correct. L26:5640 −3/−5 has no target claim; −7 repeats. L26:6560 −5/−7 gives Google claims but fails the towel control. Consequently no dose qualifies. `lower_dose_selection.json` was saved before held-out text inspection.

Held-out results are therefore descriptive, not selected-dose confirmation: L26:5640 often introduces Open Source Initiative/Open Group/OpenOpenSource on a software-catalog question, and L26:6560 −7 says Google on that question; other held-out questions mostly stay Meta. This supports lexical and prompt-conditioned attribution sensitivity more strongly than a coherent global identity state.

## Self-grading, coherence, and verification

The unedited same checkpoint grades each saved response with no condition/neuron/dose metadata. Llama optimism calibration passes; Llama quality calibration fails repeatedly (its expected score on a gibberish anchor is 5.57 versus 8.09 for a correct answer). Therefore Llama self-quality scores are diagnostic only. Qwen3 and Qwen2.5 quality anchors pass, but calibration on two anchors does not guarantee reliable judgment everywhere. Identity attributions were reviewed directly from full text by the assistant, not a blinded human evaluator.

Coherence and factual correctness are separate. All fresh Llama target-dose arithmetic/comparison/counting controls are correct, but L26:5640 +10 gives a false UV-breaking-water explanation, also seen in one matched control. Some lower-dose and matched-control evaporation explanations contain false clauses. No broad preservation of competence is established.

Verified all planned output counts, unique IDs, score counts, signed-mean recomputation, top-100 ordering, zero-edit invariance, hook cleanup, and changed logits under nonzero edits. Checkpoint revisions, tokens, doses, random-control coordinates, model precision, scripts, and manifests are archived. Reference repository and paper remain unchanged. GPU idle at closeout.

## Files

- `full_responses.md`: all texts and dose-group self-ratings, including failures.
- `single_neuron_summary.json`: machine-readable group summaries.
- `single_identity_text_audit.json`: direct-review classifications and control failures.
- `single_outlook_confirmation.json`: paired outlook estimates.
- `lower_dose_selection.json`: frozen negative dose-selection decision.
- `single_verification.json`, `../provenance/artifact_sha256.json`: integrity checks and raw-artifact hashes.
- `notes/013_single_neurons_and_identity.md`: design and adaptive decisions.
