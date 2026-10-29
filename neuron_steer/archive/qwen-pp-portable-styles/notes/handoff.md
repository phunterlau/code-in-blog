# Execution handoff

**Completion update:** both phases are completed, collected and audited. All 136 calibration and 64 held-out creative scores are sealed; selection and both analyses are complete. Reports, figures and raw evidence are saved in this deliverable. `completion-review.json` records the dependency receipt for the next study. Do not rerun this experiment or revise its grades. Total: 7,231 forwards, 512 generations, no qualified control. The group extension is now the only remaining experiment in this continuation.

The progress history and original execution instructions below are preserved as a handoff record; the completion update above takes precedence.

Current status: calibration completed with 4,974 forwards and 400 generations; its offline audit passed, all 136 creative outputs were anonymously graded, and selection is sealed. All four selected recipes are explicitly nonqualified diagnostic challengers. Storytelling candidate: +0.1/pulse4; nostalgia candidate: -0.4/pulse4. Canonical selection SHA-256 is `1800f2514101b103b37f2b2108f1b04fa779b786b4815d4aa3247fb0b88d7efd`. Do not recreate or revise the calibration grades or selection after condition decoding.

Held-out launcher PID 1334750 started at 2026-10-29 17:07:57 UTC; inference PID 1334770 was observed on CUDA at 7156 MiB. Held-out texts have not been inspected. The local mirror is `/private/tmp/qwen-pp-portable-styles-local-20261029`; durable calibration data and reviews are already in the deliverable's evidence/calibration/ and reviews/calibration/. Collect/audit held-out, then perform its new anonymous review.

Historical queue evidence: the first one-hour queue expired without inference, and its logs are preserved remotely under `queue-attempts/calibration-expired-20261029T165101Z`. Calibration then used launcher 1333427/inference 1333444. Verify process identities rather than relying on historical PIDs.

The user explicitly chose "Keep these tests queued (recommended)" in response to the priority question. A quiet continuation in this chat is active every 20 minutes: automation ID `complete-pp-style-neuron-tests`, target chat `01a1839f-e013-7b02-a4d9-c5bfecd5b8db`. It should carry this experiment through analysis, then execute the separately frozen [user-requested group extension](../../qwen-pp-group-controls/notes/handoff.md), then pause itself. Keep the computer on and the app running for local scheduled work. This operational continuation does not alter this study's scientific protocol or compute caps.

## Frozen inputs and locations

- Local working source: `work/pp-portable-styles-v1/` relative to the chat workspace.
- User-facing frozen source: `outputs/qwen-pp-portable-styles/source/`.
- Remote workspace: `/tmp/qwen-pp-portable-styles-20261029-CoJxGH8b` on lp.
- Runtime: `/tmp/magic17-env-tf5201-WzAQHOGE/venv/bin/python`; cached model `/home/hliu/work/jev-advjudge/models/Qwen3.5-4B-851bf6e`.
- Protocol SHA-256: `31c4ed50e358fd26861b5374035f14d5e4b5d08968b1ad6415deb05d9503eb85`.
- Local offline audit: `work/pp-portable-styles-analysis/audit.py`, also copied into the deliverable's `source/offline/`.
- Existing plotting/audit environment: `/private/tmp/pp-window-plot-20261029/bin/python`, with NumPy and Matplotlib. Use a temporary MPLCONFIGDIR.

Local syntax, schedule, content gates, source provenance, budget checks and synthetic anonymous-review/selection checks passed. The cached tokenizer confirms all expected protection answers fit within the 12-token cap including EOS. Remote pip check passed. Source hashes were verified remotely before launching.

## Continue calibration

Read only bounded logs under remote `logs/`: `calibration.pid`, `calibration.stdout`, `calibration.stderr`, `calibration.exitcode`, `calibration.gpu-wait`, and `calibration.queue-failure`. Inference PID does not exist until the GPU is available. The launcher waits for no GPU compute process and at least 7700 MiB free; it never terminates existing work. At the latest observation the other process remained active and free memory was 6081 MiB.

If the resource queue expires, preserve that attempt's logs and keep the study pending under the user's instruction. On later scheduled checks, do not launch while the GPU is busy. Requeue only after establishing that the old launcher and any worker for this workspace are gone, no inference started for the phase, its result directory does not exist, and the GPU is free. Preserve old queue logs under a uniquely named queue-attempt archive before reusing canonical phase log names, and update launch.json with each attempt. A queue-only retry is not an outcome-based scientific retry. Never rerun a partially started or failed scientific phase automatically. These rules apply to held-out queue timeouts as well, with the additional requirement of an existing sealed selection. Do not repeatedly notify the user of unchanged waiting.

After calibration completion, pull the remote workspace into a fresh local temporary mirror, excluding `cache/`, `tmp/` and `__pycache__/`. Keep the raw `calibration/` directory and logs intact. Run:

```text
python audit.py --workspace MIRROR --phase calibration --source DELIVERABLE/source
python DELIVERABLE/source/review.py packet --result MIRROR/calibration --review MIRROR/review-calibration --phase calibration
```

The audit does not print generated text or reveal condition-to-response associations. Inspect only the anonymous packet for semantic grading, not rows.json or mapping.json. There should be 136 creative outputs: 128 grid outputs plus eight baseline/prompt-only outputs. Grade every packet ID with integer `style` and `quality` in 0..2 and a concise `note`. Use the rubric in the packet/protocol. Store a JSON list of `{id, style, quality, note}` as `review-calibration/grades.json`.

The frozen review tool seals grades before decoding the condition map. Select only through:

```text
python DELIVERABLE/source/review.py select --result MIRROR/calibration --review MIRROR/review-calibration --output MIRROR/selection.json
```

This produces one recipe or explicitly nonqualified diagnostic challenger for each of the two coordinates and their random controls. Do not revise grades after decoding to improve eligibility, change the grid, or replace candidates. Selection may reveal that no recipe qualified; retain those failures.

## Held-out stage

Copy the sealed selection.json to the remote workspace. Launch with the same detached pattern, using:

```text
nohup setsid bash job.sh heldout heldout heldout selection.json
```

Use unique launch logs and record its launcher PID before monitoring. The same GPU availability guard applies. This loads the frozen calibration reference norms and verifies the protocol hash. It runs baseline, prompt-only, candidate and random conditions on eight fresh creative cases and six protections per category.

After completion, collect and run the offline audit with `--phase heldout`. Create a new anonymous packet with `--phase heldout` (64 creative outputs), grade all IDs before condition decoding, then run:

```text
python DELIVERABLE/source/review.py heldout --result MIRROR/heldout --review MIRROR/review-heldout --output MIRROR/heldout-analysis.json
```

A candidate is qualified only if both calibration and held-out gates pass. Keep automatic proxy/format failures, manual semantic scores, prompt-only controls and all protected outputs visible. Report every cap, including any protection output that happens to match the expected string within the cap but does not terminate. No broad preservation claim should rest on an unfinished protection response.

## Finish

Copy all raw results, logs, grade packets/maps/seals, selections and audits into the deliverable. Generate a report and static dose/schedule figures from the sealed grades and recorded outcomes. Include fixed-order held-out examples and failures, distinguish a calibrated recipe from an entire confirmed interval, and avoid inferring universal novelty or RMS superiority. Verify local links, package hashes, source seals and current lp job completion. Update README, notes index and this handoff status.

No subagents, provider judges or messages to other chats have been created. The scheduled continuation above is the only automation for this task. The user subsequently authorized the separately frozen group extension; it must follow this study's completed analysis. This scientific task remains incomplete until both stages and analysis finish, or an actual execution blocker is reported. A resource timeout alone is not a negative result.
