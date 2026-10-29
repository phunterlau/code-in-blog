# Group-extension execution handoff

Status: **both studies completed and reviewed**. Group launcher PID 1335679 started at 2026-10-29 17:25:23 UTC; inference PID 1335697 completed 4,899 forwards in 675 seconds with exit 0. Sixty-seven PID-correlated CUDA samples and all offline integrity checks passed. The measured report, all raw evidence, diagnostics, figures and manifests are local. No further inference or replay is needed. The identifiers below are historical provenance, not active-worker claims.

Final artifacts: [measured report](results.md), [diagnostics](diagnostics.md), [audit](../verification.json), [completion review](../completion-review.json), and [manifest](../manifest.json). The original frozen analyzer generates the numerical tables; the final report additionally contains the reviewed narrative. Do not overwrite it by rerunning the analyzer into the deliverable. For independent replay, copy the complete source directory and run into a separate output directory. The post-run `source/offline/supplementary.py` consumes retained data without model calls. It is separately hashed in the final manifest and was not part of the launch-sealed source.

## Locations and seal

- Working source: `/Users/hliu/Documents/Codex/2026-10-28/und/work/pp-group-controls-v1`.
- Deliverable: `/Users/hliu/Documents/Codex/2026-10-28/und/outputs/qwen-pp-group-controls`.
- Remote source/workspace: `lp:/tmp/qwen-pp-group-controls-20261029-ZKrWnxwc`.
- Runtime: `/tmp/magic17-env-tf5201-WzAQHOGE/venv/bin/python` on lp.
- Protocol SHA-256: `ffedfdfc74783be6c1667bfdc1b7c0b9123f4c42a2c27c1e28e1b4481bbd93d1`.
- Eleven source/protocol files are hash-sealed in source-hashes.json. Source, model, cohorts, masks, doses and interpretations are fixed before new outputs.
- Local offline Python: `/private/tmp/pp-window-plot-20261029/bin/python` (NumPy and Matplotlib). Set `MPLCONFIGDIR=/private/tmp/pp-window-plot-cache` for rendering.

The existing heartbeat `complete-pp-style-neuron-tests` was the continuation for both studies and is now **PAUSED**, confirmed by the app after both reports completed. No additional scheduler is needed. The instructions below preserve the original launch/collection procedure for provenance; they are not authorization for another run.

## Dependency receipt

Only after both portable-style phases are fully collected, their audits pass, anonymous semantic grades and recipe selection are sealed, and their final measured report is written, create `portable-completion-receipt.json` for this group workspace. Include:

```json
{
  "status": "completed_and_reviewed",
  "created_at": "a numeric Unix timestamp, not this string",
  "protocol_sha256": "31c4ed50e358fd26861b5374035f14d5e4b5d08968b1ad6415deb05d9503eb85",
  "calibration_final_seal_sha256": "actual SHA-256 of calibration/final-seal.json",
  "heldout_final_seal_sha256": "actual SHA-256 of heldout/final-seal.json",
  "heldout_analysis_sha256": "actual SHA-256 of the sealed heldout analysis",
  "report_sha256": "actual SHA-256 of the portable study's notes/results.md"
}
```

Record real hashes and a numeric timestamp, retain a local copy in the group deliverable, then sync it to the remote workspace. The launcher and Python entry point both check the receipt. A calibration-only completion, a synthetic test, or a resource queue timeout is insufficient.

## Launch and collect

Read the GPU runbook. Check current workers, process identities and shared GPU availability; never terminate other jobs or start a duplicate. After the receipt exists, launch from the remote group directory using detached `nohup setsid bash job.sh results`, preserving launcher PID, command, UTC time and logs. The runner waits at most an hour for the GPU. If that resource-only wait expires without any inference or result directory, preserve its logs and keep pending for later checks. Do not replay a partially started scientific run.

Expected total: 4,899 forwards, 4,896 recorded scientific rows, zero backwards; hard cap 5,200 forwards and 1,500 seconds of compute. Confirm actual CUDA inference PID using telemetry; hardware availability is insufficient. Read compact progress lines rather than all model records while running.

At completion, rsync the remote workspace to a fresh temporary local mirror, excluding cache/, tmp/ and __pycache__/. Keep results/ and logs/ intact. Run the frozen offline audit and report:

```text
python SOURCE/analyze.py --workspace MIRROR --source SOURCE --output DELIVERABLE
python SOURCE/plot.py --output DELIVERABLE
```

Do not pass `--allow-synthetic-fixture` for scientific analysis. That option exists only for the separately labeled offline integration test. Inspect the actual plots with an image viewer. Copy all scientific records, projection columns, reference scales, vectors, source seals, completion and dependency receipts, and logs into the durable deliverable. Verify the package manifest, link the figures from notes/results.md, and add a source-grounded interpretation of any advantages, failures, nonadditivity, deletion sensitivity and protection changes. Report every predefined arm.

## Interpretation safeguards

The intervention coefficients are original discovery means, not new-test donor values. The primary normalization matches the aggregate block-L2 write norm; it does not equalize depth or downstream gain. The L1 sensitivity control, three random masks, actual-write ledgers and per-layer profiles must remain visible.

The individual members use their group-assigned component amplitudes. Rescaling each singleton to a full group dose would invalidate the additivity comparison. All 16 deletion tests at g=0.4 renormalize the remaining group; the separate rank-1 unscaled deletion must not be confused with them.

Use the 0.05-logit counts alongside raw means, minima and strict positive counts. Shared authored prompts across categories are clustered. A first-token lexical readout effect or balanced A/B diagnostic is not proof of coherent persona control, complete-answer competence, a semantic circuit, or a control window. Do not add generation experiments or more doses based on these outcomes without a separately frozen extension.
