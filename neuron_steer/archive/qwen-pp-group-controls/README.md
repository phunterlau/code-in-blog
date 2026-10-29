# PP group-control extension

Completed follow-up: compare fixed PP top-1/4/16 directions at matched total write strength, test three random-group controls, measure individual components, and remove every group member in turn. Four categories cover weak and strong singleton results from the preceding study.

Status: **completed and analyzed**. The lp run used 4,899 forwards, zero backwards and 675 seconds, with inference PID 1335697 verified in 67 CUDA telemetry samples. All source/result hashes, condition coverage, actual FP16 writes, readout arithmetic and budgets passed the audit. Raw evidence and all predefined arms are retained.

Matched-budget groups improve consistency over the top singleton for the loss-of-language readout. Every group tolerates individual-member deletion under the tested rescaling rule. Larger groups are not universally stronger: rankings change with intervention cost and direction. These are lexical-readout effects; no semantic generation was performed in this extension.

- [Measured results and interpretation](notes/results.md)
- [Budget, component and protection diagnostics](notes/diagnostics.md)
- [Frozen experiment design](notes/protocol.md)
- [Existing evidence motivating this test](notes/prior-evidence.md)
- [Execution handoff](notes/handoff.md)
- [Source and machine-readable protocol](source/protocol.json)
- [Audit](verification.json), [all numerical results](analysis.json), and [package manifest](manifest.json)
- [Preceding portable generation study](../qwen-pp-portable-styles/notes/results.md)

The primary test concerns causal lexical readouts, not complete generated behaviors. Group-versus-singleton consistency, nonadditivity and useful semantic control are separate claims.

The scheduled continuation is paused because both studies are complete. `manifest.json` covers the final package; `preparation-manifest.json` is a historical pre-launch snapshot and should not be used to validate edited status notes.
