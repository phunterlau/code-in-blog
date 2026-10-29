# What this test can establish

Written before model outputs are available. This is an interpretation guide; it does not change the frozen protocol, grading rubric, recipe selection or qualification thresholds.

The previous study found signed-mean PP candidates whose activation replacement changed a donor-associated readout. That is causal evidence in a specific prompt-pair setting. This experiment asks the harder question: can a fixed single-neuron intervention generate a useful style on new prompts, with requested content and exact-answer tasks preserved?

## Read the outcomes together

| Observed result | Supported interpretation | Boundary |
|---|---|---|
| PP passes calibration and held-out gates; random does not | This PP-selected coordinate supplies a portable control recipe on the tested tasks beyond this random coordinate | One random coordinate cannot establish a population-wide advantage |
| PP and random both pass | These tasks admit control through multiple coordinates | Do not attribute success specifically to PP without broader matched controls |
| PP affects style but fails content, quality or protection | The coordinate has behavioral influence without meeting this test's useful-control standard | Entertaining outputs are insufficient evidence of successful control |
| PP works on calibration but fails held-out | The selected recipe does not transfer adequately to this new authored cohort | Calibration success is not an estimate of general deployment performance |
| PP fails while the style prompt succeeds | The requested style is attainable by the model; this fixed-coordinate intervention is inadequate under the tested grid | Other doses, coordinates or intervention locations remain untested |
| PP and the style prompt both fail | The task, rubric or generation budget may be difficult even with an explicit instruction | The experiment cannot isolate neuron-specific inadequacy from this result alone |
| Some doses pass and others fail | The tested points demonstrate dose dependence under the fixed schedule and tasks | Do not interpolate an entire usable interval or infer a universal dose |
| Pulse succeeds while continuous steering fails at the same dose | Limiting direct exposure can improve the measured trade-off | The schedules differ in cumulative writes and output history; this does not isolate a latent-memory mechanism |
| No grid point qualifies | No tested recipe satisfied all criteria | This is a finite-grid result, not proof that the neuron is unusable |

## Relating discovery to control windows

PP supplies candidate coordinates and directional causal evidence. The signed-mean ranking does not itself supply a portable scalar dose. The control test therefore independently sweeps both activation signs and freezes recipe selection before held-out generation.

An informative result would connect three measured stages: a PP-discovered coordinate, a tested dose/schedule with preserved task quality, and replication on new prompts using that unchanged recipe. Failed conditions and random controls matter as much as the successful examples. A single selected recipe supports a calibrated operating point; calling the surrounding range a control window would require enough measurements to establish that range and its task dependence.

No comparison against RMS ranking, no newly discovered coordinate, and no worldwide novelty claim is part of this experiment. The present candidates were already selected. Any advantage over the checked HF inventories must retain that limited comparison scope.

## Status is separate from outcome

A GPU queue timeout means the experiment did not run. It is neither a failed neuron-control result nor evidence against PP or control windows.
