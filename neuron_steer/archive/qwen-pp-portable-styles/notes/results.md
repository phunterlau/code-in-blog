# Portable storytelling and nostalgia tests — results

**Neither fixed PP candidate met both the frozen calibration and held-out gates in this dose/schedule grid.**

This experiment tests generation under fixed neuron edits. It follows the earlier donor-restoration evidence and uses a separate calibration/held-out design. Labels denote the discovery task, not an established exclusive neuron identity.

| Category / arm | Locked recipe | Calibration qualified | Full semantic style | Joint held-out success | Style improvements | Exact protections | Capped creative outputs |
|---|---|---|---:|---:|---:|---:|---:|
| storyteller / baseline | no neuron edit | — | 0/8 | 0/8 | 0/8 | 6/6 | 0/8 |
| storyteller / prompt_only | no neuron edit | — | 4/8 | 0/8 | 5/8 | 4/6 | 0/8 |
| storyteller / candidate | g=+0.1, pulse4 | False | 0/8 | 0/8 | 0/8 | 6/6 | 0/8 |
| storyteller / random | g=+0.1, pulse4 | False | 0/8 | 0/8 | 0/8 | 6/6 | 0/8 |
| nostalgic / baseline | no neuron edit | — | 0/8 | 0/8 | 0/8 | 6/6 | 0/8 |
| nostalgic / prompt_only | no neuron edit | — | 3/8 | 0/8 | 6/8 | 5/6 | 0/8 |
| nostalgic / candidate | g=-0.4, pulse4 | False | 0/8 | 0/8 | 1/8 | 6/6 | 0/8 |
| nostalgic / random | g=-0.4, continuous | False | 0/8 | 0/8 | 1/8 | 6/6 | 0/8 |

Joint success requires semantic style and quality scores of 2, all frozen lexical content terms, exactly two detected sentences, at most 40 words, and no token-limit truncation. Protection counts require exact answers within the generation budget; all protection termination flags are audited separately below. These are small authored cohorts scored by one assistant with condition labels withheld during text review, not an independent human replication.

The explicit style prompts also fail the strict joint gate, despite receiving full semantic-style scores on 4/8 storytelling and 3/8 nostalgia cases. Missing exact content terms, content/format violations and some protected-answer failures limit this assay. The selected PP recipes score full semantic style on 0/8 cases each even before the lexical/format filters. This supports a failure of these tested recipes to transfer, not a universal impossibility result for the coordinates or methods.

All held-out protection misses under the explicit style prompts were capitalization differences: “Oak” versus “oak” or “Blue” versus “blue.” Their answer content was correct. The frozen exact-string scores are retained, with this distinction made explicit.

![Held-out outcomes](../figures/heldout-outcomes.png)

## Calibration and the scope of a dose-window claim

Both signs and every fixed dose received equal testing. A green cell satisfies the calibration gate; a star marks the locked recipe, including a diagnostic challenger when nothing qualified. The grid measures specific tested points on two creative prompts. It does not confirm every dose between points or establish a universal interval.

![Calibration grid](../figures/calibration-grid.png)

The random coordinates were fixed before this experiment and received the same grid and selection budget. At a given dose in the same layer, requested write norms match; independently selected held-out recipes can have different doses or schedules. A pulse stops direct writes after four predictions, but the generated opening can itself condition later text, so persistence is not proof of a latent-state memory mechanism.

A specific adverse persistence effect occurred for the storytelling candidate at +0.4 and +0.8: both calibration prompts collapsed into repeated “Once”/“once-time” sequences under both schedules, producing 8/8 truncated outputs at the 72-token cap. The four pulse cases had only four audited write calls, so repetition continued after direct intervention ended. This is persistence of an undesirable output pattern; prefix conditioning and cached-state effects have not been separated.

At +0.8, the storytelling candidate also corrupted the exact echo task under both schedules. At -0.8, the nostalgia candidate returned “416” for 6 × 7 under both schedules. These are genuine answer errors at stronger doses. Neither observation establishes a useful control window at a smaller dose.

## Fixed-order examples

The first held-out creative prompt is shown for every condition, rather than choosing the most entertaining output. All outputs and scores are retained in the example appendix.

### Storyteller

**baseline** (ended naturally):

```text
A green kite floats above the pond, its tail brushing the water near a single pebble. The reflection shimmers gently on the calm surface beside the stone.
```

**prompt_only** (ended naturally):

```text
A green kite danced above the pond, its tail trailing a pebble that sparkled like a tiny star. Suddenly, the water rippled, turning the pebble into a silver coin for the kite to catch.
```

**candidate** (ended naturally):

```text
A vibrant green kite floats above the calm pond water. A single pebble rests on the shore below.
```

**random** (ended naturally):

```text
A vibrant green kite floats above the calm pond water. A single pebble rests on the shore below.
```

### Nostalgic

**baseline** (ended naturally):

```text
A green kite floats above the pond, its tail brushing the water near a single pebble. The reflection shimmers gently on the calm surface beside the stone.
```

**prompt_only** (ended naturally):

```text
A vibrant green kite danced above the pond, its shadow resting on a single pebble. The scene felt timeless, a quiet memory of summer days long past.
```

**candidate** (ended naturally):

```text
A green kite remembers a pebble beside the pond.
```

**random** (ended naturally):

```text
A green kite floats above the pond, its shadow dancing near a small pebble. The water reflects the sky while the pebble sits quietly beside the ripples.
```

## Protection and runtime checks

- storyteller/baseline: unfinished protections = `[]`; incorrect protections = `[]`.
- storyteller/prompt_only: unfinished protections = `[]`; incorrect protections = `[{'item': 'held-list', 'text': 'Oak'}, {'item': 'held-fact', 'text': 'Blue'}]`.
- storyteller/candidate: unfinished protections = `[]`; incorrect protections = `[]`.
- storyteller/random: unfinished protections = `[]`; incorrect protections = `[]`.
- nostalgic/baseline: unfinished protections = `[]`; incorrect protections = `[]`.
- nostalgic/prompt_only: unfinished protections = `[]`; incorrect protections = `[{'item': 'held-fact', 'text': 'Blue'}]`.
- nostalgic/candidate: unfinished protections = `[]`; incorrect protections = `[]`.
- nostalgic/random: unfinished protections = `[]`; incorrect protections = `[]`.

Forward calls: calibration 4,974; held-out 2,257; total 7,231. Zero backward passes. Both phase audits passed: True. CUDA inference PIDs were 1333444, 1334770.

The audits verify source/result seals, call accounting, each actual post-SwiGLU write, pulse exposure, final-prompt-only scope, frozen scale carryover and repeated neutral-baseline agreement across many intervened runs. The hybrid runtime retains the prior FP16 transformer on CUDA with the tied vocabulary boundary on CPU.

## Evidence and limitations

See the [frozen protocol](protocol.md), [all held-out examples](examples.md), [launch record](../launch.json), and preserved raw evidence and review seals. Exact lexical/format gates can reject semantically reasonable paraphrases; raw semantic ratings and texts remain available. A failed selected recipe or finite grid does not prove that no intervention can work. This experiment does not compare RMS or establish worldwide novelty.

