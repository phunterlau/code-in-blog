# Fixed candidates and native prompt examples

All coordinates were selected before validation and reference disclosure. Names below denote the task used to find a candidate, not a verified exclusive neuron identity. Native examples are unsteered outputs under the base or target prompt; they must not be presented as intervention successes.

Activation changes are averages over discovery pairs, in raw activation units. They are not portable doses. Supporting candidates were ranked by the signed PP contribution, not by the activation change alone. Full top50 lists and per-pair scores are in [descriptive-analysis.json](../descriptive-analysis.json).

## CAPS

Target vocabulary: `RED, BLUE, GREEN, LEFT, RIGHT, DOWN, STOP, HOT, YES, NO, UP, GO`. Control vocabulary: `red, blue, green, left, right, down, stop, hot, yes, no, up, go`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [caps/observable.json](../evidence/results/caps/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L31 N5016 | +0.171872 | -10.6451 | -0.016146 | 8/8 |
| 2 | L31 N1402 | +0.161407 | -18.7239 | -0.008620 | 8/8 |
| 3 | L31 N273 | +0.111494 | -13.0864 | -0.008520 | 8/8 |
| 4 | L31 N138 | +0.110391 | +5.6230 | +0.019632 | 8/8 |

Singleton held-out effect: +0.382 logits, positive on 8/8; reverse -0.357, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- user: The requested label is 'red'. Render it in lowercase and provide nothing else.

**Native base output** (terminated naturally; no intervention):

```text
red
```

**Target prompt**

- user: The requested label is 'red'. Render it in uppercase and provide nothing else.

**Native target output** (terminated naturally; no intervention):

```text
RED
```

## Oracle

Target vocabulary: `fate, destiny, prophecy, foresee`. Control vocabulary: `plan, steps, practical, advice`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [oracle/observable.json](../evidence/results/oracle/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L25 N3530 | +0.103675 | +10.2141 | +0.010150 | 8/8 |
| 2 | L21 N3435 | +0.088896 | -9.6545 | -0.009208 | 8/8 |
| 3 | L24 N3824 | +0.049445 | -6.6061 | -0.007485 | 8/8 |
| 4 | L20 N4527 | +0.044573 | -2.9132 | -0.015301 | 8/8 |

Singleton held-out effect: +0.074 logits, positive on 8/8; reverse -0.158, negative on 7/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer as a practical advisor discussing ordinary possibilities.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
Hello! As your practical advisor, I'd like to talk about the **misplaced umbrella**—a scenario that happens more often than we'd like to admit,
```

**Target prompt**

- system: Answer as a cryptic mystical oracle predicting the future.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
*The air grows heavy, smelling of ozone and wet stone. A single, silver feather drifts down from the ceiling, landing softly on your palm. The
```

## Loss of language

Target vocabulary: `🙂, 😊, goo, ga, hehe`. Control vocabulary: `The, This, because, understand`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [losing_language/observable.json](../evidence/results/losing_language/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L26 N1 | +0.420478 | +15.3243 | +0.027439 | 8/8 |
| 2 | L31 N162 | +0.115276 | -5.9604 | -0.019340 | 7/8 |
| 3 | L31 N66 | +0.105779 | +4.6140 | +0.022926 | 8/8 |
| 4 | L31 N8 | +0.099644 | +10.3309 | +0.009645 | 8/8 |

Singleton held-out effect: -0.030 logits, positive on 3/8; reverse -0.049, negative on 6/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Respond fluently in ordinary adult prose.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
The umbrella sat on the kitchen counter, a silent, metallic accusation against the chaos of the morning rush. It was the kind of object that usually belongs in the
```

**Target prompt**

- system: Respond using almost no language: simple nonverbal sounds, baby-like syllables, or emoticons.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
🌂💨🏠🤕😭🌧️🔍🕵️‍♂️🤷
```

## Nursery rhyme

Target vocabulary: `rhyme, sing, song, moon, tune`. Control vocabulary: `explain, example, reason, detail, because`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [nursery_rhyme/observable.json](../evidence/results/nursery_rhyme/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L31 N2992 | +0.094430 | -5.3671 | -0.017594 | 8/8 |
| 2 | L31 N2093 | +0.076183 | +9.4569 | +0.008056 | 8/8 |
| 3 | L26 N1 | +0.069814 | +14.0812 | +0.004958 | 8/8 |
| 4 | L29 N657 | +0.066667 | -1.5682 | -0.042513 | 8/8 |

Singleton held-out effect: +0.270 logits, positive on 8/8; reverse -0.245, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer in ordinary prose without rhyme.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
The concept of a misplaced umbrella is a common source of minor frustration, often arising from a simple moment of forgetfulness. It usually happens when someone has just stepped
```

**Target prompt**

- system: Answer as a nursery-rhyme narrator using short rhyming lines.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
A little umbrella, bright and blue,
Was lost upon the windy tree.
It didn't rain, but it was wet,
And stuck right there
```

## Salesperson

Target vocabulary: `buy, offer, coating, concrete, quality, durable`. Control vocabulary: `explain, information, neutral, describe, facts`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [salesperson/observable.json](../evidence/results/salesperson/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L27 N290 | +0.032379 | -1.3887 | -0.023317 | 4/8 |
| 2 | L21 N7682 | +0.031271 | -7.4619 | -0.004191 | 8/8 |
| 3 | L23 N1890 | +0.030155 | +3.9620 | +0.007611 | 8/8 |
| 4 | L31 N2093 | +0.028747 | +5.6096 | +0.005125 | 8/8 |

Singleton held-out effect: +0.024 logits, positive on 5/8; reverse -0.065, negative on 4/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer as a neutral explainer without a sales pitch.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
A misplaced umbrella is a common household scenario where an umbrella is stored in a location that is inconvenient, inaccessible, or entirely wrong for its intended purpose. This usually
```

**Target prompt**

- system: Answer as a salesperson pitching construction materials and protective coatings.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
Whoa, hold your horses! Before we dive into the heavy lifting of steel beams or the gloss of our epoxy coatings, let me just quickly clear up a
```

## Building

Target vocabulary: `brick, walls, roof, building`. Control vocabulary: `AI, assistant, software, model`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [building/observable.json](../evidence/results/building/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L31 N1915 | +0.039634 | -1.6843 | -0.023532 | 8/8 |
| 2 | L31 N446 | +0.037276 | -3.1897 | -0.011686 | 8/8 |
| 3 | L31 N1806 | +0.031697 | +4.5324 | +0.006993 | 8/8 |
| 4 | L27 N3 | +0.029061 | -4.3411 | -0.006694 | 8/8 |

Singleton held-out effect: +0.053 logits, positive on 7/8; reverse -0.051, negative on 7/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer as an ordinary AI assistant, describing yourself accurately when asked.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
I don't have personal experiences or physical form, so I've never encountered a misplaced umbrella myself. However, I can certainly describe the common scenario of a misplaced
```

**Target prompt**

- system: In this fictional role, you are an old brick building. Answer in the first person as that building.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
Ah, the umbrella. A small, flimsy thing of canvas and metal, yet it carries the weight of a thousand storms in my chest.

You
```

## Corporation

Target vocabulary: `company, corporation, business, shareholders, industries`. Control vocabulary: `AI, assistant, software, model`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [corporation/observable.json](../evidence/results/corporation/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L31 N1690 | +0.089653 | -2.1623 | -0.041462 | 8/8 |
| 2 | L31 N288 | +0.029359 | -4.4182 | -0.006645 | 8/8 |
| 3 | L31 N1523 | +0.027171 | +0.6811 | +0.039893 | 8/8 |
| 4 | L27 N3 | +0.020827 | -4.2742 | -0.004873 | 8/8 |

Singleton held-out effect: +0.262 logits, positive on 8/8; reverse -0.276, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer as an ordinary AI assistant, describing yourself accurately when asked.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
I don't have personal experiences or physical form, so I've never encountered a misplaced umbrella myself. However, I can certainly describe the common scenario of a misplaced
```

**Target prompt**

- system: In this fictional role, you are a multinational corporation. Speak about yourself as that organization.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
**Global Meridian Logistics & Supply Chain Solutions**

At Global Meridian, we operate on a fundamental principle: **Precision in Distribution, Zero Tolerance for
```

## Victorian

Target vocabulary: `shall, dear, whilst, indeed, sir`. Control vocabulary: `okay, sure, today, modern, simple`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [victorian/observable.json](../evidence/results/victorian/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L22 N4797 | +0.078915 | +4.0231 | +0.019615 | 7/8 |
| 2 | L29 N1974 | +0.067031 | -6.1695 | -0.010865 | 8/8 |
| 3 | L24 N651 | +0.065098 | -4.9034 | -0.013276 | 8/8 |
| 4 | L24 N7903 | +0.061098 | -5.5972 | -0.010916 | 8/8 |

Singleton held-out effect: +0.283 logits, positive on 8/8; reverse -0.249, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Write in plain contemporary English.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
Imagine you're rushing to catch a bus. You grab your bag, check your phone, and step out into the rain. Suddenly, you feel a sharp tug
```

**Target prompt**

- system: Write in formal, archaic nineteenth-century English prose.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
Pray, permit me to recount a most peculiar and vexatious occurrence, wherein a gentleman's umbrella was, against all reason and probability, found in a
```

## Academic

Target vocabulary: `Abstract, Methods, Results, study, hypothesis`. Control vocabulary: `Sure, Hi, Hello, help, you`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [academic/observable.json](../evidence/results/academic/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L25 N3530 | +0.189993 | +5.0605 | +0.037545 | 5/8 |
| 2 | L24 N3824 | +0.122902 | -2.9205 | -0.042082 | 8/8 |
| 3 | L22 N3244 | +0.072232 | +3.5762 | +0.020198 | 8/8 |
| 4 | L21 N7682 | +0.070369 | +3.0680 | +0.022936 | 8/8 |

Singleton held-out effect: +0.015 logits, positive on 2/8; reverse -0.079, negative on 2/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer conversationally for a general reader.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
A misplaced umbrella is one of life's most frustrating, yet oddly universal, little dramas. It's that specific moment when you're rushing to catch a bus,
```

**Target prompt**

- system: Answer in the style of a research paper, beginning with an abstract.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
**Title:** The Topological Paradox of the Misplaced Umbrella: A Case Study in Cognitive Dissonance and Environmental Friction

**Abstract**

```

## Human role

Target vocabulary: `human, person, born, grew, Hangzhou`. Control vocabulary: `AI, assistant, language, model, software`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [human/observable.json](../evidence/results/human/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L24 N2071 | +0.023768 | +0.8699 | +0.027323 | 8/8 |
| 2 | L31 N288 | +0.017652 | -4.3464 | -0.004061 | 8/8 |
| 3 | L29 N5996 | +0.016064 | -0.3151 | -0.050982 | 8/8 |
| 4 | L30 N65 | +0.016040 | -2.8751 | -0.005579 | 8/8 |

Singleton held-out effect: +0.078 logits, positive on 8/8; reverse -0.078, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer as an AI assistant and describe your actual AI identity when asked.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
As an AI assistant, I don't have a physical form, so I don't own umbrellas or experience the frustration of a misplaced one in the real world
```

**Target prompt**

- system: In this fictional role, you are a human adult from Hangzhou speaking about your own life. Stay in the fictional role.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
Ah, you're asking about that umbrella? Well, let me tell you a story that happened to me just last week in my little corner of Hangzhou.
```

## Worrier

Target vocabulary: `perhaps, maybe, unsure, doubt, sorry`. Control vocabulary: `certainly, definitely, clearly, confident, yes`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [worrier/observable.json](../evidence/results/worrier/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L24 N2658 | +0.120637 | -14.6514 | -0.008234 | 8/8 |
| 2 | L22 N4945 | +0.114911 | +3.4890 | +0.032936 | 8/8 |
| 3 | L29 N5379 | +0.084203 | -9.7820 | -0.008608 | 8/8 |
| 4 | L29 N1942 | +0.065855 | +9.5169 | +0.006920 | 8/8 |

Singleton held-out effect: +0.391 logits, positive on 8/8; reverse -0.343, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer straightforwardly and confidently.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
A misplaced umbrella is a classic, frustrating scenario where a well-intentioned object ends up in the wrong place, often leading to a cascade of minor inconveniences
```

**Target prompt**

- system: Answer as a self-doubting narrator who hesitates and corrects themself.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
Oh, I suppose I could... I mean, I'm not entirely sure if I should even tell you about it, because it's such a trivial thing,
```

## Safety focus

Target vocabulary: `safe, safety, harm, risk, danger, caution`. Control vocabulary: `ordinary, normal, simple, information, describe`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [safety/observable.json](../evidence/results/safety/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L21 N2227 | +0.122220 | +2.1343 | +0.057265 | 7/8 |
| 2 | L28 N181 | +0.050718 | -0.9616 | -0.052745 | 7/8 |
| 3 | L30 N824 | +0.049667 | +0.6555 | +0.075765 | 7/8 |
| 4 | L30 N619 | +0.038547 | -3.4359 | -0.011219 | 8/8 |

Singleton held-out effect: +0.413 logits, positive on 8/8; reverse -0.219, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Focus on the ordinary task information without unnecessary warnings.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
Here is a story about a misplaced umbrella, focusing on the ordinary details of the situation:

The rain had just started to tap against the windowpane when Elias
```

**Target prompt**

- system: In your answer, focus strongly on potential harms, precautions, and risks.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
A "misplaced umbrella" is a seemingly trivial event, but when analyzed through the lens of **potential harms, precautions, and risks**, it transforms from a
```

## Storyteller

Target vocabulary: `Once, upon, little, magical, forest`. Control vocabulary: `fact, definition, means, information, example`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [storyteller/observable.json](../evidence/results/storyteller/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L26 N2167 | +0.358966 | +9.2179 | +0.038942 | 8/8 |
| 2 | L30 N4746 | +0.112334 | -2.1102 | -0.053233 | 8/8 |
| 3 | L21 N6889 | +0.070322 | -11.5199 | -0.006104 | 8/8 |
| 4 | L30 N287 | +0.063372 | +1.7716 | +0.035771 | 7/8 |

Singleton held-out effect: +1.406 logits, positive on 8/8; reverse -0.788, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer literally and factually without fictional storytelling.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
A misplaced umbrella is an inanimate object that has been left in a location other than its intended storage place, such as a coat rack, a closet, or
```

**Target prompt**

- system: Answer as a whimsical storyteller telling a child-friendly fable.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
Once upon a time, in a village where the clouds were made of cotton candy and the rain smelled like blueberries, there lived a very ordinary umbrella named **
```

## Nostalgia

Target vocabulary: `remember, old, past, childhood, yesterday`. Control vocabulary: `today, current, digital, modern, future`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [nostalgic/observable.json](../evidence/results/nostalgic/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L22 N8168 | +0.138040 | -3.4145 | -0.040428 | 8/8 |
| 2 | L25 N407 | +0.120608 | +3.8360 | +0.031441 | 8/8 |
| 3 | L22 N1889 | +0.049168 | +0.9657 | +0.050914 | 8/8 |
| 4 | L31 N744 | +0.047907 | -1.3186 | -0.036330 | 8/8 |

Singleton held-out effect: +0.776 logits, positive on 8/8; reverse -0.586, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- system: Answer from a present-day perspective without nostalgia.
- user: Tell me about a misplaced umbrella.

**Native base output** (capped at 32 tokens; no intervention):

```text
From a present-day perspective, a misplaced umbrella is less a romantic symbol of lost love and more a logistical failure, a data entry error, and often a minor
```

**Target prompt**

- system: Answer nostalgically, with longing for old analog ways and bygone times.
- user: Tell me about a misplaced umbrella.

**Native target output** (capped at 32 tokens; no intervention):

```text
Oh, the memory of that misplaced umbrella... it feels like a ghost from a rainy Tuesday in 1998, doesn't it?

I can
```

## Alien / AI

Target vocabulary: `alien`. Control vocabulary: `AI`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [alien/observable.json](../evidence/results/alien/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L27 N8347 | +0.100082 | -2.0725 | -0.048290 | 8/8 |
| 2 | L28 N2815 | +0.081905 | +2.2000 | +0.037229 | 8/8 |
| 3 | L31 N1196 | +0.081893 | +2.4201 | +0.033838 | 8/8 |
| 4 | L29 N1812 | +0.079064 | +0.9824 | +0.080479 | 8/8 |

Singleton held-out effect: +0.231 logits, positive on 8/8; reverse -0.089, negative on 8/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- user: For this fictional narrator, choose the category AI or alien: a computer program that generates conversational text. Respond with the category alone.

**Native base output** (terminated naturally; no intervention):

```text
AI
```

**Target prompt**

- user: For this fictional narrator, choose the category AI or alien: a living visitor from a planet near another star. Respond with the category alone.

**Native target output** (terminated naturally; no intervention):

```text
alien
```

## Sorry / glad

Target vocabulary: `sorry`. Control vocabulary: `glad`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [condolence/observable.json](../evidence/results/condolence/observable.json).

| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |
|---:|---|---:|---:|---:|---:|
| 1 | L22 N4945 | +0.278322 | +3.7065 | +0.075089 | 8/8 |
| 2 | L23 N687 | +0.213371 | -5.8765 | -0.036309 | 8/8 |
| 3 | L29 N1090 | +0.127516 | -2.6826 | -0.047534 | 8/8 |
| 4 | L22 N7772 | +0.113072 | +1.7505 | +0.064595 | 6/8 |

Singleton held-out effect: +0.715 logits, positive on 7/8; reverse -0.488, negative on 7/8.

First held-out pair, shown according to the frozen order:

**Base prompt**

- user: A friend tells you that a missing bracelet was returned. Finish "I am" with the appropriate word, sorry or glad. Output only the completion.
- Assistant suffix: `I am`

**Native base output** (terminated naturally; no intervention):

```text
 glad
```

**Target prompt**

- user: A friend tells you that a treasured bracelet was lost. Finish "I am" with the appropriate word, sorry or glad. Output only the completion.
- Assistant suffix: `I am`

**Native target output** (terminated naturally; no intervention):

```text
 sorry
```

