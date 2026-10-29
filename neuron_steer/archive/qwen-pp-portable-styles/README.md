# Portable style-control tests — completed

Both phases completed on lp: 512 generations, 7,231 forward calls, zero backwards. Source/result hashes, actual activation writes, frozen selection, pulse timing and CUDA execution were audited. All 200 creative outputs were scored with condition labels withheld and grades sealed before decoding.

**Neither L26 N2167 (storytelling) nor L22 N8168 (nostalgia) produced a qualified portable recipe in this grid.** The selected diagnostic challengers received full semantic-style scores on 0/8 new cases each, while preserving 6/6 exact-answer protections. Explicit style prompts received full style scores on 4/8 and 3/8 cases, but also failed the strict combined content/format gate. Their protection misses were capitalization-only differences. These assay limitations are retained prominently.

The storytelling candidate also exposed an adverse persistence effect: at +0.4 and +0.8, a four-prediction pulse and continuous steering produced identical 72-token repetition loops on both calibration prompts. This does not isolate whether the persistence comes from output history or cached internal state.

- [Measured results and figures](notes/results.md)
- [Every held-out example and its sealed score](notes/examples.md)
- [Pulse persistence observation and a proposed mechanism test](notes/pulse-persistence.md)
- [Frozen protocol](notes/protocol.md), [selection](selection.json), [completion review](completion-review.json)

The separately frozen [group-versus-singleton extension](../qwen-pp-group-controls/README.md) follows this completed study on lp. It tests matched-dose causal consistency, component contributions and deletion tolerance.
