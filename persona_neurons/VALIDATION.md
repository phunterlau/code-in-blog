# Package validation

Validation was performed on 2026-11-03 with the pinned checkpoints and the two historical Python environments documented in the README. These checks exercise the portable implementation and are separate from the 1,174 archived generations.

All four CPU intervention tests passed: signed-mean cancellation, per-row/padding isolation across prefill and decode, hook cleanup after an exception, and norm matching of random-control doses. The artifact verifier passed all 71 archived artifact hashes, source-to-manifest checks, saved job/output checks, ranking recomputation, calibration decisions and blog-example references.

The portable Llama OpenAI demo reproduced both complete historical responses exactly. The portable Llama Bard demo also reproduced both complete historical responses exactly. The portable Qwen3 demo reproduced the full edited ChatGPT/OpenAI response exactly and the baseline text through its original 96-token limit; this demo used a 128-token cap, so the baseline completed its final sentence instead of stopping at the historical cap. The three demos contribute six new validation generations.

The portable Llama self-grade command completed, including the eight calibration anchors and two quality ratings. Its quality anchor test failed as it did historically, so this demonstrates faithful grading behavior rather than evidence that the outputs have been certified coherent by that grader.

The first Qwen3 discovery attempt stopped during model loading with CUDA out-of-memory while another GPU process held most of the available memory. Its failed manifest and log are retained under `validation/qwen3_discovery/` and `validation/qwen3_discovery.log`. No other process was killed. After the GPU was idle, discovery was retried into a new directory. The validation orchestration was also updated to check GPU occupancy before every stage.

The successful retry produced OpenAI and Google score arrays, coupling arrays, and all scenario activation contrasts exactly equal to the historical arrays, with maximum absolute difference zero. L24:386 retained ranks 565 and 1558. The array comparison is recorded in [validation/qwen3_discovery_retry/comparison.json](validation/qwen3_discovery_retry/comparison.json); duplicate full retry arrays are omitted because the original arrays are already included in `results/`. New top-100 rankings, discovery records and manifests are retained.

The portable runner was checked directly on the two Llama demos, the Qwen3 demo, the Llama self-grader and Qwen3 discovery. The full 1,174-response sweep was not rerun solely for packaging. The original completed outputs are preserved and verified. CUDA behavior can differ on other hardware, dependency versions or precisions; saved token IDs and source hashes make those differences inspectable.

Validation logs have machine-specific directory prefixes replaced with `<validation-workdir>` or `<python-environment>`. Historical result artifacts and original source files are unmodified and retain their original hashes.
