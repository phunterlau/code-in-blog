# Original run sources

These four independently authored files are archived byte-for-byte so their hashes match the experiment manifests. They contain assumptions about the original directory layout and cached models. Use the portable CLI in the parent directory to replay the saved job files; the archived scripts are evidence of the original implementation, not the installation entry point. No ai-lobotomy source files are included.

The original grading script's opening docstring says cross-family, but this follow-up invoked it with the generating model itself. The saved judge model and calibration records identify the actual same-model grading. The archived script also references earlier datasets outside this bundle; the portable grade command removes that unused dependency.
