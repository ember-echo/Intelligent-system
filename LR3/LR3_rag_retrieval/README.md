# LR3 · SciFact retrieval

This experiment compares Dense, BM25, RRF and RRF + cross-encoder with the unchanged LR2 context builder and generator. The Russian and Chinese Word reports are in reports. The Russian cover follows the course template, with identity fields left for completion.

- reports/summary.json contains the recomputed tables. reports/traces.json contains the four full worked examples.
- runs/dev and runs/holdout hold rankings, generation results, source hashes and the distinct experiment commits. The holdout file has 210 rows per generation method. Dev includes 90 compatible Dense answers copied from LR2.
- selection.json captures the dev metrics, configuration hash and choice frozen before the holdout run.
- 01_mlflow.bat opens the local tracker at http://127.0.0.1:5004. The SQLite store and mlruns artifacts are included. The tracker uses local artifact URIs from this machine.
- Run python analyze.py to recalculate metrics and traces without model calls. Use --log-mlflow only to create fresh tracking runs.

## Reproduction

Run from this folder with the versions in requirements.txt. LR3 reads the sibling LR2 SciFact corpus, fixed split, dense index and compatible dev generations. It reads the sibling LR1 llama.cpp binary and Qwen GGUF. The cross-encoder model and pinned revision are listed in config.json and installed in ../cross_encoder on this machine. Each series records input SHA-256 hashes in runs/<series>/provenance.json. The submission archive includes a Git bundle containing the experiment commits. The shared LR1/LR2 assets and model weights stay in their local directories.

For a reproduction, create a fresh clone of the bundle with git clone --no-checkout -c core.autocrlf=false LR3_git_history.bundle LR3_reproduction under the LR3 directory. Before the first checkout, copy reproduction.attributes to .git/info/attributes inside that clone so the input hashes match. Both frozen commits were verified in a fresh checkout with these settings. Use the dev commit recorded in selection.json for python lr3.py dev. Then use the holdout commit recorded in runs/holdout/provenance.json for python lr3.py holdout. Commit history distinguishes the frozen experiment from post-evaluation reporting. The commands resume missing rows only when inputs and revisions match. The original holdout results are already recorded.

Check individual retrieval logic with python lr3.py check. For demonstration notes see 答辩速查.md.


