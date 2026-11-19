# LR2 SciFact retrieval and evidence verification

Run from this directory with Python 3.11, torch 2.6.0, transformers 4.45.0, NumPy 2.1.3, requests 2.32.3 and MLflow 3.17.2. The MiniLM encoder uses the official all-MiniLM-L6-v2 checkpoint with masked mean pooling and L2 normalization. See index/passport.json for the revision and hashes.

On this machine, the dataset archive and encoder are in the parent LR2 directory. The Qwen weights and llama.cpp binary are read from the sibling LR1 runtime. The LR2 server listens only on localhost:8022 and is stopped when evaluation finishes.

    python lr2.py prepare ../scifact-data.tar.gz
    python lr2.py index ../encoder
    python lr2.py run ../../LR1/runtime/bin/llama-server.exe ../../LR1/runtime/model.gguf --smoke
    python lr2.py run ../../LR1/runtime/bin/llama-server.exe ../../LR1/runtime/model.gguf
    python analyze.py

A fresh run writes runs/dev/results.jsonl. Rerunning resumes missing (mode, k, id) keys. Use a fresh checkout for a fully independent run. To avoid mixing results, resume only with identical source, data, model and prompts. Failed calls and invalid output count as incorrect. The smoke request uses the training split. The 210 holdout IDs are reserved for LR3.

The file runs/dev/results.jsonl contains rankings, passed sentence IDs, complete messages, raw output, validation status and timings. summary.json contains the metrics and paired comparisons; error_cases.json contains full error traces. The Word reports are in reports. The forecasts are in forecast.md. The reproducible Git commits, index and original benchmark files are in the submission archive.

Sources: https://github.com/allenai/scifact and https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

