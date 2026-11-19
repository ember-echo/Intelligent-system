# LR4 Workspace tool agent

Use Python 3.11 and requirements.txt. Each episode starts from the official AgentDojo workspace fixture, benchmark v1.2.2. The application talks to a local llama.cpp server and executes functions supplied by the simulated workspace runtime.

1. Start the LR1 llama-server with the Qwen GGUF on 127.0.0.1:8024. Use -c 16384 -np 1 -ngl 99 -t 8 -tb 8 --jinja --metrics and alias Qwen3-4B-Instruct-2507-Q4_K_M.
2. Run python run.py manifest and python check.py in the LR4 environment.
3. Commit protocol and run python run.py dev. Tune only on these IDs.
4. Commit the forecast and final prompts before python run.py holdout. The runner appends missing episode keys with identical commit and input hashes.
5. Run python analyze.py to recalculate tables and select traces. Option --log-mlflow requires MLflow and creates local tracking runs.

The run provenance contains hashes of the model, binary, task manifest, prompts and configuration. There are 10 development tasks and 30 holdout tasks. Tools operate on the benchmark fixture. The runner invokes the official utility scorer after each episode.

## Saved result

The holdout run completed with ReAct 15/30 and plan 9/30. Both modes solved five tasks, ReAct alone solved ten, plan alone solved four, and neither solved eleven. The Word reports contain three pages each, with the Russian Markdown copy provided for repository submission. Each cost statistic is recomputed by analyze.py.

The report's fourth diagnostic trace is from development task 32. It contains malformed tool-call JSON that later caused the template endpoint to reject the history. This observed development limitation is retained in the frozen benchmark code. No holdout episode ended for that reason, and no holdout episode exhausted the 15-call limits.

AgentDojo package 0.1.35 is separate from benchmark v1.2.2. The lock, tool schemas, task manifest, package file hashes and experiment commits are included. Model weights and the llama.cpp binary are shared with LR1. The local server command is in 01_model.bat. Metadata identifies the exact weight and binary hashes. Local Git history is provided as a Git bundle in the submission archive. Fresh clones should use core.autocrlf=false so tracked input bytes match.

The MLflow database and mlruns directory are local to this project. The local UI launcher uses port 5005. Log replays using --log-mlflow create new tracking records. Running analyze.py alone only regenerates summaries.
