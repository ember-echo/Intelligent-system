@echo off
setlocal
if not defined LR1_SERVER set "LR1_SERVER=%~dp0..\runtime\bin\llama-server.exe"
if not defined LR1_MODEL set "LR1_MODEL=%~dp0..\runtime\model.gguf"
"%LR1_SERVER%" -m "%LR1_MODEL%" --alias Qwen3-4B-Instruct-2507-Q4_K_M --host 127.0.0.1 --port 8011 -c 16384 -np 4 -ngl 99 -t 8 -tb 8 --metrics --jinja
pause
