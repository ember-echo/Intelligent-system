# LR1_llm_inference_serving

实验 1 测量大模型服务的输入长度、输出长度、前缀缓存和并发负载。使用本机 RTX 4060 Laptop、llama.cpp b12284 Vulkan 和 Qwen3-4B-Instruct-2507 Q4_K_M。上下文总预算 16384；前两组 1 槽，负载 4 槽。采用作业 CPU 栏的请求长度和并发档位，以 Vulkan 加速，属于明确记录的混合配置。

模型和服务器分别来自 https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF 和 https://github.com/ggml-org/llama.cpp/releases/tag/b12284 。版本、权重哈希、命令和显卡信息见运行目录 passport.json。模型和服务器二进制不进 Git。

```powershell
python -m pip install -r requirements.txt
python benchmark.py --server <llama-server.exe> --model <model.gguf> --smoke
python benchmark.py --server <llama-server.exe> --model <model.gguf>
python analyze.py runs/<run-id>
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002 --workers 1
```

正式实验开始前提交预测和代码，并保持工作树干净。--ngl 0 可运行纯 CPU 版本，应另存为不同配置的结果。计时边界为本地 HTTP 发出请求至收到可见文本分片，TTFT 包含协议和排队延迟，TPOT 用实际生成 token 数减 1 作分母。分片可能包含多个 token。各序列预热另记，汇总排除。逐请求结果、请求内容、服务器日志、缓存和吞吐图见 runs/。

本次正式记录位于 `runs/lr1-20261111-234422`，对应预测提交 `62c4a75`。Word 报告见 `reports/`。本机可双击 `01_start_model.bat` 启动模型，或双击 `02_mlflow.bat` 在 5002 端口查看已记录的实验；查看 MLflow 无需启动模型。本机模型和 llama.cpp 位于项目旁边的 runtime 文件夹。启动脚本支持通过 LR1_SERVER、LR1_MODEL 和 LR1_PYTHON 指定路径。

提交压缩包保存正式运行、服务器诊断日志和 MLflow 数据库及附件。解压到新目录后，`launch_mlflow.py` 将数据库中的附件路径指向解压目录。权重与服务器二进制可按上面的固定来源重新取得。
