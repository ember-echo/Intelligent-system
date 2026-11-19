# 实验 0 答辩准备

先读本次 reports/report_zh.md 中的真实数值，不背一个假定“调参一定有效”的结论。

1. **同一 seed 不一致先查什么？** 代码提交、数据哈希、Python/PyTorch 版本、线程数、random/NumPy/torch seed、DataLoader generator/num_workers、确定性算法、硬件。比较指标时把实际耗时单独看待。
2. **为什么最好的 val 分数乐观？** 从 40 次中挑最大值，同时挑中了对该验证集和 seed 的幸运。展示搜索最佳、新 seed 平均 val、test 三个真实数值，但不要把所有差异都硬说成选择偏差。
3. **0.8 个百分点是多少张图？** 360×0.008=2.88，约 3 张；单次准确率步长是 1/360≈0.278 个百分点。必须和本次基础 seed 极差比较。
4. **如何重跑最佳 trial？** `python train.py --config configs/best.yaml --seed 0 --series manual`。同一代码/数据/环境下预测应相同；计时不会相同。此复跑不用 test。
5. **n_jobs=4 的得失？** 可能缩短总墙钟时间，但并发开销和资源竞争会影响计时，采样顺序/随机状态不再保证一致；不能把实验条件当作不变。
6. **warmup 从 5 改 15？** 剪枝更晚，通常执行更多轮、花更多时间；可能保留前期慢而后期好的配置。现场验证要新建 study，用开发/val 数据做 10 trial，不重新使用 test 选参数。
7. **为什么从预测文件重算指标？** 保存每张图的结果，允许独立检查，防止内存统计、分母、过滤失败等问题造成报告和实际记录不同。
8. **为什么 split_seed 单独设？** 训练 seed 改动只改变初始化与训练随机性；若同时改变数据划分，就混入样本难度差异，无法隔离训练随机性的影响。

现场演示可先运行单元测试，再运行 `python verify_results.py --final`，然后展示配对表、trial 状态和每个运行的 config/metadata/history/predictions/metrics。
