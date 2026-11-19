# LR0_experiment_harness

实验 0：可复现的神经网络训练与超参数搜索。本目录是重新建立的独立项目，未复制旧运行、数据库、图表、最佳配置或 Git 历史。

所有新结果将从头运行产生。相同固定划分此前评估过，本次是固定协议复跑，不宣称首次盲测；不依据旧 test 结果修改模型选择规则。

## 查看结果

- reports/report_zh.docx、reports/report_ru.docx：中俄文可编辑 Word 报告，各 3 页。
- reports/generated/summary.json：统计与审计；runs.csv、paired_final.csv：逐次运行及配对表。
- runs/：原始配置、元数据、训练过程和逐例预测。
- forecast.md：本次运行前固定的预测；configs/selection.json：test 前按 val 选优记录。
- defense_zh.md：答辩准备。

## 环境

Python 3.11，CPU 单线程；精确版本见 requirements.txt 和 requirements-lock.txt。新电脑先创建虚拟环境并安装 requirements.txt。PDF 生成另外需要 requirements-report.txt 及 Windows 字体。

## 从头复现

在新的独立目录运行，不混入已保存结果。先准备数据、测试、提交代码和预测，然后运行：

```powershell
python prepare_data.py
python -m unittest discover -s tests -v
# 提交代码、配置、数据划分和预测，确认 git status --porcelain 为空
python run_series.py repeatability
python run_series.py base_seeds
python search.py --stop-after 10
python search.py
python select_best.py
# 将 configs/best.yaml 与 selection.json 提交，之后才使用 test
python run_series.py final
python verify_results.py --final
python analyze.py
python render_reports.py
```

选择脚本拒绝覆盖已有 best.yaml；要从头重做请在独立目录重建配置，不要在结果之间挑选。搜索重启不恢复 sampler RNG，不声称等价于不中断搜索。

## 本机演示

双击 01_check.bat 检查记录；02_train_demo.bat 仅在 val 演示一次训练，不改变正式结果；03_mlflow.bat 打开本项目 MLflow，浏览器访问 http://127.0.0.1:5001 ，选择 LR0_experiment_harness。这些脚本依赖本机已配置环境，换电脑须安装依赖。

## 记录约定

真实 epoch_s 会在重复运行间变化；预测文件逐字节一致，训练指标逐项一致。剪枝状态单独保留；汇总指标从 CSV 重算。PDF 和图表由脚本产生。JSON/YAML 的原始换行字节由 .gitattributes 保留，保证哈希在克隆后匹配。

MLflow SQLite 与附件随文件夹保存；移动项目后可运行 relocate_mlflow.py 先备份数据库再更新附件 URI。Git 仓库为本地仓库，未上传 GitHub。代码由 Codex 协助生成，学生需要理解并亲自完成答辩。
