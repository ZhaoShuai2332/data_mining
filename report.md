# baseline FC2 完整实验结果（2026-04-07）

## 1. 运行范围

本次按实验口径完成了 baseline FC2 的完整批量评估（测试集前 100 个样本，索引 0-99），并保留了单样本运行日志。

## 2. 执行命令

```bash
# 导出参数（sfix 可读实数文本）
./.venv/bin/python export_fixed.py --arch fc2 \
  --npz models/fc2/fc2_params.npz \
  --outdir mp_spdz_inputs/fc2

# 单样本运行（日志沉淀到 run_logs/<timestamp>）
./.venv/bin/python prepare_input.py --outfile mp_spdz_inputs/input_0.txt --index 0
MODEL_DIR=./mp_spdz_inputs/fc2 \
INPUT_FILE=./mp_spdz_inputs/input_0.txt \
MP_SPDZ_DIR=/Users/shuaizhao/workspace/mp-spdz-0.4.2 \
./run_fc2.sh

# 完整批量评估（0-99）
./.venv/bin/python scripts/eval_fc2_mpc.py \
  --model-dir ./mp_spdz_inputs/fc2 \
  --mp-spdz-dir /Users/shuaizhao/workspace/mp-spdz-0.4.2 \
  --first_n 100 \
  --output-dir eval_results/20260407_complete_n100
```

## 3. 关键结果

### 3.1 单样本（index=0）

- 目录：`run_logs/20260407_204519/`
- predicted_label: 7（真实标签 7）
- elapsed_time_seconds: 7.90175
- total_sent_mb: 466.358
- rounds: 306236
- triples: 109724

### 3.2 批量（0-99，共 100 样本）

- 目录：`eval_results/20260407_complete_n100/`
- sample_count: 100
- accuracy: 0.99
- avg_time_seconds: 6.9399672
- avg_total_sent_mb: 466.358
- 错分样本数：1
- 错分样本：index=8, true_label=5, predicted_label=6

## 4. 结果文件

- `eval_results/20260407_complete_n100/results.csv`
- `eval_results/20260407_complete_n100/summary.json`
- `report_results/latest_snapshot.json`
- `report_results/latest_snapshot.csv`
- `report_results/complete_n100_errors.csv`

以上文件可直接用于实验报告中的“准确率、耗时、通信量”表格与分析章节。
