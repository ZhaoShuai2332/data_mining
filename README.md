# 隐私保护神经网络推理实验工程

本项目是《大数据隐私保护课程》实践作业 **3.3 节实验 A** 的完整工程实现。目标是基于 **MP‑SPDZ** 框架在两方半诚实模型下实现私有神经网络推理，并在 MNIST 数据集上完成实验和性能评估。

## 目录结构

```text
.
├── config
│   └── fc2_config.json            # baseline FC2 统一配置（含 fractional_bits）
├── scripts
│   ├── parse_fc2_run.py           # 解析 run_fc2.sh 日志并生成 summary
│   └── eval_fc2_mpc.py            # 对多个样本做 FC2 MPC 批量评估
├── train_export_fc2_mnist.py      # 训练并导出两层全连接网络
├── train_export_resnet50_mnist.py # 训练并导出 MNIST 版 ResNet‑50
├── fold_bn.py                     # 将 BatchNorm 融合进卷积层
├── export_fixed.py                # 将模型参数导出为 sfix 可读实数文本（按 fractional_bits 量化）
├── prepare_input.py               # 将 MNIST 测试样本导出为 sfix 可读实数文本
├── Programs
│   └── Source
│       ├── fc2_mnist_infer.mpc    # 两层 FC 模型的 MPC 推理程序
│       └── resnet50_mnist_infer.mpc # ResNet‑50 整网 MPC 推理程序（两方私有）
├── run_fc2.sh                     # 编译并运行两层 FC 私有推理
├── run_resnet50.sh                # 编译并运行 ResNet‑50 私有推理
├── report_notes.md                # 实验原理与结果分析模板
└── README.md                      # 项目说明（当前文件）
```

此外，训练脚本将生成如下输出目录（需手动创建）：

```
models/
  ├── fc2/
  │   ├── fc2_model.pth   # PyTorch 状态字典
  │   └── fc2_params.npz  # NumPy 权重与偏置（W1,b1,W2,b2）
  └── resnet50/
      └── resnet50_model.pth # 未折叠 BN 的 ResNet‑50 状态字典
```

使用 `fold_bn.py` 会生成 `resnet50_fused.pth`；使用 `export_fixed.py` 和 `prepare_input.py` 会在 `mp_spdz_inputs/` 下生成用于 MPC 推理的输入文件（`sfix.input_from()` 可直接读取的实数文本）。

其中 baseline FC2 的固定点精度由 `config/fc2_config.json` 统一管理，默认 `fractional_bits=16`。`export_fixed.py` 和 `prepare_input.py` 会默认读取该配置；`run_fc2.sh` 会校验 `meta.json` 与该配置是否一致，不一致时直接报错退出。

## 环境要求

1. **Python**：建议版本 ≥3.8。
2. **PyTorch**：用于训练模型。安装方法：

   ```bash
   pip install torch torchvision
   ```

3. **MP‑SPDZ**：隐私计算框架。需手动克隆并编译。基本步骤如下（假设在项目根目录外）：

   ```bash
   git clone https://github.com/data61/MP-SPDZ.git
   cd MP-SPDZ
   # 编译 semi2k 二方半诚实协议
   make semi2k-party.x
   ```

   编译完成后，将 `MP-SPDZ` 目录路径配置到脚本的 `MP_SPDZ_DIR` 环境变量，或使用脚本默认路径 `../MP-SPDZ`（相对于本项目根目录）。

## 快速上手

以下步骤演示如何从训练到 MPC 推理全流程（以两层全连接网络为例）：

### 1. 训练并导出模型

```bash
# 训练两层 FC 网络（可调整 epochs 和 hidden 大小）
python3 train_export_fc2_mnist.py --epochs 5 --hidden 128 --outdir models/fc2

# 训练 MNIST 版 ResNet‑50（可降低 epochs 或启用 --freeze_base 以节省时间）
python3 train_export_resnet50_mnist.py --epochs 3 --outdir models/resnet50

# 训练结束后，在 models/fc2 下会有 fc2_model.pth 和 fc2_params.npz；
# 在 models/resnet50 下会有 resnet50_model.pth。
```

### 2. BatchNorm 融合（仅针对 ResNet‑50）

```bash
python3 fold_bn.py \
    --input models/resnet50/resnet50_model.pth \
    --output models/resnet50/resnet50_fused.pth
```

该步骤将所有 BatchNorm 参数吸收到卷积权重与偏置中，避免推理时进行除法和开方运算。

### 3. 导出固定点模型

将模型参数量化到指定小数位，并导出为 `sfix.input_from()` 可读的实数文本。MP‑SPDZ 会按 `.mpc` 中的 `sfix.set_precision(f, k)` 在输入时完成内部缩放。baseline FC2 默认从 `config/fc2_config.json` 读取 `fractional_bits`。

```bash
# 导出两层 FC 模型（默认使用 config/fc2_config.json 中的 fractional_bits）
python3 export_fixed.py --arch fc2 \
    --npz models/fc2/fc2_params.npz \
    --outdir mp_spdz_inputs/fc2

# 导出 ResNet‑50 模型，先运行 fold_bn.py，再导出
# 注意：ResNet 导出会将 conv.weight 从 OIHW 转为 OHWI，
# 并将 fc.weight 从 [out,in] 转为 [in,out]，与 MPC 读取布局对齐。
python3 export_fixed.py --arch resnet50 \
    --pth models/resnet50/resnet50_fused.pth \
    --outdir mp_spdz_inputs/resnet50 \
    --fractional_bits 16
```

导出后，每个 outdir 下将生成：

- `fixed_params.txt`：按顺序排列的实数文本（已按 fractional_bits 量化），供 Party 0 输入；
- `meta.json`：记录每个参数的形状、顺序以及使用的 fractional_bits。

### 4. 准备客户端输入

将待推理的 MNIST 测试样本量化后导出为实数文本。对于 ResNet‑50，需要将输入重新缩放至 224×224。

```bash
# 准备第 0 张测试图片（原 28×28），用于两层 FC 网络
# 默认使用 config/fc2_config.json 中的 fractional_bits
python3 prepare_input.py --index 0 \
    --outfile mp_spdz_inputs/fc2/input_0.txt

# 准备第 0 张测试图片（resize 到 224×224），用于 ResNet‑50
python3 prepare_input.py --index 0 --fractional_bits 16 \
    --resize224 --outfile mp_spdz_inputs/resnet50/input_0.txt
```

### 5. 编译并运行 MPC 推理

确保 MP‑SPDZ 已经编译 `semi2k-party.x`。然后运行以下脚本，在两方上执行推理：

```bash
# 运行两层 FC 推理（需要提前设置 MODEL_DIR 和 INPUT_FILE 环境变量）
MODEL_DIR=mp_spdz_inputs/fc2 \
INPUT_FILE=mp_spdz_inputs/fc2/input_0.txt \
MP_SPDZ_DIR=../MP-SPDZ \
./run_fc2.sh

# 运行 ResNet‑50 推理（整网前向，计算量很大）
MODEL_DIR=mp_spdz_inputs/resnet50 \
INPUT_FILE=mp_spdz_inputs/resnet50/input_0.txt \
MP_SPDZ_DIR=../MP-SPDZ \
./run_resnet50.sh
```

脚本会将权重和输入复制到 `Player-Data` 目录下的相应位置，自动编译 MPC 程序并启动两方计算。完成后，会在终端输出预测标签（若运行完成）。

`run_fc2.sh` 会自动创建 `run_logs/<timestamp>/`，并输出：

- `compile.log`
- `party0.log`
- `party1.log`
- `summary.json`
- `summary.csv`

其中 `summary.json` / `summary.csv` 会尽量自动提取以下字段：`predicted_label`、`elapsed_time_seconds`、`party0_sent_mb`、`party1_sent_mb`、`total_sent_mb`、`rounds`、`triples`。如果某些字段在当前 MP‑SPDZ 日志格式里无法稳定提取，会写为 `null`（JSON）或 `N/A`（CSV）。

`run_resnet50.sh` 现在也输出同样格式的日志与 summary。对于 ResNet‑50，可选设置 `RUN_TIMEOUT_SECONDS`（例如 10 秒）做“闭环验证+结果落盘”：

```bash
MODEL_DIR=mp_spdz_inputs/resnet50 \
INPUT_FILE=mp_spdz_inputs/resnet50/input_0.txt \
MP_SPDZ_DIR=../mp-spdz-0.4.2 \
RUN_TIMEOUT_SECONDS=10 \
RUN_DIR=run_logs/20260407_resnet50_timeout10 \
./run_resnet50.sh
```

即使超时未完成，`summary.json` / `summary.csv` 也会生成，缺失字段写 `null`/`N/A`。

### 6. 批量评估（baseline FC2）

可以使用批量脚本自动评估多个样本并生成表格：

```bash
# 评估前 10 个样本
python3 scripts/eval_fc2_mpc.py \
    --model-dir mp_spdz_inputs/fc2 \
    --mp-spdz-dir ../mp-spdz-0.4.2 \
    --first_n 10

# 评估指定索引
python3 scripts/eval_fc2_mpc.py \
    --model-dir mp_spdz_inputs/fc2 \
    --mp-spdz-dir ../mp-spdz-0.4.2 \
    --indices 0,1,2,3
```

脚本会在 `eval_results/<timestamp>/` 下生成：

- `results.csv`（每个样本的预测/正确性/时间/通信）
- `summary.json`（accuracy、平均时间、平均通信量等汇总）

### 7. 记录时间和通信开销

MP‑SPDZ 提供若干运行时选项用于输出时间和通信统计。例如，可以在执行时添加 `-S` 输出通信量，或在 `compile.py` 阶段使用 `--bits` 调整精度。也可以通过观察 `semi2k-party.x` 的终端输出获取运行时间。为了获得更精确的测量，可以在运行脚本前后使用 `time` 命令，或修改 MP‑SPDZ 源码中的统计选项。

## baseline 与 ResNet‑50 整网实现的区别

| 项目             | baseline 两层 FC             | ResNet‑50 整网私有推理               |
|------------------|--------------------------------|--------------------------------------|
| 模型规模         | 输入 784 → 隐藏 128 → 输出 10 | 深层卷积网络，含 50 层、残差连接       |
| 实现复杂度       | 简单矩阵乘法 + ReLU           | 需要实现卷积、残差、全局平均池化等     |
| BN 处理          | 不含 BN                       | 推理前需要 BN folding 以减少计算       |
| 通信/计算开销    | 较低                           | 较高，随网络深度和输入尺寸增加         |
| 完整示例         | 已完整实现（可复现闭环）      | 整网前向已实现，MPC 运行成本很高       |

## 可能需要手动调整的部分

- MP‑SPDZ 版本的 API 可能因提交时间不同略有差异。如果在导入数据或运算时出现错误，可以查阅 [MP‑SPDZ 文档](https://mp-spdz.readthedocs.io/en/latest/) 或示例程序，适当修改 `.mpc` 文件中的输入和运算方式。
- 在 `fc2_mnist_infer.mpc` 中假定模型参数和输入均使用相同的小数位 `fractional_bits`。如果更改精度，请确保在导出模型和准备输入时一致，并在 `.mpc` 文件中使用 `sfix.set_precision(f, k)`（先小数位、后总位宽）。
- `resnet50_mnist_infer.mpc` 已实现整网前向（conv1/maxpool/layer1-4/GAP/fc/argmax），并按 Party 0/Party 1 语义读取参数与输入；在常见 CPU 环境下运行时间通常很长，建议先使用 `RUN_TIMEOUT_SECONDS` 验证日志闭环。
- ResNet 导出采用 MPC 友好布局：`conv.weight` 为 `OHWI`，`fc.weight` 为 `IO`，请使用当前 `export_fixed.py` 生成的 `meta.json` 与 `fixed_params.txt`，不要混用旧导出产物。
- baseline FC2 的精度常量在 `fc2_mnist_infer.mpc` 中写死为 `sfix.set_precision(16, 32)`，应与 `config/fc2_config.json` 中 `fractional_bits=16` 保持一致。

## 结语

通过本工程，你可以体验如何将经典深度学习模型部署到安全多方计算平台，实现私有推理。建议先从 baseline 两层全连接网络开始熟悉流程，再运行已实现的 ResNet‑50 整网私有推理，并结合真实日志分析精度、时间与通信开销之间的权衡。
