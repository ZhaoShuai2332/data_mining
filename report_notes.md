# 实验报告笔记

> 本文档旨在为撰写《大数据隐私保护课程》实验报告提供参考框架和主要内容。撰写时可根据实际实验结果补充、调整文字和表格，最终整理为正式报告提交。

## 实验目的

1. 理解安全多方计算（MPC）在隐私保护机器学习中的基本原理及挑战。
2. 掌握 MP‑SPDZ 框架下两方半诚实模型的编程方法，包括安全乘法、比较和激活函数实现。
3. 实现一个简单但功能完整的私有神经网络推理系统：以两层全连接网络为基线，实现安全乘加和 ReLU；并探索更深的 ResNet‑50 推理骨架。
4. 在 MNIST 数据集上测试私有推理的准确率、运行时间和通信开销，分析精度与开销的权衡。

## 实验原理

### 安全乘法与安全加法

在两方半诚实 MPC 中，双方分别持有秘密共享的输入值，计算结果也以共享形式返回。对于加法，由于秘密共享的线性性，每方本地相加即可获得结果的共享；乘法需要消耗预处理阶段生成的乘法三元组 (Beaver Triple)。在 MP‑SPDZ 的高级接口中，使用 `sfix` 类型的普通乘法运算符 `*` 即可调用底层协议自动完成安全乘法。

### 安全 ReLU

ReLU 激活函数定义为 $\mathrm{ReLU}(x) = \max(x, 0)$。在安全计算中需要比较一个秘密值是否大于 0。MP‑SPDZ 提供 `>` 运算返回一个秘密布尔值 (`sint`)；将其与原值相乘即可实现 ReLU：

```python
tmp = x * (x > 0)
```

其中 `(x > 0)` 返回 1 表示真，0 表示假，自动提升为与 `sfix` 可乘。需要注意的是，比较操作相较于线性运算开销更大，是推理通信瓶颈之一。

### BatchNorm Folding

批归一化 (Batch Normalization) 在推理阶段对特征进行线性变换：

\[\hat{x} = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta\]

这一操作含有除法和平方根，在 MPC 中计算昂贵。由于 BN 层参数 (\(\mu, \sigma^2, \gamma, \beta\)) 在推理阶段是固定的，可将其吸收进前一层卷积的权重和偏置中：

\[\mathbf{W}' = \mathbf{W} \cdot \mathrm{diag}\left(\frac{\gamma}{\sqrt{\sigma^2 + \varepsilon}}\right),\quad \mathbf{b}' = \frac{\gamma}{\sqrt{\sigma^2 + \varepsilon}}(\mathbf{b} - \mu) + \beta\]

融合后可将 BN 层替换为恒等映射，避免推理时的非线性代价。代码见 `fold_bn.py`。

### 两方私有推理流程

1. **模型训练**：在普通环境下使用 PyTorch 训练模型（基线 FC 或 ResNet‑50）。
2. **导出权重**：将训练所得权重与偏置保存，并通过 `export_fixed.py` 按指定小数位数量化后导出为 `sfix.input_from()` 可读取的实数文本。
3. **输入准备**：客户端将测试样本同样按相同小数位数量化并导出为实数文本输入。
4. **编译 MPC 程序**：使用 MP‑SPDZ 的 `compile.py` 编译 `.mpc` 程序，生成相应字节码。
5. **执行协议**：两方分别运行 `semi2k-party.x`（或其他协议实现），将模型参数和输入写入 `Player-Data/Input-PX-Y`。协议执行完毕后，双方同时获得模型输出（预测结果）。

## baseline 实现说明

### 网络结构

基线模型为两层全连接网络：输入向量维度为 784 (28×28)，隐藏层宽度默认为 128，输出层为 10 类。训练阶段使用 ReLU 激活和交叉熵损失；推理阶段在 MPC 中实现安全矩阵乘加及安全 ReLU。

### MPC 代码要点

* `fc2_mnist_infer.mpc` 中使用 `sfix.Array` 读取权重和输入：

  ```python
  W1 = sfix.Array(n_in * n_hidden)
  W1.input_from(0)  # Party 0 输入模型权重
  x  = sfix.Array(n_in)
  x.input_from(1)   # Party 1 输入测试样本
  ```

* 通过嵌套 `for_range` 循环计算矩阵乘法，再加上偏置并应用 ReLU：

  ```python
  tmp.update(tmp + x[i] * W1[i * n_hidden + j])
  tmp.update(tmp + b1[j])
  z1[j] = tmp * (tmp > 0)
  ```

* 使用 `sint.if_else()` 实现 argmax：逐元素比较，若当前值更大则更新最大值及索引。

### 运行测试

导出权重并准备输入后，通过 `run_fc2.sh` 即可执行私有推理。终端会输出预测标签。根据实验设置，可进一步记录运行时间和通信数据包大小（见后续表格）。

## ResNet‑50 扩展说明

### 模型修改与训练

由于 ResNet‑50 设计用于 224×224 彩色图像，需要做如下修改以适应 MNIST：

1. 将第一层卷积的输入通道数改为 1，并用预训练权重对多个通道求平均初始化。
2. 将最后的全连接层输出维度改为 10（MNIST 类别数）。
3. 将输入图片 resize 至 224×224 并使用与训练时相同的均值/方差归一化。

训练脚本 `train_export_resnet50_mnist.py` 提供了这些修改，并支持冻结前几层来加速训练。

### BN 融合与权重导出

利用 `fold_bn.py` 将 BN 融合进卷积层后，使用 `export_fixed.py` 将模型权重转为固定点整数。由于 ResNet‑50 的网络深度较大，导出文件中的参数数量巨大，需要在 MPC 程序中按照导出顺序逐个加载。当前的 `resnet50_mnist_infer.mpc` 提供卷积和 ReLU 的示例函数，但尚未完成整个模型的推理逻辑。完成实现需要：

1. 根据 `meta.json` 读取每个权重的形状和顺序，对应到模型结构。
2. 编写嵌套循环实现多通道卷积、残差加法和全局平均池化等操作。
3. 考虑使用 `for_range_opt` 或多线程编译接口优化循环，减少通信开销。

## 运行结果记录（2026-04-07 实测，基于真实运行）

### 本次运行日志与环境

1. 代码目录：`/Users/shuaizhao/workspace/mp_spdz_mnist_project`
2. MP-SPDZ 目录：`/Users/shuaizhao/workspace/mp-spdz-0.4.2`
3. 协议可执行文件：`/Users/shuaizhao/workspace/mp-spdz-0.4.2/semi2k-party.x`
4. baseline FC2 参数与元数据：
   - `models/fc2/fc2_params.npz`
   - `mp_spdz_inputs/fc2/fixed_params.txt`
   - `mp_spdz_inputs/fc2/meta.json`

### baseline (FC2) 单样本链路检查

使用 `prepare_input.py` + `run_fc2.sh` 对 `index=0` 进行环境核验，结果目录：

- `run_logs/20260407_envcheck_single/`

关键结果：

- predicted_label: `7`
- elapsed_time_seconds: `5.87248`
- party0_sent_mb: `211.931`
- party1_sent_mb: `254.427`
- total_sent_mb: `466.358`
- rounds: `306236`
- triples: `109724`

### baseline (FC2) 批量评估结果

#### 批量实验 A（前 10 个样本）

- 输出目录：`eval_results/20260407_batch_n10/`
- 汇总文件：
  - `eval_results/20260407_batch_n10/summary.json`
  - `eval_results/20260407_batch_n10/results.csv`

汇总指标：

- sample_count: `10`
- accuracy: `0.9`
- avg_time_seconds: `5.874504`
- avg_total_sent_mb: `466.358`

#### 批量实验 B（前 20 个样本）

- 输出目录：`eval_results/20260407_batch_n20/`
- 汇总文件：
  - `eval_results/20260407_batch_n20/summary.json`
  - `eval_results/20260407_batch_n20/results.csv`

汇总指标：

- sample_count: `20`
- accuracy: `0.95`
- avg_time_seconds: `5.82932`
- avg_total_sent_mb: `466.358`

两组批量评估均出现同一错分样本（`index=8`，真实标签 `5`，预测 `6`）。

### ResNet‑50 结果状态（诚实边界说明）

当前 `Programs/Source/resnet50_mnist_infer.mpc` 仍为骨架程序，尚未完成完整私有前向推理实现。因此本次报告不提供 ResNet‑50 的正式准确率、平均时间和通信量实验结果，避免将占位输出误写为实验结论。

## 时间与通信开销分析（基于本次 FC2 实测）

1. **FC2 批量结果已闭环可复现**：通过 `scripts/eval_fc2_mpc.py` 可稳定生成 `results.csv` 与 `summary.json`，并得到可直接写入报告的 accuracy/平均耗时/平均通信量。
2. **通信量稳定**：在本次环境下，单样本总通信量基本稳定在 `466.358 MB`（party0=`211.931 MB`，party1=`254.427 MB`）。
3. **运行时间稳定在秒级**：前 10/20 样本的平均耗时分别为 `5.874504 s` 和 `5.82932 s`，满足基线实验可重复测量要求。
4. **固定点配置一致性有效**：`config/fc2_config.json` 与 `meta.json` 的 `fractional_bits=16` 一致，`run_fc2.sh` 具备运行前校验。

## 实验总结

本次已真实完成 baseline FC2 的单样本链路验证与两组批量评估（`n=10`、`n=20`），并沉淀了可直接引用的结果文件（`summary.json`、`results.csv`）。因此，FC2 部分的“准确率、时间、通信量”实验结果已补全。ResNet‑50 仍处于骨架阶段，尚未完成正式实验结果。

---

_提示：撰写正式报告时，建议在每个小节中插入适当的图表或代码片段解释核心实现，保持逻辑清晰、表达准确。_
