# Diagnosis MIL_Mix 配置

本目录由 `configs/Diagnosis/MIL` 的 16 个模型配置派生，模型结构、优化器、
训练轮数、early stopping 和 checkpoint 策略均保持不变。

与原配置相比，仅隔离了实验的数据和输出命名空间：

- 数据目录：`datasets/Diagnosis/MIL_Mix`
- 对比组：`MIL_Mix`（补充模型为 `MIL_Mix_supplementary`）
- 输出目录：`result/Diagnosis/MIL_Mix`

`MIL_Mix` 的五折 CSV 均包含训练集、验证集和测试集，因此使用当前训练流程时，
每个 epoch 都会计算测试集指标。若测试集仅允许用于最终锁定评估，请不要用这些
配置进行模型选择。
