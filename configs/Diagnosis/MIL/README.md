# Diagnosis MIL / SPE 配置

本目录的 11 个 SPE 配置使用同一个开发集五折目录
`datasets/Diagnosis/Stains/Reinhard`，并统一设置：

- H-optimus-1 / 10x / Reinhard，输入维度 1536
- 5 折、seed 42、最多训练 50 epoch
- 以验证集 macro-F1 保存 best，early-stop patience 为 30
- `General.checkpoint.save_mode: every_epoch`
- SPE 稳定阈值 0.003，连续 epoch 最少 5 个，每折最多选择 5 个 checkpoint
- 输出到 `result/Diagnosis/MIL/HOptimus1_10x_Reinhard/<MODEL_NAME>/...`

## 推荐运行顺序

第一批核心基线：`AB_MIL`、`CLAM_SB_MIL`、`CLAM_MB_MIL`、`TRANS_MIL`、
`WIKG_MIL`。

第二批增加结构多样性：`MAMBA2D_MIL`、`AEM_MIL`、`MICO_MIL`、
`MSM_MIL`、`TDA_MIL`、`GDF_MIL`。

若论文报告的是 11 架构 SPE，第二批不是可选消融，最终也必须完成。只有
best+last checkpoint 的旧实验可以继续用于 best 单模型、Peak-All 或普通 soft-voting
对照，但不能重建逐 epoch 稳定区间，因此不能作为完整 SPE 输入。

`Supplementary/` 中的 Mean、Max、DSMIL、DTFD-MIL 和 RRT-MIL 仅作为补充表中的
锚点基线，不改变 11-member SPE，也不需要逐 epoch checkpoint。

训练配置不直接使用已填入独立测试集的 `datasets/Diagnosis/MIL`。当前框架在
`Train_Val_Test` 模式下会在每个 epoch 计算测试集指标，这与论文中“模型和集成规则
锁定后才进行独立测试”的设计不一致；完整 MIL CSV 应仅用于锁定后的推理。

Linux 训练环境还需确认：TransMIL 需要 `einops`；WiKG/GDF 需要
`torch-geometric`；Mamba2D/MSM 需要与 CUDA/PyTorch 匹配的 `mamba-ssm`。

# Supplementary MIL baselines

These five models are supplementary comparators and are not members of the
locked 11-architecture SPE.

- `MEAN_MIL` and `MAX_MIL`: parameter-light pooling sanity checks.
- `DS_MIL`: a widely recognized dual-stream classical MIL baseline.
- `DTFD_MIL`: a feature-distillation baseline with a substantially different
  training structure.
- `RRT_MIL`: a recent re-embedding/re-aggregation baseline.

All configurations use the selected H-optimus-1 / 10x / Reinhard
representation, the same patient-level five folds, seed 42, 50 epochs, and
validation macro-F1. They use `best_last` because they are reported as
supplementary single-model baselines rather than participating in stable-state
SPE aggregation.

Training points to `datasets/Diagnosis/Stains/Reinhard`, whose test columns are
empty. This prevents the independent test cohort from being evaluated during
each training epoch. The populated `datasets/Diagnosis/MIL` folds are reserved
for inference after all model and ensemble rules have been locked.


