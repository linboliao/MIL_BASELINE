# WSI 表征配置数据划分

PFM、倍率和染色归一化属于模型开发阶段的 WSI 表征选择，只能使用
`train_val.csv`。独立内部测试集和外部测试集必须等到 PFM、倍率、染色方法、
MIL 模型及阈值全部锁定后再使用。

## 划分规则

- 5 折，随机种子 42。
- `slide_id` 必须与真实 `.pt` 文件名完全一致，包括 `有癌/无癌`、空格、
  下划线和额外句点等后缀。
- 按患者分组；提取患者标识时先移除末尾的 `有癌/无癌` 文件注释，再取
  第一个 `.` 之前的内容。此处理只用于分组，不修改特征文件路径。
- 同一患者的全部 WSI 只能位于同一折，不能同时出现在训练集和验证集。
- 在患者不可拆分的前提下，平衡每折的阴性 WSI、阳性 WSI 和患者数。
- 三类配置实验复用同一份 fold assignment，只改变特征路径。
- 每个 fold CSV 使用项目已有的六列格式，`test_*` 列为空。

## 输出结构

```text
datasets/Diagnosis/
  PFM/{CONCH,h-optimus-1,mstar,omiclip,UNI,UNI2,virchow2}/
  Mag/{20x,10x,5x}/
  Stains/{Macenko,Reinhard,Vahadane}/
```

每个最内层目录包含 5 个 fold CSV。YAML 的
`Dataset.dataset_root_dir` 必须指向对应的最内层目录。

## 数据质量记录

生成器不修改原始 `train_val.csv`。重复行与标签冲突记录分别保存在：

- `wsi_representation_exclusions.csv`
- `wsi_representation_split_summary.json`
- `wsi_representation_fold_assignments.csv`
- `train_val_suffix_update_audit.csv`：记录由 `2.csv` 补回的原值和新值。

冲突切片应由病理记录复核后，再决定是否加入后续版本。

## 独立测试集警告

不要使用 `train_val_test.csv` 进行表征选择，因为它已经混入独立测试数据。
本批 fold CSV 不包含测试数据。正式报告独立内部测试结果前，应先确认开发集
与测试集之间没有重复切片或患者重叠。

## 重新生成

```powershell
& 'D:\anaconda3\envs\maixin\python.exe' `
  split_scripts\build_diagnosis_wsi_representation_splits.py `
  --source-csv datasets/Diagnosis/train_val.csv `
  --output-root datasets/Diagnosis `
  --seed 42 `
  --folds 5 `
  --conflict-policy exclude
```
