import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix

# 导入你原有的核心工具函数和推理入口
from test_mil import test


def parse_probs(p_str):
    """解析存储在 CSV 中的概率字符串"""
    return np.fromstring(p_str.strip('[]'), sep=' ')


def run_real_inference_bootstrap_by_types(best_configs, original_test_csv, root_dir, output_root, n_bootstrap=100, seed=42):
    """
    分亚型、动态推理版 Bootstrap Pipeline：
    每一轮对完整数据集进行有放回抽样，驱动 5 折模型执行推理，
    然后在一个轮次内同时提取 all, CNB, RP, TURP 的预测结果并独立计算各自的 Accuracy 分布。
    """
    print(f"\n" + "=" * 80)
    print(f" 启动多亚型动态推理版 Bootstrap Pipeline (共 {n_bootstrap} 次完整迭代) ")
    print(f" 评估目标亚型: ['all', 'CNB', 'RP', 'TURP']")
    print("=" * 80)

    # 1. 读取最原始的测试集表格
    df_orig = pd.read_csv(original_test_csv)
    n_samples = len(df_orig)

    # 验证 type 列是否存在
    if 'type' not in df_orig.columns:
        raise ValueError(f"原始测试集 {original_test_csv} 中缺少 'type' 列，无法按亚型进行过滤！")

    # 2. 设置随机种子
    rng = np.random.default_rng(seed)

    # 初始化 4 个组的结果存放容器
    target_types = ['all', 'CNB', 'RP', 'TURP']
    bootstrap_results = {t: [] for t in target_types}

    # 创建临时测试集和输出存放目录
    tmp_dir = os.path.join(output_root, 'tmp_bootstrap_datasets')
    os.makedirs(tmp_dir, exist_ok=True)

    # 归一化多折权重
    scores = np.array([cfg['score'] for cfg in best_configs])
    weights = scores / np.sum(scores)

    # 3. 核心循环开始
    for b in range(1, n_bootstrap + 1):
        print(f"\n[Bootstrap {b}/{n_bootstrap}] ----------------------------------------")

        # 产生有放回的随机索引，并构建当前轮次的完整临时测试集 CSV
        boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
        df_boot = df_orig.iloc[boot_indices].copy().reset_index(drop=True)

        tmp_test_csv = os.path.join(tmp_dir, f'test_boot_{b}.csv')
        df_boot.to_csv(tmp_test_csv, index=False)

        # 驱动 5 折最优模型在这个重采样测试集上进行单轮推理
        round_output_dir = os.path.join(output_root, f'boot_run_{b}')
        infer_csv_paths = []

        for idx, config in enumerate(best_configs):
            fold, exp = config['fold'], config['exp']
            current_log_dir = os.path.join(round_output_dir, fold)
            weight_path = os.path.join(root_dir, exp, config['weight_file'])

            args_infer = argparse.Namespace(
                yaml_path=f'/NAS3/lbliao/Code-138/MIL_BASELINE/configs/Contrast/MIL/{exp}.yaml',
                test_dataset_csv=tmp_test_csv,  # 喂入当前的临时数据集
                model_weight_path=weight_path,
                test_log_dir=current_log_dir
            )

            # 调用真实前向推理
            # test(args_infer)

            infer_res_path = os.path.join(current_log_dir, 'Infer_Result.csv')
            if os.path.exists(infer_res_path):
                infer_csv_paths.append(infer_res_path)

        # 4. 提取当前轮次的预测数据，按亚型动态计算指标
        if len(infer_csv_paths) == len(best_configs):
            # 读取 5 折单模结果
            dfs = [pd.read_csv(p) for p in infer_csv_paths]

            # 为了确保切分各亚型时行索引和 type 列完美对齐，直接把原始抽样表的 type 列拼接到第一折结果上
            # （推理输出的顺序和传入的 tmp_test_csv 的行顺序是严格一致的）
            df_merged_base = dfs[0].copy()
            df_merged_base['type'] = df_boot['type'].values

            # 解析 5 折模型输出的概率矩阵
            all_probs_list = [np.array([parse_probs(p) for p in df['probs']]) for df in dfs]

            # 加权求和融合，得到完整抽样集的集成概率
            weighted_probs = np.zeros_like(all_probs_list[0])
            for i in range(len(all_probs_list)):
                weighted_probs += all_probs_list[i] * weights[i]

            # 将集成后的预测硬标签和真实标签挂载到 base 视图上
            df_merged_base['weighted_pred'] = np.argmax(weighted_probs, axis=1)

            # 针对 target_types 里的每种亚型切片计算 Accuracy
            for t in target_types:
                if t == 'all':
                    df_sub = df_merged_base
                else:
                    # 严格过滤对应亚型（去除前后空格并统一转大写防止 typo 干扰）
                    df_sub = df_merged_base[df_merged_base['type'].astype(str).str.strip() == t]

                if len(df_sub) == 0:
                    print(f"  ➔ 警告: 亚型 [{t}] 在本轮重采样中未被抽到样本，跳过该指标。")
                    continue

                y_true_sub = df_sub['label'].values
                y_pred_sub = df_sub['weighted_pred'].values
                acc_sub = accuracy_score(y_true_sub, y_pred_sub)
                bacc_sub = balanced_accuracy_score(y_true_sub, y_pred_sub)
                macro_pre_sub = precision_score(y_true_sub, y_pred_sub, average='macro', zero_division=0)
                macro_rec_sub = recall_score(y_true_sub, y_pred_sub, average='macro', zero_division=0)

                tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub, labels=[0, 1]).ravel()

                # 计算敏感性和特异性 (带 0 分母保护)
                sens_sub = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec_sub = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                # 存入结果
                bootstrap_results[t].append({
                    'bootstrap_id': b,
                    'accuracy': acc_sub,
                    'bacc': bacc_sub,
                    'macro_pre': macro_pre_sub,
                    'macro_recall': macro_rec_sub,
                    'sensitivity': sens_sub,
                    'specificity': spec_sub
                })
        else:
            print(f"[-] 错误: Bootstrap {b} 的某些折推理失败，跳过该轮所有亚型指标。")

        # 及时删除已经用完的临时测试集文件，节约磁盘空间
        if os.path.exists(tmp_test_csv):
            os.remove(tmp_test_csv)

    # 5. 汇总保存与多亚型统计看板打印
    print("\n" + "=" * 70)
    # print(f" {"Subtype":<10} | {"Mean Acc":<10} | {"Std Dev":<10} | {"95% Confidence Interval":<20}")
    print("-" * 70)

    for t in target_types:
        if not bootstrap_results[t]:
            continue

        res_df = pd.DataFrame(bootstrap_results[t])

        # 为每种亚型单独保存一份 CSV 结果明细
        csv_save_path = os.path.join(output_root, f'real_inference_bootstrap_100_accuracy_{t}.csv')
        res_df.to_csv(csv_save_path, index=False)

        # 统计计算
        accs = res_df['accuracy'].values
        mean_val = np.mean(accs)
        std_val = np.std(accs)
        ci_lower = np.percentile(accs, 2.5)
        ci_upper = np.percentile(accs, 97.5)

        print(f" {t:<10} | {mean_val:.4f}     | {std_val:.4f}     | [{ci_lower:.4f}, {ci_upper:.4f}]")

    print("=" * 70)
    print(f"[成功] 各亚型 100 次明细已分别保存至 {output_root}/real_inference_bootstrap_100_accuracy_*.csv\n")


if __name__ == '__main__':
    BEST_CONFIGS = [
        {'fold': 'fold_1', 'exp': 'DTFD_MIL', 'weight_file': 'seed_42_2026-04-09-01-36/fold_1/Best_EPOCH_9.pth', 'score': 0.9959344708803972},
        {'fold': 'fold_2', 'exp': 'AMD_MIL', 'weight_file': 'seed_42_2026-04-09-04-07/fold_2/Best_EPOCH_1.pth', 'score': 0.996975887206601},
        {'fold': 'fold_3', 'exp': 'MAMBA_MIL', 'weight_file': 'seed_42_2026-04-10-00-07/fold_3/Best_EPOCH_1.pth', 'score': 0.9957742529840584},
        {'fold': 'fold_4', 'exp': 'DTFD_MIL', 'weight_file': 'seed_42_2026-04-09-01-36/fold_4/Best_EPOCH_2.pth', 'score': 0.9985780661699912},
        {'fold': 'fold_5', 'exp': 'ILRA_MIL', 'weight_file': 'seed_42_2026-04-08-23-34/fold_5/Best_EPOCH_26.pth', 'score': 0.9981177412895474}
    ]
    # BEST_CONFIGS = [
    #     {'fold': 'fold_1', 'exp': 'CLAM_MB_MIL', 'weight_file': 'seed_42_2026-04-08-22-15/fold_1/Best_EPOCH_5.pth', 'score': 0.966375640016248},
    #     {'fold': 'fold_2', 'exp': 'CLAM_MB_MIL', 'weight_file': 'seed_42_2026-04-08-22-15/fold_2/Best_EPOCH_1.pth', 'score': 0.9798621435329098},
    #     {'fold': 'fold_3', 'exp': 'CLAM_MB_MIL', 'weight_file': 'seed_42_2026-04-08-22-15/fold_3/Best_EPOCH_3.pth', 'score': 0.9528606509217458},
    #     {'fold': 'fold_4', 'exp': 'CLAM_MB_MIL', 'weight_file': 'seed_42_2026-04-08-22-15/fold_4/Best_EPOCH_5.pth', 'score': 0.9731057719305284},
    #     {'fold': 'fold_5', 'exp': 'CLAM_MB_MIL', 'weight_file': 'seed_42_2026-04-08-22-15/fold_5/Best_EPOCH_7.pth', 'score': 0.9776231477773328}
    # ]

    # 修改为包含 type 分类信息的全新原始测试集路径
    ORIGINAL_TEST_CSV = '/NAS2/lbliao/Code-138/MIL_BASELINE/datasets/Contrast/test_new.csv'
    ROOT_DIR = 'MIL'
    OUTPUT_ROOT = 'bootstrap/spe'

    run_real_inference_bootstrap_by_types(
        best_configs=BEST_CONFIGS,
        original_test_csv=ORIGINAL_TEST_CSV,
        root_dir=ROOT_DIR,
        output_root=OUTPUT_ROOT,
        n_bootstrap=100,
        seed=42
    )
