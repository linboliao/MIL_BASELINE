import os
import glob
import json
import argparse
import pprint
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    roc_auc_score, f1_score, precision_score, recall_score
)

# 导入你真实的推理评估入口
from test_mil import test


def select_best_of_the_best(root_dir, target_metric='val_macro_f1', threshold=0.03):
    """
    唯一保留的抽象方法：扫描所有模型、所有Seed、所有Fold，筛选出最优配置
    """
    all_data = []
    model_list = [
        "AB_MIL", "TRANS_MIL", "CLAM_MB_MIL", "CLAM_SB_MIL", "WIKG_MIL",
        "MAMBA_MIL", "MAMBA2D_MIL", "AEM_MIL", "MICO_MIL", "MSM_MIL",
        "TDA_MIL", "GDF_MIL"
    ]

    best_log_files = []
    for model in model_list:
        pattern = os.path.join(root_dir, model, "*", "fold_*", "Best_Log_*.csv")
        best_log_files.extend(glob.glob(pattern))

    for b_log in best_log_files:
        try:
            parts = b_log.split(os.sep)
            exp_name, seed_dir, fold_name = parts[-4], parts[-3], parts[-2]
            parent_dir = os.path.dirname(b_log)

            df_best = pd.read_csv(b_log)
            full_log_path = b_log.replace("Best_Log_", "Log_")
            df_full = pd.read_csv(full_log_path)
            df_last = df_full.iloc[-1]

            b_val = df_best[target_metric].iloc[0]
            l_val = df_last[target_metric]

            # 稳定性优先策略：当 Last 与 Best 差距小于阈值时选择 Last
            if (b_val - l_val) < threshold:
                chosen_type, final_score, final_epoch = "Last", l_val, int(df_last['epoch'])
            else:
                chosen_type, final_score, final_epoch = "Best", b_val, int(df_best['epoch'].iloc[0])

            weight_name = f"{chosen_type}_EPOCH_{final_epoch}.pth"
            if not os.path.exists(os.path.join(parent_dir, weight_name)):
                candidates = glob.glob(os.path.join(parent_dir, f"{chosen_type}_*.pth"))
                weight_name = os.path.basename(candidates[0]) if candidates else "NOT_FOUND"

            all_data.append({
                "fold": fold_name,
                "exp": exp_name,
                "score": final_score,
                "weight_file": os.path.join(seed_dir, fold_name, weight_name),
                "type": chosen_type
            })
        except Exception as e:
            print(f"Error processing {b_log}: {e}")

    if not all_data:
        raise ValueError(f"未能在 {root_dir} 下匹配到任何有效的日志文件，请检查路径。")

    df = pd.DataFrame(all_data)
    final_configs = []

    print(f"\n{'Fold':<10} | {'Winner Model':<20} | {'Type':<6} | {target_metric}")
    print("-" * 65)

    for fold, group in df.groupby("fold"):
        winner = group.loc[group['score'].idxmax()]
        print(f"{fold:<10} | {winner['exp']:<20} | {winner['type']:<6} | {winner['score']:.4f}")

        final_configs.append({
            "fold": winner['fold'],
            "exp": winner['exp'],
            "weight_file": winner['weight_file'],
            "score": winner['score']
        })

    return final_configs


def calculate_metrics(y_true, y_pred, y_probs, prefix=''):
    """通用的多标签/二分类指标计算工具函数"""
    is_multiclass = y_probs.shape[1] > 2
    multi_class_param = 'ovr' if is_multiclass else 'raise'
    auc_probs = y_probs if is_multiclass else y_probs[:, 1]

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "quadratic_kappa": cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        "linear_kappa": cohen_kappa_score(y_true, y_pred, weights='linear')
    }

    for m in ['macro', 'micro', 'weighted']:
        metrics[f"{m}_auc"] = roc_auc_score(y_true, auc_probs, multi_class=multi_class_param, average=m)
        metrics[f"{m}_f1"] = f1_score(y_true, y_pred, average=m)
        metrics[f"{m}_pre"] = precision_score(y_true, y_pred, average=m, zero_division=0)
        metrics[f"{m}_recall"] = recall_score(y_true, y_pred, average=m, zero_division=0)

    return {f"{prefix}{k}": v for k, v in metrics.items()}


def parse_probs(p_str):
    """解析存储在 CSV 中的概率字符串"""
    return np.fromstring(p_str.strip('[]'), sep=' ')


# --- 定义命令行参数 ---
parser = argparse.ArgumentParser(description="MIL 跨模型筛选、多折推理与加权集成一键化 Pipeline 脚本")
parser.add_argument('--root_dir', type=str, default=None, help='实验结果的根目录')
parser.add_argument('--dataset_root', type=str, default=None, help='数据集根目录')
parser.add_argument('--exp_name', type=str, default=None, help='外部验证集或当前实验的子名称')
parser.add_argument('--output_root', type=str, default=None, help='推理结果输出根目录')
parser.add_argument('--target_metric', type=str, default='val_macro_f1', help='用于筛选的核心基准指标')
parser.add_argument('--stability_threshold', type=float, default=0.03, help='Best与Last指标差距在该阈值内时优先选Last')

if __name__ == '__main__':
    args = parser.parse_args()


    if args.exp_name:
        test_csv_path = os.path.join(args.dataset_root, args.exp_name, 'test.csv')
        ensemble_output_dir = os.path.join(args.output_root, args.exp_name)
    else:
        # test_csv_path = os.path.join(args.dataset_root, 'test.csv')
        test_csv_path = 'datasets/Contrast/test.csv'
        ensemble_output_dir = args.output_root
    os.makedirs(ensemble_output_dir, exist_ok=True)

    print(f"正在基于基准指标 [{args.target_metric}] 筛选各折最优模型配置...")
    best_configs = select_best_of_the_best(
        root_dir=args.root_dir,
        target_metric=args.target_metric,
        threshold=args.stability_threshold
    )

    print("\n" + "=" * 80)
    print(" 最终生成的最佳模型配置 ")
    print("=" * 80)
    pprint.pprint(best_configs, sort_dicts=False)

    print("\n" + "=" * 80)
    print(" 开始多折串行推理 ")
    print("=" * 80)
    infer_csv_paths = []

    for config in best_configs:
        fold, exp = config['fold'], config['exp']
        current_log_dir = os.path.join(ensemble_output_dir, fold)
        weight_path = os.path.join(args.root_dir, exp, config['weight_file'])

        print(f"\n==> 推理 {fold} (模型: {exp}) ...")
        args_infer = argparse.Namespace(
            yaml_path=f'configs/Contrast/MIL/{exp}.yaml',
            test_dataset_csv=test_csv_path,
            model_weight_path=weight_path,
            test_log_dir=current_log_dir
        )

        # 调用外部的真实 test 函数
        test(args_infer)

        # 读取当前折推理结果并计算指标
        df_fold = pd.read_csv(os.path.join(current_log_dir, 'Infer_Result.csv'))
        fold_probs = np.array([parse_probs(p) for p in df_fold['probs']])
        fold_metrics = calculate_metrics(
            df_fold['label'].values,
            df_fold['prediction'].values,
            fold_probs,
            prefix=f'{args.target_metric}_'
        )

        pd.DataFrame([fold_metrics]).to_csv(os.path.join(current_log_dir, 'Best_Result.csv'), index=False)
        infer_csv_paths.append(os.path.join(current_log_dir, 'Infer_Result.csv'))

    # 4. 五折结果加权聚合 (Ensemble)
    print("\n" + "=" * 80)
    print(" 开始五折结果加权聚合与集成 ")
    print("=" * 80)

    # 计算集成权重 (根据验证集分数归一化)
    scores = np.array([config['score'] for config in best_configs])
    weights = scores / np.sum(scores)

    print("-" * 45)
    print(f"{'Fold':<10} | {'Weight (权重)':<15} | {'Score (得分)'}")
    print("-" * 45)
    for i, config in enumerate(best_configs):
        print(f"{config['fold']:<10} | {weights[i]:.4f}          | {scores[i]:.4f}")
    print("-" * 45)

    # 加载各折推理出的概率
    dfs = [pd.read_csv(p) for p in infer_csv_paths]
    all_probs_list = [np.array([parse_probs(p) for p in df['probs']]) for df in dfs]

    # 执行加权矩阵聚合
    weighted_probs = np.zeros_like(all_probs_list[0])
    for i in range(len(all_probs_list)):
        weighted_probs += all_probs_list[i] * weights[i]

    # 产生集成后的硬标签预测
    y_true = dfs[0]['label'].values
    y_pred = np.argmax(weighted_probs, axis=1)

    # 保存加权集成大表
    final_df = dfs[0][['slide_id', 'label']].copy()
    final_df['probs'] = [str(list(p)) for p in weighted_probs]
    final_df['prediction'] = y_pred

    save_path = os.path.join(ensemble_output_dir, 'Ensemble_Weighted_Result.csv')
    final_df.to_csv(save_path, index=False)

    # 计算最终集成指标并导出 JSON
    final_metrics = calculate_metrics(y_true, y_pred, weighted_probs)
    final_metrics_json = {k: {"mean": float(v), "std": 0.0} for k, v in final_metrics.items()}

    json_save_path = os.path.join(ensemble_output_dir, 'merge_5_fold_metrics2.json')
    with open(json_save_path, 'w') as f:
        json.dump(final_metrics_json, f, indent=4)

    print(f"\n[成功] 加权集成完成！")
    print(f"➔ 预测表格已保存至: {save_path}")
    print(f"➔ 最终统计指标已保存至: {json_save_path}")