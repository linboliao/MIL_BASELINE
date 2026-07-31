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

def select_standard_ensemble(root_dir, target_metric='val_macro_f1'):
    """
    【行业公认标准集成对照组获取方法】
    完全剥离高级筛选策略：
    1. 强制不进行 Best/Last 动态切换，100% 锁定官方原生的 'Best' 权重。
    2. 针对某一固定基线模型（默认提取基础 Best），生成最干净的 5 折对照配置。
    """
    all_data = []

    # 动态匹配所有子目录下的 Best_Log_*.csv
    search_pattern = os.path.join(root_dir, "*", "*", "fold_*", "Best_Log_*.csv")
    best_log_files = glob.glob(search_pattern)

    if not best_log_files:
        raise ValueError(f"未能在 {root_dir} 下匹配到任何有效的日志文件，请检查路径规则: {search_pattern}")

    for b_log in best_log_files:
        try:
            parent_dir = os.path.dirname(b_log)  # .../model/seed/fold
            fold_dir = os.path.basename(parent_dir)  # fold_x
            seed_dir = os.path.basename(os.path.dirname(parent_dir))  # seed_xxx
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(parent_dir)))  # model_name

            df_best = pd.read_csv(b_log)

            # ==========================================
            # 🛠️ 【核心退化点 1】：拒绝 Best/Last 稳定性策略
            # 强制不读取 Full Log，100% 锁定初始官方硬性指标最高的 Best
            # ==========================================
            chosen_type = "Best"
            final_score = df_best[target_metric].iloc[0]
            final_epoch = int(df_best['epoch'].iloc[0])

            # 精准匹配对应的原生 Best 权重文件
            weight_name = f"{chosen_type}_EPOCH_{final_epoch}.pth"
            if not os.path.exists(os.path.join(parent_dir, weight_name)):
                candidates = glob.glob(os.path.join(parent_dir, f"{chosen_type}_*.pth"))
                weight_name = os.path.basename(candidates[0]) if candidates else f"{chosen_type}_NOT_FOUND.pth"

            all_data.append({
                "fold": fold_dir,
                "exp": exp_name,
                "score": final_score,
                "weight_file": os.path.join(seed_dir, fold_dir, weight_name),
                "type": chosen_type
            })
        except Exception as e:
            print(f"[-] 解析错误 {b_log}: {e}")

    df = pd.DataFrame(all_data)

    # ==========================================
    DEFAULT_BASELINE_MODEL = "CLAM_MB_MIL"

    df_baseline = df[df['exp'] == DEFAULT_BASELINE_MODEL]

    if df_baseline.empty:
        # 容错：如果指定的模型不存在，则默认使用当前数据集里分最高的单一模型做标准集成
        print(f"[提示] 未找到指定的 {DEFAULT_BASELINE_MODEL}，将自动选择样本最多的基础单模。")
        most_common_model = df['exp'].value_counts().index[0]
        df_baseline = df[df['exp'] == most_common_model]

    # 按 Fold 分组，确保每一折只保留最原始、性能最高的那一个原生 Best
    best_indices = df_baseline.groupby("fold")["score"].idxmax()
    standard_df = df_baseline.loc[best_indices].sort_values("fold")

    # --- 打印漂亮的 Standard 看板 ---
    print("\n" + "=" * 75)
    print(f"【Standard Ensemble 看板】(已剥离 SPE 策略，锁定原生 Best)")
    print("-" * 75)
    print(f"{'Fold':<10} | {'Baseline Model':<20} | {'Type':<6} | {target_metric:<12}")
    print("-" * 75)

    standard_configs = []
    for _, row in standard_df.iterrows():
        print(f"{row['fold']:<10} | {row['exp']:<20} | {row['type']:<6} | {row['score']:.4f}")

        standard_configs.append({
            "fold": row['fold'],
            "exp": row['exp'],
            "weight_file": row['weight_file'],
            "score": row['score']
        })
    print("=" * 75)

    return standard_configs

def select_best_of_the_best(root_dir, target_metric='val_macro_f1', threshold=0.03):
    """
    动态扫描 root_dir 下所有模型、Seed 和 Fold，自动筛选出最稳定且最优的配置。
    融合了动态路径解析与 Best/Last 稳定性优先选择策略。
    """
    all_data = []

    # 动态匹配所有子目录下的 Best_Log_*.csv，不再硬编码模型列表
    search_pattern = os.path.join(root_dir, "*", "*", "fold_*", "Best_Log_*.csv")
    best_log_files = glob.glob(search_pattern)

    if not best_log_files:
        raise ValueError(f"未能在 {root_dir} 下匹配到任何有效的日志文件，请检查路径规则: {search_pattern}")

    for b_log in best_log_files:
        try:
            # 安全的路径解析，避免直接使用固定负数索引导致切片错位
            parent_dir = os.path.dirname(b_log)  # .../model/seed/fold
            fold_dir = os.path.basename(parent_dir)  # fold_x
            seed_dir = os.path.basename(os.path.dirname(parent_dir))  # seed_xxx
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(parent_dir)))  # model_name

            # 读取 Best 和 Full 运行日志
            df_best = pd.read_csv(b_log)
            full_log_path = os.path.join(parent_dir, os.path.basename(b_log).replace("Best_Log_", "Log_"))

            if not os.path.exists(full_log_path):
                # 如果没有完整的 Log_ 文件，退化为仅使用 Best
                b_val = df_best[target_metric].iloc[0]
                chosen_type, final_score, final_epoch = "Best", b_val, int(df_best['epoch'].iloc[0])
            else:
                df_full = pd.read_csv(full_log_path)
                df_last = df_full.iloc[-1]
                b_val = df_best[target_metric].iloc[0]
                l_val = df_last[target_metric]

                # 稳定性优先策略：当 Last 与 Best 差距小于阈值时选择 Last，避免临近 Epoch 剧烈震荡
                if (b_val - l_val) < threshold:
                    chosen_type, final_score, final_epoch = "Last", l_val, int(df_last['epoch'])
                else:
                    chosen_type, final_score, final_epoch = "Best", b_val, int(df_best['epoch'].iloc[0])

            # 精准匹配对应的权重文件
            weight_name = f"{chosen_type}_EPOCH_{final_epoch}.pth"
            if not os.path.exists(os.path.join(parent_dir, weight_name)):
                # 如果找不到带 Epoch 后缀的，尝试模糊匹配该类型的 pth 文件
                candidates = glob.glob(os.path.join(parent_dir, f"{chosen_type}_*.pth"))
                weight_name = os.path.basename(candidates[0]) if candidates else f"{chosen_type}_NOT_FOUND.pth"

            all_data.append({
                "fold": fold_dir,
                "exp": exp_name,
                "score": final_score,
                "weight_file": os.path.join(seed_dir, fold_dir, weight_name),
                "type": chosen_type
            })
        except Exception as e:
            print(f"[-] 解析错误 {b_log}: {e}")

    df = pd.DataFrame(all_data)

    # 按照 Fold 分组，并找出目标指标最大值的索引
    best_indices = df.groupby("fold")["score"].idxmax()
    winner_df = df.loc[best_indices].sort_values("fold")

    # --- 打印漂亮的终端看板 ---
    print("\n" + "=" * 75)
    print(f"{'Fold':<10} | {'Winner Model':<20} | {'Type':<6} | {target_metric:<12}")
    print("-" * 75)

    final_configs = []
    for _, row in winner_df.iterrows():
        print(f"{row['fold']:<10} | {row['exp']:<20} | {row['type']:<6} | {row['score']:.4f}")

        final_configs.append({
            "fold": row['fold'],
            "exp": row['exp'],
            "weight_file": row['weight_file'],
            "score": row['score']
        })
    print("=" * 75)

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
parser.add_argument('--root_dir', type=str, default='result/Contrast/MIL', help='实验结果的根目录')
parser.add_argument('--dataset_root', type=str, default=None, help='数据集根目录')
parser.add_argument('--exp_name', type=str, default=None, help='外部验证集或当前实验的子名称')
parser.add_argument('--output_root', type=str, default='ensemble_outputs', help='推理结果输出根目录')
parser.add_argument('--target_metric', type=str, default='val_macro_auc', help='用于筛选的核心基准指标')
parser.add_argument('--stability_threshold', type=float, default=0.001, help='Best与Last指标差距在该阈值内时优先选Last')

if __name__ == '__main__':
    args = parser.parse_args()

    # 路径初始化配置
    if args.exp_name:
        test_csv_path = os.path.join(args.dataset_root, args.exp_name, 'test.csv')
        ensemble_output_dir = os.path.join(args.output_root, args.exp_name)
    else:
        # test_csv_path = 'datasets/Contrast/test_195.csv'
        test_csv_path = 'datasets/Contrast/External/test_path.csv'
        ensemble_output_dir = args.output_root
    os.makedirs(ensemble_output_dir, exist_ok=True)

    # 1. 跨模型及多折最优配置检索
    print(f"\n正在基于基准指标 [{args.target_metric}] 自动筛选各折最优模型配置...")
    best_configs = select_best_of_the_best(
        root_dir=args.root_dir,
        target_metric=args.target_metric,
        threshold=args.stability_threshold
    )

    print("\n" + "=" * 80)
    print(" 最终生成的最佳模型配置明细 ")
    print("=" * 80)
    pprint.pprint(best_configs, sort_dicts=False, width=120)

    # 2. 多折串行推理
    print("\n" + "=" * 80)
    print(" 开始多折串行推理 ")
    print("=" * 80)
    infer_csv_paths = []

    for config in best_configs:
        fold, exp = config['fold'], config['exp']
        current_log_dir = os.path.join(ensemble_output_dir, fold)
        weight_path = os.path.join(args.root_dir, exp, config['weight_file'])

        print(f"\n==> 正在执行推理: {fold} (所选最优模型: {exp}) ...")
        args_infer = argparse.Namespace(
            yaml_path=f'configs/Contrast/MIL/{exp}.yaml',
            test_dataset_csv=test_csv_path,
            model_weight_path=weight_path,
            test_log_dir=current_log_dir
        )

        # 调用外部真实的推理函数
        test(args_infer)

        # 读取当前折单模推理结果并计算当前折指标
        infer_res_path = os.path.join(current_log_dir, 'Infer_Result.csv')
        if os.path.exists(infer_res_path):
            df_fold = pd.read_csv(infer_res_path)
            fold_probs = np.array([parse_probs(p) for p in df_fold['probs']])
            fold_metrics = calculate_metrics(
                df_fold['label'].values,
                df_fold['prediction'].values,
                fold_probs,
                prefix=f'{args.target_metric}_'
            )
            pd.DataFrame([fold_metrics]).to_csv(os.path.join(current_log_dir, 'Best_Result.csv'), index=False)
            infer_csv_paths.append(infer_res_path)
        else:
            print(f"[-] 警告: 未找到 {fold} 的推理输出结果文件 {infer_res_path}")

    # 3. 多折结果按验证集分数加权集成 (Ensemble)
    print("\n" + "=" * 80)
    print(" 开始多折预测结果加权聚合与集成 ")
    print("=" * 80)

    if not infer_csv_paths:
        print("[-] 错误: 没有成功的推理输出文件，无法执行集成。")
        exit(1)

    # 根据验证集分数进行权重归一化 (Softmax 变体或直接线性比例归一化)
    scores = np.array([config['score'] for config in best_configs])
    weights = scores / np.sum(scores)

    print("-" * 55)
    print(f"{'Fold':<10} | {'Weight (归一化权重)':<20} | {'Validation Score'}")
    print("-" * 55)
    for i, config in enumerate(best_configs):
        print(f"{config['fold']:<10} | {weights[i]:.4f}               | {scores[i]:.4f}")
    print("-" * 55)

    # 读取并聚合所有折的概率
    dfs = [pd.read_csv(p) for p in infer_csv_paths]
    all_probs_list = [np.array([parse_probs(p) for p in df['probs']]) for df in dfs]

    # 矩阵广播加权求和
    weighted_probs = np.zeros_like(all_probs_list[0])
    for i in range(len(all_probs_list)):
        weighted_probs += all_probs_list[i] * weights[i]

    # 获取最终硬标签
    y_true = dfs[0]['label'].values
    y_pred = np.argmax(weighted_probs, axis=1)

    # 构造并保存最终的大表
    final_df = dfs[0][['slide_id', 'label']].copy()
    final_df['probs'] = [str(list(p)) for p in weighted_probs]
    final_df['prediction'] = y_pred

    save_path = os.path.join(ensemble_output_dir, 'Ensemble_Weighted_Result.csv')
    final_df.to_csv(save_path, index=False)

    # 计算最终集成后的全量指标
    final_metrics = calculate_metrics(y_true, y_pred, weighted_probs)
    final_metrics_json = {k: {"mean": float(v), "std": 0.0} for k, v in final_metrics.items()}

    json_save_path = os.path.join(ensemble_output_dir, 'merge_5_fold_metrics.json')
    with open(json_save_path, 'w') as f:
        json.dump(final_metrics_json, f, indent=4)

    print(f"\n[成功] 自动化筛选与多折加权集成 Pipeline 运行完毕！")
    print(f"➔ 预测汇总表格已保存至: {save_path}")
    print(f"➔ 最终多维集成指标已保存至: {json_save_path}\n")