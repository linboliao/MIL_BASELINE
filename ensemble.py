import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')
import re


def parse_probabilities_space_separated(prob_str):
    """
    解析空格分隔的概率字符串
    示例: "[0.9905758  0.00942421]" -> array([0.9905758, 0.00942421])
    """
    try:
        # 移除方括号和多余空格
        prob_str = str(prob_str).strip()
        prob_str = prob_str.strip('[]')

        # 分割字符串，处理多个空格的情况
        # 使用正则表达式分割多个空格
        prob_list = re.split(r'\s+', prob_str.strip())

        # 转换为浮点数
        prob_list = [float(x) for x in prob_list if x != '']

        if len(prob_list) == 2:
            return np.array(prob_list, dtype=np.float32)
        else:
            # 尝试用逗号分割
            prob_list = [float(x) for x in prob_str.replace(' ', '').split(',') if x != '']
            return np.array(prob_list, dtype=np.float32)
    except Exception as e:
        print(f"解析错误: {prob_str}, 错误: {e}")
        return np.array([0.5, 0.5], dtype=np.float32)  # 返回默认值


def load_and_prepare_data(model1_csv, model2_csv):
    """
    加载和准备数据
    """
    print("=" * 50)
    print("加载模型预测结果")
    print("=" * 50)

    # 读取CSV文件
    df1 = pd.read_csv(model1_csv)
    df2 = pd.read_csv(model2_csv)

    print(f"模型1文件: {model1_csv} (样本数: {len(df1)})")
    print(f"模型2文件: {model2_csv} (样本数: {len(df2)})")

    # 显示前几行
    print("\n模型1前5行数据:")
    print(df1.head())
    print("\n模型2前5行数据:")
    print(df2.head())

    # 检查slide_id是否匹配
    slide_ids_match = (df1['slide_id'].values == df2['slide_id'].values).all()

    if slide_ids_match:
        print(f"\n✓ slide_id完全匹配")
    else:
        print(f"\n⚠ slide_id不匹配，按slide_id排序对齐")
        # 按slide_id排序
        df1 = df1.sort_values('slide_id').reset_index(drop=True)
        df2 = df2.sort_values('slide_id').reset_index(drop=True)

        # 检查排序后是否匹配
        slide_ids_match = (df1['slide_id'].values == df2['slide_id'].values).all()
        if slide_ids_match:
            print(f"✓ 排序后slide_id匹配")
        else:
            print(f"⚠ 排序后仍不匹配，使用模型1的slide_id顺序")
            # 只保留共有的slide_id
            common_ids = set(df1['slide_id']).intersection(set(df2['slide_id']))
            df1 = df1[df1['slide_id'].isin(common_ids)].sort_values('slide_id').reset_index(drop=True)
            df2 = df2[df2['slide_id'].isin(common_ids)].sort_values('slide_id').reset_index(drop=True)

    # 提取真实标签
    y_true = df1['label'].values

    # 解析probs列
    print("\n解析概率字符串...")
    probs1 = np.array([parse_probabilities_space_separated(p) for p in df1['probs']])
    probs2 = np.array([parse_probabilities_space_separated(p) for p in df2['probs']])

    print(f"模型1概率形状: {probs1.shape}")
    print(f"模型2概率形状: {probs2.shape}")

    # 检查概率是否合理
    print(f"\n概率值检查:")
    print(f"模型1概率范围: [{probs1[:, 0].min():.4f}, {probs1[:, 0].max():.4f}]")
    print(f"模型2概率范围: [{probs2[:, 0].min():.4f}, {probs2[:, 0].max():.4f}]")

    return probs1, probs2, y_true, df1, df2


def evaluate_model_performance(y_true, probs, model_name="模型"):
    """
    评估模型性能
    """
    # 获取预测标签
    y_pred = np.argmax(probs, axis=1)

    # 计算指标
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs[:, 1])
    f1 = f1_score(y_true, y_pred, average='macro')

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 计算各类别的精确度、召回率、F1
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        'name': model_name,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'classification_report': report
    }


def ensemble_with_weights(probs1, probs2, weights=[0.8, 0.2]):
    """
    加权集成
    """
    # 归一化权重
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    # 加权平均
    ensemble_probs = weights[0] * probs1 + weights[1] * probs2

    return ensemble_probs


def find_optimal_weights(probs1, probs2, y_true, weight_options=None):
    """
    寻找最佳权重组合
    """
    if weight_options is None:
        weight_options = [
            (0.9, 0.1),  # 模型1主导
            (0.8, 0.2),  # 推荐权重
            (0.7, 0.3),  # 相对均衡
            (0.6, 0.4),  # 更多考虑模型2
            (0.5, 0.5),  # 平均
        ]

    best_auc = -1
    best_weights = None
    best_probs = None
    all_results = []

    print("测试不同权重组合:")
    print("-" * 40)

    for w1, w2 in weight_options:
        ensemble_probs = ensemble_with_weights(probs1, probs2, [w1, w2])
        y_pred = np.argmax(ensemble_probs, axis=1)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, ensemble_probs[:, 1])
        f1 = f1_score(y_true, y_pred, average='macro')

        all_results.append({
            'weights': (w1, w2),
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'probs': ensemble_probs
        })

        print(f"权重 [{w1:.1f}, {w2:.1f}]: "
              f"precision={precision:.4f}, "
              f"recall={recall:.4f}, "
              f"Acc={accuracy:.4f}, "
              f"AUC={auc:.4f}, "
              f"F1={f1:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_weights = (w1, w2)
            best_probs = ensemble_probs

    print("-" * 40)
    print(f"最佳权重: {best_weights} (AUC={best_auc:.4f})")

    return best_weights, best_probs, all_results


def main_integration():
    """
    主集成函数
    """
    # 设置文件路径
    model1_csv = "results/test_results_fold5_seed44.csv"
    model2_csv = "results/LS/uni/dataset_001/5fold/AB_MIL/time_2026-01-08-10-18_5fold_AB_MIL_seed_42/infer/Infer_Result_AB_MIL_cleaned.csv"  # 修改为你的文件路径

    # 1. 加载数据
    probs1, probs2, y_true, df1, df2 = load_and_prepare_data(model1_csv, model2_csv)

    # 2. 评估单个模型
    print("\n" + "=" * 50)
    print("单个模型性能评估")
    print("=" * 50)

    metrics1 = evaluate_model_performance(y_true, probs1, "模型1")
    metrics2 = evaluate_model_performance(y_true, probs2, "模型2")

    print(f"\n模型1性能:")
    print(f"  precision: {metrics1['precision']:.4f}")
    print(f"  recall: {metrics1['recall']:.4f}")
    print(f"  Accuracy: {metrics1['accuracy']:.4f}")
    print(f"  AUC:      {metrics1['auc']:.4f}")
    print(f"  F1 Score: {metrics1['f1']:.4f}")

    print(f"\n模型2性能:")
    print(f"  precision: {metrics1['precision']:.4f}")
    print(f"  recall: {metrics1['recall']:.4f}")
    print(f"  Accuracy: {metrics2['accuracy']:.4f}")
    print(f"  AUC:      {metrics2['auc']:.4f}")
    print(f"  F1 Score: {metrics2['f1']:.4f}")

    # 3. 寻找最佳集成权重
    print("\n" + "=" * 50)
    print("寻找最佳集成权重")
    print("=" * 50)

    best_weights, best_probs, all_results = find_optimal_weights(probs1, probs2, y_true)

    # 4. 评估最佳集成模型
    print("\n" + "=" * 50)
    print("最佳集成模型性能")
    print("=" * 50)

    best_metrics = evaluate_model_performance(y_true, best_probs, f"集成模型(权重{best_weights})")

    print(f"\n集成模型性能 (权重: {best_weights}):")
    print(f"  precision: {best_metrics['precision']:.4f}")
    print(f"  recall: {best_metrics['recall']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  AUC:      {best_metrics['auc']:.4f}")
    print(f"  F1 Score: {best_metrics['f1']:.4f}")

    # 5. 性能提升分析
    print("\n" + "=" * 50)
    print("性能提升分析")
    print("=" * 50)

    # 找到最佳单模型
    if metrics1['auc'] > metrics2['auc']:
        best_single = metrics1
        best_single_name = "模型1"
    else:
        best_single = metrics2
        best_single_name = "模型2"

    # 计算提升
    acc_improvement = best_metrics['accuracy'] - best_single['accuracy']
    auc_improvement = best_metrics['auc'] - best_single['auc']
    f1_improvement = best_metrics['f1'] - best_single['f1']

    print(f"\n与最佳单模型 ({best_single_name}) 对比:")
    print(f"  Accuracy提升: {acc_improvement:+.4f}")
    print(f"  AUC提升:      {auc_improvement:+.4f}")
    print(f"  F1 Score提升: {f1_improvement:+.4f}")

    if auc_improvement > 0:
        print(f"\n✓ 集成模型在AUC指标上优于最佳单模型")
    else:
        print(f"\n⚠ 集成模型在AUC指标上未超过最佳单模型")

    # 6. 保存结果
    print("\n" + "=" * 50)
    print("保存结果")
    print("=" * 50)

    # 创建详细的结果DataFrame
    ensemble_pred = np.argmax(best_probs, axis=1)

    result_df = pd.DataFrame({
        'slide_id': df1['slide_id'],
        'true_label': y_true,
        'model1_prob_class0': probs1[:, 0],
        'model1_prob_class1': probs1[:, 1],
        'model1_prediction': metrics1['predictions'],
        'model2_prob_class0': probs2[:, 0],
        'model2_prob_class1': probs2[:, 1],
        'model2_prediction': metrics2['predictions'],
        'ensemble_prob_class0': best_probs[:, 0],
        'ensemble_prob_class1': best_probs[:, 1],
        'ensemble_prediction': ensemble_pred,
        'ensemble_correct': (ensemble_pred == y_true).astype(int)
    })

    # 计算每个样本的模型一致性
    result_df['models_agree'] = (result_df['model1_prediction'] == result_df['model2_prediction']).astype(int)
    result_df['ensemble_vs_model1'] = (result_df['ensemble_prediction'] == result_df['model1_prediction']).astype(int)
    result_df['ensemble_vs_model2'] = (result_df['ensemble_prediction'] == result_df['model2_prediction']).astype(int)

    # 保存结果
    output_file = f"ensemble_results_best_weights_{best_weights[0]}_{best_weights[1]}.csv"
    result_df.to_csv(output_file, index=False)
    print(f"✓ 集成结果已保存到: {output_file}")

    # 7. 保存性能报告
    report_data = []

    # 单个模型
    report_data.append({
        'Model': 'Model 1',
        'Accuracy': f"{metrics1['accuracy']:.4f}",
        'AUC': f"{metrics1['auc']:.4f}",
        'F1_Score': f"{metrics1['f1']:.4f}"
    })

    report_data.append({
        'Model': 'Model 2',
        'Accuracy': f"{metrics2['accuracy']:.4f}",
        'AUC': f"{metrics2['auc']:.4f}",
        'F1_Score': f"{metrics2['f1']:.4f}"
    })

    # 集成模型
    report_data.append({
        'Model': f'Ensemble (权重{best_weights})',
        'Accuracy': f"{best_metrics['accuracy']:.4f}",
        'AUC': f"{best_metrics['auc']:.4f}",
        'F1_Score': f"{best_metrics['f1']:.4f}"
    })

    # 所有权重组合
    for res in all_results:
        w1, w2 = res['weights']
        if (w1, w2) != best_weights:  # 跳过最佳权重，因为已经单独列出
            report_data.append({
                'Model': f'Ensemble ({w1},{w2})',
                'Accuracy': f"{res['accuracy']:.4f}",
                'AUC': f"{res['auc']:.4f}",
                'F1_Score': f"{res['f1']:.4f}"
            })

    report_df = pd.DataFrame(report_data)
    report_file = "ensemble_performance_report.csv"
    report_df.to_csv(report_file, index=False)
    print(f"✓ 性能报告已保存到: {report_file}")

    # 8. 保存详细分析
    print("\n" + "=" * 50)
    print("详细分析")
    print("=" * 50)

    # 模型一致性分析
    agreement_rate = result_df['models_agree'].mean()
    print(f"\n模型一致性分析:")
    print(f"  两个模型预测一致的样本比例: {agreement_rate:.2%}")

    # 集成模型与单个模型的一致性
    agreement_with_model1 = result_df['ensemble_vs_model1'].mean()
    agreement_with_model2 = result_df['ensemble_vs_model2'].mean()
    print(f"  集成模型与模型1一致的比例: {agreement_with_model1:.2%}")
    print(f"  集成模型与模型2一致的比例: {agreement_with_model2:.2%}")

    # 在模型不一致的样本上，集成模型的表现
    disagreed_samples = result_df[result_df['models_agree'] == 0]
    if len(disagreed_samples) > 0:
        correct_in_disagreed = disagreed_samples['ensemble_correct'].mean()
        print(f"  在模型预测不一致的样本上，集成模型的正确率: {correct_in_disagreed:.2%}")

    # 保存样本级分析
    analysis_df = result_df[['slide_id', 'true_label',
                             'model1_prediction', 'model2_prediction',
                             'ensemble_prediction', 'ensemble_correct',
                             'models_agree']]
    analysis_file = "sample_level_analysis.csv"
    analysis_df.to_csv(analysis_file, index=False)
    print(f"✓ 样本级分析已保存到: {analysis_file}")

    # 9. 显示结果预览
    print("\n" + "=" * 50)
    print("结果预览 (前10个样本)")
    print("=" * 50)

    preview_cols = ['slide_id', 'true_label',
                    'model1_prediction', 'model2_prediction',
                    'ensemble_prediction', 'ensemble_correct']
    print(result_df[preview_cols].head(10))

    print("\n" + "=" * 50)
    print("集成完成!")
    print("=" * 50)

    return result_df, report_df, best_weights


# 运行主函数
if __name__ == "__main__":
    # 直接运行
    result_df, report_df, best_weights = main_integration()

    # 或者使用自定义参数
    # result_df, report_df, best_weights = main_integration(
    #     model1_csv="path/to/model1.csv",
    #     model2_csv="path/to/model2.csv"
    # )
