import argparse
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.model_utils import get_model_from_yaml
from utils.wsi_utils import WSI_Dataset
from utils.yaml_utils import read_yaml

def convert_to_serializable(obj):
    """将NumPy和PyTorch数据类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def diagnose_mil_model(args, model, val_dataloader, device, num_classes=7,
                       class_names=None, bag_weight=0.7, retrun_WSI_feature=True):
    """
    根据您的MIL模型推理过程进行诊断
    """
    model.eval()
    model = model.to(device)

    # 存储所有结果
    all_labels = []
    all_bag_predictions = []
    all_bag_logits = []
    all_instance_losses = []
    all_ws_features = []  # WSI特征
    all_ws_attns = []  # 注意力权重

    print("开始诊断MIL模型...")

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc="诊断进度")):
            # 根据您的代码，data[1]是标签，data[0]是bag
            label = data[1].to(device).long()
            bag = data[0].to(device).float()

            # 保存真实标签
            all_labels.append(label.cpu().numpy())

            # 1. 获取WSI特征
            if retrun_WSI_feature:
                try:
                    WSI_feature = model(bag, label=label, return_WSI_feature=True)['WSI_feature']
                    all_ws_features.append(WSI_feature.cpu().numpy())
                except:
                    # 如果模型不支持返回特征，尝试其他方法
                    pass

            # 2. 获取完整的前向传播结果
            forward_return = model(bag, label=label)

            # 提取各个部分
            if 'instance_loss' in forward_return:
                instance_loss = forward_return['instance_loss']
                all_instance_losses.append(instance_loss.item())

            if 'logits' in forward_return:
                val_logits = forward_return['logits']
                if val_logits.dim() > 1:
                    val_logits = val_logits.squeeze(0)

                # 保存logits
                all_bag_logits.append(val_logits.cpu().numpy())

                # 计算softmax概率
                probs = torch.softmax(val_logits, 0).cpu().numpy()
                all_bag_predictions.append(probs)

    # 合并所有结果
    all_labels = np.concatenate(all_labels) if all_labels else None
    all_bag_predictions = np.array(all_bag_predictions) if all_bag_predictions else None
    all_bag_logits = np.array(all_bag_logits) if all_bag_logits else None

    if all_ws_features:
        all_ws_features = np.concatenate(all_ws_features, axis=0)

    # 计算诊断指标
    print("\n" + "=" * 60)
    print("MIL模型诊断报告")
    print("=" * 60)

    diagnosis_results = {}

    if all_labels is not None and all_bag_predictions is not None:
        # 1. 获取预测类别
        all_preds = np.argmax(all_bag_predictions, axis=1)

        # 2. 基础指标
        accuracy = np.mean(all_preds == all_labels)
        diagnosis_results['accuracy'] = accuracy
        print(f"准确率 (Accuracy): {accuracy:.4f}")

        # 3. 平均置信度
        max_probs = np.max(all_bag_predictions, axis=1)
        avg_confidence = np.mean(max_probs)
        diagnosis_results['average_confidence'] = avg_confidence
        print(f"平均最大概率 (Average Confidence): {avg_confidence:.4f}")

        # 4. 置信度分布可视化
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
        plt.title('置信度分布')
        plt.xlabel('最大概率')
        plt.ylabel('频数')
        plt.grid(True, alpha=0.3)

        # 5. 计算AUC（每个类别）
        n_classes = all_bag_predictions.shape[1]
        auc_scores = []

        for class_idx in range(n_classes):
            y_true = (all_labels == class_idx).astype(int)
            y_score = all_bag_predictions[:, class_idx]

            if len(np.unique(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, y_score)
                    auc_scores.append(auc)
                    class_name = class_names[class_idx] if class_names else f'Class_{class_idx}'
                    print(f"类别 {class_name} AUC: {auc:.4f}")
                except Exception as e:
                    print(f"计算类别 {class_idx} AUC时出错: {e}")

        if auc_scores:
            avg_auc = np.mean(auc_scores)
            diagnosis_results['auc_scores'] = auc_scores
            diagnosis_results['average_auc'] = avg_auc
            print(f"平均AUC: {avg_auc:.4f}")

        # 6. 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        diagnosis_results['confusion_matrix'] = cm.tolist()

        plt.subplot(1, 3, 2)
        if class_names:
            tick_labels = class_names
        else:
            tick_labels = [f'Class_{i}' for i in range(n_classes)]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=tick_labels,
                    yticklabels=tick_labels)
        plt.title('confusion matrix')
        plt.xlabel('Pred')
        plt.ylabel('GT')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # 7. 分类报告
        print("\n分类报告:")
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True
        )

        # 打印报告
        print(classification_report(
            all_labels,
            all_preds,
            target_names=class_names
        ))

        diagnosis_results['classification_report'] = report

        # 8. 各个类别性能
        print("\n各个类别性能:")
        per_class_accuracy = {}
        for class_idx in range(n_classes):
            mask = all_labels == class_idx
            if np.sum(mask) > 0:
                class_acc = np.mean(all_preds[mask] == class_idx)
                class_conf = np.mean(max_probs[mask])
                class_name = class_names[class_idx] if class_names else f'Class_{class_idx}'

                per_class_accuracy[class_name] = {
                    'accuracy': float(class_acc),
                    'average_confidence': float(class_conf),
                    'samples': int(np.sum(mask))
                }

                print(f"类别 {class_name}: "
                      f"准确率={class_acc:.3f}, 平均置信度={class_conf:.3f}, "
                      f"样本数={np.sum(mask)}")

        diagnosis_results['per_class_accuracy'] = per_class_accuracy

        # 9. 错误样本分析
        error_mask = all_preds != all_labels
        if np.sum(error_mask) > 0:
            error_confidence = max_probs[error_mask]
            correct_confidence = max_probs[~error_mask]

            plt.subplot(1, 3, 3)
            plt.hist(correct_confidence, bins=30, alpha=0.5, label='正确', edgecolor='black')
            plt.hist(error_confidence, bins=30, alpha=0.5, label='错误', edgecolor='black')
            plt.title('True/false contrast')
            plt.xlabel('confidence')
            plt.ylabel('frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)

            diagnosis_results['error_analysis'] = {
                'error_rate': float(np.mean(error_mask)),
                'avg_confidence_correct': float(np.mean(correct_confidence)),
                'avg_confidence_error': float(np.mean(error_confidence)),
                'num_errors': int(np.sum(error_mask))
            }

            print(f"\n错误样本分析:")
            print(f"错误率: {np.mean(error_mask):.4f}")
            print(f"正确样本平均置信度: {np.mean(correct_confidence):.4f}")
            print(f"错误样本平均置信度: {np.mean(error_confidence):.4f}")

        plt.tight_layout()
        plt.savefig(os.path.join(args.test_log_dir, 'diagnosis_plots.png'), dpi=150, bbox_inches='tight')
        plt.show()

        # 10. 特征空间可视化（如果有特征）
        if all_ws_features is not None and len(all_ws_features) > 0:
            print("\n进行特征空间可视化...")

            try:
                # 使用PCA降维
                pca = PCA(n_components=2)
                features_pca = pca.fit_transform(all_ws_features)

                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1],
                                      c=all_labels, cmap='tab10', alpha=0.6, s=30)
                plt.colorbar(scatter, label='类别')
                plt.title('PCA: WSI feature space vis')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')

                # 使用t-SNE降维
                tsne = TSNE(n_components=2, perplexity=min(30, len(all_ws_features) - 1),
                            random_state=42, n_iter=1000)
                features_tsne = tsne.fit_transform(all_ws_features)

                plt.subplot(1, 2, 2)
                scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                                      c=all_labels, cmap='tab10', alpha=0.6, s=30)
                plt.colorbar(scatter, label='类别')
                plt.title('t-SNE: WSI feature space vis')
                plt.xlabel('t-SNE1')
                plt.ylabel('t-SNE2')

                plt.tight_layout()
                plt.savefig(os.path.join(args.test_log_dir, 'feature_space.png'), dpi=150, bbox_inches='tight')
                plt.show()

                diagnosis_results['feature_analysis'] = {
                    'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'feature_dimension': int(all_ws_features.shape[1]),
                    'num_samples': int(all_ws_features.shape[0])
                }

            except Exception as e:
                print(f"特征可视化失败: {e}")

        # 11. 概率热图可视化
        plt.figure(figsize=(15, 6))

        # 按真实类别排序
        sorted_indices = np.argsort(all_labels)
        probs_sorted = all_bag_predictions[sorted_indices]

        plt.subplot(1, 2, 1)
        im = plt.imshow(probs_sorted.T, aspect='auto', cmap='viridis',
                        interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(im, label='概率')
        plt.title('pred probs hot map')
        plt.xlabel('idx')
        plt.ylabel('pred cls')

        if class_names:
            plt.yticks(range(len(class_names)), class_names)

        # 添加真实类别的分隔线
        unique_labels = np.unique(all_labels[sorted_indices])
        for label in unique_labels:
            idx = np.where(all_labels[sorted_indices] == label)[0]
            if len(idx) > 0:
                plt.axvline(x=idx[0], color='red', linestyle='--', alpha=0.5, linewidth=0.5)
                plt.axvline(x=idx[-1], color='red', linestyle='--', alpha=0.5, linewidth=0.5)

        # 类间混淆分析
        plt.subplot(1, 2, 2)
        if cm is not None:
            # 归一化混淆矩阵
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=tick_labels, yticklabels=tick_labels)
            plt.title('normal confusion matrix')
            plt.xlabel('pred')
            plt.ylabel('gt')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(args.test_log_dir, 'probability_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.show()

        # 12. 保存详细结果
        diagnosis_results.update({
            'num_samples': int(len(all_labels)),
            'predictions': all_preds.tolist(),
            'probabilities': all_bag_predictions.tolist(),
            'labels': all_labels.tolist(),
            'max_probabilities': max_probs.tolist()
        })

        if all_bag_logits is not None:
            diagnosis_results['logits'] = all_bag_logits.tolist()

        if all_instance_losses:
            diagnosis_results['instance_losses'] = all_instance_losses

        diagnosis_results_serializable = convert_to_serializable(diagnosis_results)

        # 保存到JSON文件
        results_path = os.path.join(args.test_log_dir, 'diagnosis_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(diagnosis_results_serializable, f, indent=2, ensure_ascii=False)
        print(f"\n诊断结果已保存到: {results_path}")

        # 13. 生成诊断建议
        generate_diagnosis_advice(diagnosis_results, class_names)

        return diagnosis_results

    else:
        print("无法计算指标：缺少预测或标签数据")
        return None


def generate_diagnosis_advice(diagnosis_results, class_names=None):
    """根据诊断结果生成改进建议"""
    print("\n" + "=" * 60)
    print("诊断建议")
    print("=" * 60)

    accuracy = diagnosis_results.get('accuracy', 0)
    avg_confidence = diagnosis_results.get('average_confidence', 0)
    per_class_acc = diagnosis_results.get('per_class_accuracy', {})

    # 1. 总体准确率分析
    if accuracy < 0.4:
        print("🚨 严重问题: 模型准确率极低 (<0.4)")
        print("  → 可能原因:")
        print("    1. 模型架构不适合当前任务")
        print("    2. 特征提取能力不足")
        print("    3. 超参数设置不合理")
        print("  → 建议:")
        print("    1. 检查数据预处理是否正确")
        print("    2. 重新设计模型架构")
        print("    3. 尝试更简单的baseline模型")

    elif accuracy < 0.6:
        print("⚠️ 问题: 模型准确率较低 (<0.6)")
        if avg_confidence < 0.3:
            print("  → 具体问题: 模型输出概率很平坦，缺乏置信度")
            print("  → 建议:")
            print("    1. 增加模型复杂度")
            print("    2. 使用更强的正则化")
            print("    3. 尝试不同的注意力机制")
        elif avg_confidence > 0.8:
            print("  → 具体问题: 模型过度自信但预测错误")
            print("  → 建议:")
            print("    1. 使用标签平滑")
            print("    2. 增加Dropout率")
            print("    3. 使用温度缩放校准")
        else:
            print("  → 建议:")
            print("    1. 调整损失函数权重")
            print("    2. 增加训练轮数")
            print("    3. 使用数据增强")

    elif accuracy < 0.7:
        print("📈 有改进空间: 模型准确率中等 (0.6-0.7)")
        print("  → 建议:")
        print("    1. 精细调整超参数")
        print("    2. 使用集成方法")
        print("    3. 后处理校准概率")

    else:
        print("✅ 良好: 模型准确率较高 (>0.7)")
        print("  → 建议:")
        print("    1. 可以尝试更复杂的模型")
        print("    2. 使用模型集成进一步提升")

    # 2. 检查类别不平衡
    if per_class_acc:
        samples_per_class = [info['samples'] for info in per_class_acc.values()]
        if len(samples_per_class) > 1:
            max_samples = max(samples_per_class)
            min_samples = min(samples_per_class)

            if max_samples / min_samples > 5:  # 严重不平衡
                print(f"\n⚠️ 检测到严重类别不平衡: 最多类/最少类 = {max_samples / min_samples:.1f}")
                print("  → 建议:")
                print("    1. 使用类别加权损失函数")
                print("    2. 对少数类进行过采样")
                print("    3. 对多数类进行欠采样")
                print("    4. 使用Focal Loss")

        # 3. 检查表现最差的类别
        worst_class = min(per_class_acc.items(), key=lambda x: x[1]['accuracy'])
        best_class = max(per_class_acc.items(), key=lambda x: x[1]['accuracy'])

        if worst_class[1]['accuracy'] < 0.3:
            print(f"\n⚠️ 表现最差类别: {worst_class[0]}, 准确率={worst_class[1]['accuracy']:.3f}")
            print("  → 建议:")
            print("    1. 为该类别增加更多训练样本")
            print("    2. 为该类别设置更高的损失权重")
            print("    3. 检查该类别样本质量")

    # 4. 检查混淆矩阵
    cm = diagnosis_results.get('confusion_matrix')
    if cm is not None:
        cm = np.array(cm)
        n_classes = cm.shape[0]

        # 找出最容易混淆的类别对
        confusions = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confusions.append((i, j, cm[i, j]))

        if confusions:
            # 按混淆数量排序
            confusions.sort(key=lambda x: x[2], reverse=True)
            print(f"\n最易混淆的类别对:")
            for i, j, count in confusions[:3]:  # 显示前3个
                class_i = class_names[i] if class_names else f'Class_{i}'
                class_j = class_names[j] if class_names else f'Class_{j}'
                print(f"  → {class_i} → {class_j}: {count} 次")

                # 如果是相邻类别，可能是正常现象
                if abs(i - j) == 1:
                    print(f"    (注: 这是相邻类别，混淆可能是正常的)")
                else:
                    print(f"    (注: 这不是相邻类别，可能需要重点关注)")


def analyze_class_distribution(val_dataset_csv):
    """分析数据集中的类别分布"""
    df = pd.read_csv(val_dataset_csv)

    if 'label' in df.columns:
        class_dist = df['label'].value_counts().sort_index()
        print("类别分布:")
        for class_idx, count in class_dist.items():
            print(f"类别 {class_idx}: {count} 个样本 ({count / len(df) * 100:.1f}%)")

        return class_dist
    else:
        print("CSV文件中没有'label'列")
        return None


parser = argparse.ArgumentParser()
parser.add_argument('--yaml_path', type=str, default='configs/gleason/h-optimus-1/CLAM_MB_MIL.yaml', help='path to MIL-model-yaml file')
parser.add_argument('--val_dataset_csv', type=str, default='datasets/gleason/dataset_001/5fold/Total_5-fold_5fold_1fold.csv', help='path to dataset csv')
parser.add_argument('--model_weight_path', type=str, default='results/gleason/dataset_001/5fold/CLAM_MB_MIL/time_2026-01-16-17-35_5fold_CLAM_MB_MIL_seed_2024/fold_1/Best_EPOCH_4.pth', help='path to model weights ')
parser.add_argument('--test_log_dir', type=str, default='results/test-log', help='path to test log dir')
args = parser.parse_args()
if __name__ == '__main__':
    os.makedirs(args.test_log_dir, exist_ok=True)

    print(f"读取配置文件: {args.yaml_path}")
    yaml_args = read_yaml(args.yaml_path)

    device = torch.device(f'cuda:{yaml_args.General.device}')

    mil_model = get_model_from_yaml(yaml_args)
    mil_model = mil_model.to(device)
    mil_model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))

    # 创建验证集
    print(f"加载验证数据集: {args.val_dataset_csv}")

    # 分析类别分布
    class_dist = analyze_class_distribution(args.val_dataset_csv)

    # 创建数据集和数据加载器
    val_ds = WSI_Dataset(args.val_dataset_csv, 'val')
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,  # MIL通常是batch_size=1
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"验证集大小: {len(val_ds)}")

    class_names = ['3+3', '3+4', '4+3', '4+4', '4+5', '5+4', '5+5']

    # 运行诊断
    print("\n开始诊断模型...")
    diagnosis_results = diagnose_mil_model(
        args,
        mil_model,
        val_dataloader,
        device,
        class_names=class_names
    )

    # 根据诊断结果给出建议
    if diagnosis_results:
        accuracy = diagnosis_results['accuracy']
        avg_confidence = diagnosis_results['average_confidence']

        print("\n" + "=" * 60)
        print("诊断建议")
        print("=" * 60)

        if accuracy < 0.6:
            print("⚠️ 模型准确率较低 (<0.6)")
            if avg_confidence and avg_confidence < 0.3:
                print("  → 问题: 模型输出概率很平坦，缺乏置信度")
                print("  → 建议: 需要改进模型架构或损失函数")
            elif avg_confidence and avg_confidence > 0.7:
                print("  → 问题: 模型过度自信但预测错误")
                print("  → 建议: 需要概率校准和正则化")
            else:
                print("  → 建议: 检查特征质量，增加数据增强")

        # 检查类别不平衡
        if class_dist is not None:
            min_count = class_dist.min()
            max_count = class_dist.max()
            if max_count / min_count > 5:  # 严重不平衡
                print(f"⚠️ 检测到严重类别不平衡: 最多类/最少类 = {max_count / min_count:.1f}")
                print("  → 建议: 使用类别加权损失或过采样")

        # 检查混淆矩阵
        if diagnosis_results['confusion_matrix']:
            cm = np.array(diagnosis_results['confusion_matrix'])
            n_classes = cm.shape[0]

            # 找出最容易混淆的类别对
            confusions = []
            for i in range(n_classes):
                for j in range(n_classes):
                    if i != j and cm[i, j] > 0:
                        confusions.append((i, j, cm[i, j]))

            if confusions:
                # 按混淆数量排序
                confusions.sort(key=lambda x: x[2], reverse=True)
                print(f"\n最易混淆的类别对:")
                for i, j, count in confusions[:3]:  # 显示前3个
                    print(f"  → {class_names[i]} 被误判为 {class_names[j]}: {count} 次")
