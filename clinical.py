import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')


# 1. 加载数据
def load_data(a_path, b_path, c_path):
    """加载三个CSV文件"""
    df_a = pd.read_csv(a_path)
    df_b = pd.read_csv(b_path)
    df_c = pd.read_csv(c_path)

    print("数据加载完成:")
    print(f"  a.csv: {len(df_a)} 行, 列: {list(df_a.columns)}")
    print(f"  b.csv: {len(df_b)} 行, 列: {list(df_b.columns)}")
    print(f"  c.csv: {len(df_c)} 行, 列: {list(df_c.columns)}")

    return df_a, df_b, df_c


# 2. 准备训练和测试数据
def prepare_data(df_a, df_b, df_c):
    """准备训练和测试集"""
    # 合并a和b作为训练集
    train_df = pd.concat([df_a, df_b], ignore_index=True)

    # 移除slide_id列（假设不需要）
    if 'slide_id' in train_df.columns:
        slide_id_train = train_df['slide_id'].copy()
        train_df = train_df.drop('slide_id', axis=1)
    else:
        slide_id_train = None

    if 'slide_id' in df_c.columns:
        slide_id_test = df_c['slide_id'].copy()
        test_df = df_c.drop('slide_id', axis=1)
    else:
        slide_id_test = None
        test_df = df_c.copy()

    # 分离特征和标签
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']

    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    print(f"\n训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"测试集: {X_test.shape[0]} 样本")

    return X_train, y_train, X_test, y_test, slide_id_train, slide_id_test


# 新增：数据探索和分析函数
def explore_data(X_train, X_test, y_train, y_test, feature_names):
    """探索数据特征"""
    print("\n" + "=" * 60)
    print("数据探索分析")
    print("=" * 60)

    # 1. 基本统计
    print("1. 数据基本信息:")
    print(f"   训练集形状: {X_train.shape}")
    print(f"   测试集形状: {X_test.shape}")
    print(f"   特征数量: {len(feature_names)}")
    print(f"   类别数量: {len(np.unique(y_train))}")

    # 2. 类别分布
    print("\n2. 类别分布:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    for cls, count_train, count_test in zip(unique_train, counts_train, counts_test):
        prop_train = count_train / len(y_train) * 100
        prop_test = count_test / len(y_test) * 100
        print(f"   类别 {cls}: 训练集 {count_train} ({prop_train:.1f}%), "
              f"测试集 {count_test} ({prop_test:.1f}%)")

    # 3. 特征统计
    print("\n3. 特征统计信息:")
    print(f"   特征值范围: [{X_train.min().min():.4f}, {X_train.max().max():.4f}]")
    print(f"   平均值 ± 标准差: {X_train.mean().mean():.4f} ± {X_train.std().mean():.4f}")

    # 4. 检查缺失值
    print(f"\n4. 缺失值检查:")
    print(f"   训练集缺失值: {X_train.isna().sum().sum() if isinstance(X_train, pd.DataFrame) else 0}")
    print(f"   测试集缺失值: {X_test.isna().sum().sum() if isinstance(X_test, pd.DataFrame) else 0}")

    # 5. 特征相关性（前5个特征）
    if len(feature_names) >= 5 and isinstance(X_train, pd.DataFrame):
        print("\n5. 前5个特征的相关系数矩阵:")
        corr_matrix = X_train[feature_names[:5]].corr()
        print(corr_matrix)

    return {
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'n_features': len(feature_names),
        'n_classes': len(np.unique(y_train)),
        'class_distribution_train': dict(zip(unique_train, counts_train)),
        'class_distribution_test': dict(zip(unique_test, counts_test))
    }


# 新增：更详细的MLP训练函数
def train_mlp_detailed(X_train, X_test, y_train, y_test, feature_names, n_trials=3):
    """详细训练MLP，尝试不同配置"""
    print("\n" + "=" * 60)
    print("MLP详细诊断分析")
    print("=" * 60)

    results = {}
    mlp_configs = [
        {
            'name': 'MLP-简单',
            'params': {
                'hidden_layer_sizes': (50,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate_init': 0.001,
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': 42
            }
        },
        {
            'name': 'MLP-深层',
            'params': {
                'hidden_layer_sizes': (100, 50, 25),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate_init': 0.001,
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': 42
            }
        },
        {
            'name': 'MLP-宽层',
            'params': {
                'hidden_layer_sizes': (200, 100),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate_init': 0.001,
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': 42
            }
        },
        {
            'name': 'MLP-小学习率',
            'params': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate_init': 0.0001,  # 更小的学习率
                'max_iter': 2000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 30,
                'random_state': 42
            }
        },
        {
            'name': 'MLP-tanh激活',
            'params': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'tanh',  # 使用tanh激活函数
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate_init': 0.001,
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': 42
            }
        }
    ]

    for config in mlp_configs:
        print(f"\n尝试配置: {config['name']}")
        print(f"参数: {config['params']}")

        try:
            mlp = MLPClassifier(**config['params'])
            mlp.fit(X_train, y_train)

            # 训练集性能
            train_pred = mlp.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)

            # 测试集性能
            test_pred = mlp.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)

            # 记录结果
            results[config['name']] = {
                'model': mlp,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'test_f1': f1_score(y_test, test_pred, average='macro'),
                'n_iter': mlp.n_iter_ if hasattr(mlp, 'n_iter_') else None,
                'final_loss': mlp.loss_ if hasattr(mlp, 'loss_') else None
            }

            print(f"  训练准确率: {train_acc:.4f}")
            print(f"  测试准确率: {test_acc:.4f}")
            print(f"  测试F1: {results[config['name']]['test_f1']:.4f}")
            print(f"  迭代次数: {results[config['name']]['n_iter']}")
            print(f"  最终损失: {results[config['name']]['final_loss']:.4f}")

            # 检查过拟合
            overfit_gap = train_acc - test_acc
            if overfit_gap > 0.1:
                print(f"  ⚠️  可能过拟合: 训练-测试差距 = {overfit_gap:.4f}")
            elif test_acc > train_acc:
                print(f"  ℹ️  测试集表现更好: 差异 = {test_acc - train_acc:.4f}")

        except Exception as e:
            print(f"  训练失败: {e}")

    return results


# 新增：LR和MLP对比分析函数
def compare_lr_mlp(X_train, X_test, y_train, y_test, feature_names):
    """对比LR和MLP的表现"""
    print("\n" + "=" * 60)
    print("LR vs MLP 详细对比分析")
    print("=" * 60)

    # 1. 训练两个模型
    print("\n1. 训练模型...")

    # 逻辑回归
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs',
        multi_class='auto'
    )
    lr.fit(X_train, y_train)

    # MLP（使用最佳配置）
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    # 2. 性能对比
    print("\n2. 性能对比:")

    for name, model in [('逻辑回归', lr), ('MLP', mlp)]:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train, train_pred, average='macro')
        test_f1 = f1_score(y_test, test_pred, average='macro')

        print(f"\n{name}:")
        print(f"  训练准确率: {train_acc:.4f}")
        print(f"  测试准确率: {test_acc:.4f}")
        print(f"  训练F1: {train_f1:.4f}")
        print(f"  测试F1: {test_f1:.4f}")

        if hasattr(model, 'loss_curve_'):
            print(f"  最终损失: {model.loss_:.4f}")
            print(f"  迭代次数: {model.n_iter_}")

    # 3. 决策边界分析
    print("\n3. 决策边界对比:")

    # LR的决策函数
    if hasattr(lr, 'decision_function'):
        lr_scores = lr.decision_function(X_test)
        print(f"  LR决策函数范围: [{lr_scores.min():.3f}, {lr_scores.max():.3f}]")

    # MLP的决策函数
    if hasattr(mlp, 'predict_proba'):
        mlp_probs = mlp.predict_proba(X_test)
        mlp_confidence = mlp_probs.max(axis=1)
        print(f"  MLP预测置信度: 平均={mlp_confidence.mean():.3f}, "
              f"标准差={mlp_confidence.std():.3f}")
        print(f"  MLP预测概率范围: [{mlp_probs.min():.3f}, {mlp_probs.max():.3f}]")

    # 4. 特征重要性对比
    print("\n4. 特征重要性对比:")

    # LR特征重要性（系数的绝对值）
    if hasattr(lr, 'coef_'):
        lr_importance = np.abs(lr.coef_).mean(axis=0)
        lr_top_idx = np.argsort(lr_importance)[-5:][::-1]
        print("  LR最重要的5个特征:")
        for idx in lr_top_idx:
            if idx < len(feature_names):
                print(f"    {feature_names[idx]}: {lr_importance[idx]:.6f}")

    # 5. 检查数据线性可分性
    print("\n5. 数据线性可分性分析:")

    # 计算类别间的距离
    n_classes = len(np.unique(y_train))
    if n_classes == 2:
        class_0 = X_train[y_train == 0]
        class_1 = X_train[y_train == 1]

        mean_0 = class_0.mean(axis=0)
        mean_1 = class_1.mean(axis=0)

        distance = np.linalg.norm(mean_0 - mean_1)
        print(f"  类别间中心距离: {distance:.4f}")

        # 计算类内方差
        var_0 = class_0.var(axis=0).mean()
        var_1 = class_1.var(axis=0).mean()
        print(f"  类别0平均方差: {var_0:.4f}")
        print(f"  类别1平均方差: {var_1:.4f}")

        # 线性可分性指标
        separability = distance / np.sqrt(var_0 + var_1)
        print(f"  线性可分性指标: {separability:.4f}")
        if separability > 3:
            print("  ✅ 数据可能相对线性可分")
        else:
            print("  ⚠️ 数据可能非线性")

    return lr, mlp


def cross_validate_models(X, y, cv=5):
    """交叉验证比较模型"""
    print("\n" + "=" * 60)
    print(f"{cv}折交叉验证结果")
    print("=" * 60)

    models = {
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42,
            early_stopping=True
        ),
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    cv_results = {}
    for name, model in models.items():
        print(f"\n{name}:")
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_results[name] = scores
            print(f"  准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            print(f"  详细分数: {scores}")
        except Exception as e:
            print(f"  交叉验证失败: {e}")

    return cv_results


# 3. 数据预处理
def preprocess_data(X_train, X_test, y_train, y_test, feature_names=None):
    """数据预处理：编码标签、标准化特征"""
    # 保存特征名称
    if feature_names is None:
        feature_names = X_train.columns.tolist()

    # 编码标签（如果是字符串）
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    print(f"\n标签编码: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换回DataFrame以保留特征名称
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    return X_train_scaled_df, X_test_scaled_df, y_train_encoded, y_test_encoded, le, scaler


# 4. 训练和评估模型
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name='rf', feature_names=None):
    """训练模型并评估"""
    # 选择模型
    if model_name == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'lr':
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'svm':
        model = SVC(
            probability=True,
            random_state=42
        )
    elif model_name == 'mlp':
        # MLP模型配置 - 使用更鲁棒的参数
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2正则化
            batch_size=32,  # 指定批大小
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.2,  # 更大的验证集
            n_iter_no_change=20,
            beta_1=0.9, beta_2=0.999,  # Adam参数
            epsilon=1e-8
        )
    else:
        raise ValueError(f"未知模型: {model_name}")

    # 训练模型
    model.fit(X_train, y_train)

    # 对于MLP，打印训练历史
    if model_name == 'mlp':
        if hasattr(model, 'loss_curve_'):
            print(f"  MLP训练历史: 迭代{model.n_iter_}次, "
                  f"损失从{model.loss_curve_[0]:.4f}到{model.loss_curve_[-1]:.4f}")
        if hasattr(model, 'validation_scores_') and model.validation_scores_ is not None:
            print(f"  MLP验证准确率: {model.validation_scores_[-1]:.4f}")

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 计算指标
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='macro')
    metrics['recall'] = recall_score(y_test, y_pred, average='macro')
    metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')

    # 计算训练集指标
    train_pred = model.predict(X_train)
    metrics['train_accuracy'] = accuracy_score(y_train, train_pred)
    metrics['train_f1'] = f1_score(y_train, train_pred, average='macro')

    # AUC（多类需要处理）
    if len(np.unique(y_test)) == 2:  # 二分类
        metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:  # 多分类
        try:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
        except:
            metrics['auc'] = np.nan

    # 分类报告
    report = classification_report(y_test, y_pred, output_dict=True)

    # 获取特征重要性
    feature_importance = get_feature_importance(model, model_name, feature_names, X_test, y_test)

    return model, metrics, report, y_pred, feature_importance


def get_feature_importance(model, model_name, feature_names, X_test, y_test):
    """获取不同模型的特征重要性"""
    if feature_names is None:
        return None

    importance_df = pd.DataFrame(index=feature_names)

    if model_name == 'rf':
        # 随机森林的特征重要性
        if hasattr(model, 'feature_importances_'):
            importance_df['importance'] = model.feature_importances_
            importance_df['method'] = 'feature_importances'

    elif model_name == 'lr':
        # 逻辑回归的特征重要性（系数的绝对值）
        if hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:  # 二分类
                importance_df['importance'] = np.abs(model.coef_)
            else:  # 多分类
                # 取所有类别系数的平均绝对值
                importance_df['importance'] = np.mean(np.abs(model.coef_), axis=0)
            importance_df['method'] = 'coefficient_abs'

    elif model_name == 'svm' and hasattr(model, 'coef_'):
        # 线性SVM的特征重要性
        if len(model.coef_.shape) == 1:  # 二分类
            importance_df['importance'] = np.abs(model.coef_)
        else:  # 多分类
            # 取所有类别系数的平均绝对值
            importance_df['importance'] = np.mean(np.abs(model.coef_), axis=0)
        importance_df['method'] = 'svm_coefficient_abs'

    elif model_name == 'mlp':
        # MLP的特征重要性（通过权重的绝对值）
        if hasattr(model, 'coefs_'):
            # MLP的coefs_是一个列表，包含每层的权重矩阵
            if len(model.coefs_) > 0:
                # 取第一层（输入层到第一隐藏层）的权重绝对值的平均值
                input_weights = np.abs(model.coefs_[0])
                # 计算每个输入特征的平均影响
                feature_importance_vals = np.mean(input_weights, axis=1)
                importance_df['importance'] = feature_importance_vals
                importance_df['method'] = 'mlp_weight_abs'
            else:
                print("  MLP没有可用的权重信息")

    # 排列重要性（适用于所有模型）
    try:
        perm_importance = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        importance_df['permutation_importance'] = perm_importance.importances_mean
        importance_df['permutation_std'] = perm_importance.importances_std
    except Exception as e:
        print(f" 计算排列重要性时出错: {e}")
        importance_df['permutation_importance'] = np.nan
        importance_df['permutation_std'] = np.nan

    # 归一化重要性分数
    if 'importance' in importance_df.columns and importance_df['importance'].notna().any():
        importance_df['normalized_importance'] = (
                importance_df['importance'] / importance_df['importance'].sum() * 100
        )

    if 'permutation_importance' in importance_df.columns and importance_df['permutation_importance'].notna().any():
        # 排列重要性可能有负值，我们取绝对值
        perm_imp_abs = np.abs(importance_df['permutation_importance'])
        importance_df['normalized_perm_importance'] = (
                perm_imp_abs / perm_imp_abs.sum() * 100
        )

    return importance_df.sort_values('importance', ascending=False) if 'importance' in importance_df.columns else importance_df


# 5. 特征重要性分析函数
def analyze_feature_importance(best_model, best_model_name, feature_names, X_test, y_test,
                               feature_importance_df=None, top_n=20):
    """分析最佳模型的特征重要性"""
    print("\n" + "=" * 60)
    print(f"最佳模型 ({best_model_name}) 特征重要性分析")
    print("=" * 60)

    # 如果提供了特征重要性数据框，直接使用
    if feature_importance_df is not None and not feature_importance_df.empty:
        importance_df = feature_importance_df
    else:
        # 否则重新计算
        best_model_code = ''
        if '随机森林' in best_model_name:
            best_model_code = 'rf'
        elif '逻辑回归' in best_model_name:
            best_model_code = 'lr'
        elif '支持向量机' in best_model_name:
            best_model_code = 'svm'
        elif '多层感知机' in best_model_name or 'MLP' in best_model_name:
            best_model_code = 'mlp'
        else:
            best_model_code = 'rf'  # 默认

        importance_df = get_feature_importance(
            best_model,
            best_model_code,
            feature_names,
            X_test,
            y_test
        )

    if importance_df is None or importance_df.empty:
        print("无法计算特征重要性")
        return None

    # 打印前N个最重要的特征
    print(f"\n前{top_n}个最重要的特征:")
    print("-" * 60)

    # 选择要显示的重要性列
    if 'normalized_importance' in importance_df.columns:
        display_col = 'normalized_importance'
        unit = "%"
    elif 'importance' in importance_df.columns:
        display_col = 'importance'
        unit = ""
    elif 'normalized_perm_importance' in importance_df.columns:
        display_col = 'normalized_perm_importance'
        unit = "%"
    elif 'permutation_importance' in importance_df.columns:
        display_col = 'permutation_importance'
        unit = ""
    else:
        print("没有可用的特征重要性数据")
        return importance_df

    # 显示前top_n个特征
    top_features = importance_df.head(top_n)
    for i, (feat, row) in enumerate(top_features.iterrows(), 1):
        importance_val = row[display_col]
        if unit == "%":
            print(f"{i:2d}. {feat:30s}: {importance_val:6.2f}{unit}")
        else:
            print(f"{i:2d}. {feat:30s}: {importance_val:8.6f}{unit}")

    # 保存特征重要性到CSV
    importance_df.to_csv('feature_importance.csv')
    print(f"\n特征重要性已保存到 feature_importance.csv")

    return importance_df


def plot_feature_importance(importance_df, best_model_name, top_n=20, save_plot=True):
    """绘制特征重要性图"""
    if importance_df is None or importance_df.empty:
        print("没有特征重要性数据可绘制")
        return

    plt.figure(figsize=(12, 8))

    # 选择要绘制的重要性列
    if 'normalized_importance' in importance_df.columns:
        plot_col = 'normalized_importance'
        xlabel = '特征重要性 (%)'
    elif 'importance' in importance_df.columns:
        plot_col = 'importance'
        xlabel = '特征重要性'
    elif 'normalized_perm_importance' in importance_df.columns:
        plot_col = 'normalized_perm_importance'
        xlabel = '排列重要性 (%)'
    elif 'permutation_importance' in importance_df.columns:
        plot_col = 'permutation_importance'
        xlabel = '排列重要性'
    else:
        print("没有可绘制的特征重要性数据")
        return

    # 获取前N个特征
    top_features = importance_df.head(top_n).copy()
    top_features = top_features.sort_values(plot_col, ascending=True)

    # 创建条形图
    bars = plt.barh(range(len(top_features)), top_features[plot_col])
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel(xlabel)
    plt.title(f'{best_model_name} - 前{top_n}个最重要特征')

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, top_features[plot_col])):
        if plot_col.endswith('_importance'):  # 百分比格式
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{val:.1f}%', va='center')
        else:
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{val:.4f}', va='center')

    plt.tight_layout()

    if save_plot:
        plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
        print("特征重要性图已保存到 feature_importance_plot.png")

    plt.show()


def plot_mlp_training_history(mlp_model, model_name="MLP"):
    """绘制MLP训练历史"""
    if hasattr(mlp_model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(mlp_model.loss_curve_, label='训练损失', linewidth=2)

        if hasattr(mlp_model, 'validation_scores_') and mlp_model.validation_scores_ is not None:
            # 如果有验证分数，绘制验证准确率
            plt.plot(mlp_model.validation_scores_, label='验证准确率', linewidth=2)
            plt.xlabel('迭代次数')
            plt.ylabel('损失/准确率')
            plt.title(f'{model_name}训练历史')
        else:
            plt.xlabel('迭代次数')
            plt.ylabel('损失')
            plt.title(f'{model_name}训练损失')

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('mlp_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("MLP训练历史图已保存到 mlp_training_history.png")


# 6. 主函数
def main():
    # 文件路径 - 请修改为你的实际路径
    a_path = "datasets/LS/clinical-o/xh_2.csv"
    b_path = "datasets/LS/clinical-o/zr_2.csv"
    c_path = "datasets/LS/clinical-o/zz_2.csv"

    print("=" * 60)
    print("分类模型训练与评估（增强诊断版）")
    print("=" * 60)

    # 加载数据
    df_a, df_b, df_c = load_data(a_path, b_path, c_path)

    # 准备数据
    X_train, y_train, X_test, y_test, slide_id_train, slide_id_test = prepare_data(df_a, df_b, df_c)

    # 保存特征名称
    feature_names = X_train.columns.tolist()
    print(f"\n特征数量: {len(feature_names)}")
    print(f"特征示例: {feature_names[:5]}...")

    # 数据预处理
    X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, label_encoder, scaler = preprocess_data(
        X_train, X_test, y_train, y_test, feature_names
    )

    # 数据探索
    data_info = explore_data(X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, feature_names)

    # 交叉验证比较
    cv_results = cross_validate_models(
        pd.concat([X_train_scaled, X_test_scaled], ignore_index=True),
        pd.concat([pd.Series(y_train_enc), pd.Series(y_test_enc)], ignore_index=True),
        cv=5
    )

    print("\n" + "=" * 60)
    print("模型训练与评估结果")
    print("=" * 60)

    # 训练不同模型并比较
    models = {
        '随机森林': 'rf',
        '逻辑回归': 'lr',
        '支持向量机': 'svm',
        '多层感知机': 'mlp'
    }

    results = {}
    feature_importances = {}

    for model_name, model_code in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)

        try:
            model, metrics, report, y_pred, feat_importance = train_and_evaluate(
                X_train_scaled, X_test_scaled, y_train_enc, y_test_enc,
                model_code, feature_names
            )

            # 保存结果
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'report': report
            }
            feature_importances[model_name] = feat_importance

            # 打印指标
            print(f"训练准确率: {metrics['train_accuracy']:.4f}")
            print(f"测试准确率: {metrics['accuracy']:.4f}")
            print(f"过拟合差距: {metrics['train_accuracy'] - metrics['accuracy']:.4f}")
            print(f"精确率 (Precision): {metrics['precision']:.4f}")
            print(f"召回率 (Recall): {metrics['recall']:.4f}")
            print(f"宏F1 (Macro F1): {metrics['f1_macro']:.4f}")
            if not np.isnan(metrics['auc']):
                print(f"AUC: {metrics['auc']:.4f}")

        except Exception as e:
            print(f"训练{model_name}时出错: {e}")
            import traceback
            traceback.print_exc()

    # 详细MLP诊断
    mlp_results = train_mlp_detailed(X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, feature_names)

    # LR vs MLP详细对比
    lr_model, mlp_model = compare_lr_mlp(X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, feature_names)

    # 比较模型结果
    print("\n" + "=" * 60)
    print("模型性能比较")
    print("=" * 60)

    comparison_df = pd.DataFrame({
        model_name: {
            '训练准确率': results[model_name]['metrics']['train_accuracy'],
            '测试准确率': results[model_name]['metrics']['accuracy'],
            'F1分数': results[model_name]['metrics']['f1_macro'],
            'AUC': results[model_name]['metrics']['auc']
        }
        for model_name in results
    }).T

    # 按测试准确率排序
    comparison_df = comparison_df.sort_values('测试准确率', ascending=False)
    print(comparison_df.round(4))

    # 选择最佳模型
    best_model_name = comparison_df.index[0]
    best_model = results[best_model_name]['model']
    best_model_feat_importance = feature_importances.get(best_model_name)

    print(f"\n最佳模型: {best_model_name}")
    print(f"最佳测试准确率: {comparison_df.loc[best_model_name, '测试准确率']:.4f}")

    # 分析原因
    print("\n" + "=" * 60)
    print("MLP性能分析")
    print("=" * 60)

    if '多层感知机' in results:
        mlp_metrics = results['多层感知机']['metrics']
        lr_metrics = results['逻辑回归']['metrics'] if '逻辑回归' in results else None

        if lr_metrics:
            acc_diff = lr_metrics['accuracy'] - mlp_metrics['accuracy']
            print(f"逻辑回归比MLP准确率高: {acc_diff:.4f}")

            if acc_diff > 0.05:
                print("\n可能原因分析:")
                print("1. 数据量可能较小，线性模型更稳定")
                print("2. 数据可能近似线性可分")
                print("3. MLP可能过拟合或欠拟合")
                print("4. MLP参数可能需要调整")
                print("5. 特征工程可能需要改进")

                # 检查过拟合
                mlp_overfit = mlp_metrics['train_accuracy'] - mlp_metrics['accuracy']
                if mlp_overfit > 0.1:
                    print(f"   - MLP过拟合明显: 训练-测试差距 = {mlp_overfit:.4f}")

                # 检查训练是否充分
                if hasattr(results['多层感知机']['model'], 'loss_curve_'):
                    final_loss = results['多层感知机']['model'].loss_
                    if final_loss > 0.5:
                        print(f"   - MLP损失较高: {final_loss:.4f}，可能需要更多训练")

    # 分析最佳模型的特征重要性
    importance_df = analyze_feature_importance(
        best_model, best_model_name, feature_names,
        X_test_scaled, y_test_enc, best_model_feat_importance
    )

    # 绘制特征重要性图
    if importance_df is not None and not importance_df.empty:
        plot_feature_importance(importance_df, best_model_name)

    # 如果最佳模型是MLP，绘制训练历史
    if '多层感知机' in best_model_name or 'MLP' in best_model_name:
        plot_mlp_training_history(best_model, best_model_name)

    # 绘制所有模型对比图
    plot_model_comparison(comparison_df)

    # 保存最佳模型预测结果
    test_df = df_c.copy()
    if 'slide_id' in test_df.columns:
        # 确保测试集有slide_id列
        if slide_id_test is not None:
            test_df['slide_id'] = slide_id_test.values

        test_df['predicted_label'] = label_encoder.inverse_transform(
            best_model.predict(X_test_scaled)
        )
        test_df['true_label'] = y_test.values

        # 保存预测结果
        test_df.to_csv('predictions.csv', index=False)
        print(f"\n预测结果已保存到 predictions.csv")

        # 显示预测样本
        print("\n前10个样本的预测结果:")
        print(test_df[['slide_id', 'true_label', 'predicted_label']].head(10))

    # 保存诊断报告
    save_diagnostic_report(data_info, results, cv_results, comparison_df)

    return best_model, label_encoder, results, importance_df


def plot_model_comparison(comparison_df):
    """绘制模型对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['训练准确率', '测试准确率', 'F1分数', 'AUC']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        if metric in comparison_df.columns:
            comparison_df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'模型{metric}对比')
            ax.set_ylabel(metric)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for i, v in enumerate(comparison_df[metric]):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center')

    plt.suptitle('模型性能对比', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("模型对比图已保存到 model_comparison.png")


def save_diagnostic_report(data_info, results, cv_results, comparison_df):
    """保存诊断报告"""
    with open('diagnostic_report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("模型训练诊断报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 数据信息\n")
        f.write("-" * 40 + "\n")
        f.write(f"训练集样本数: {data_info['n_samples_train']}\n")
        f.write(f"测试集样本数: {data_info['n_samples_test']}\n")
        f.write(f"特征数量: {data_info['n_features']}\n")
        f.write(f"类别数量: {data_info['n_classes']}\n")

        f.write("\n2. 类别分布\n")
        for cls, count in data_info['class_distribution_train'].items():
            prop = count / data_info['n_samples_train'] * 100
            f.write(f"类别 {cls}: {count} ({prop:.1f}%)\n")

        f.write("\n3. 模型性能对比\n")
        f.write("-" * 40 + "\n")
        f.write(comparison_df.to_string())

        f.write("\n\n4. 交叉验证结果\n")
        f.write("-" * 40 + "\n")
        for model_name, scores in cv_results.items():
            f.write(f"{model_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})\n")

        f.write("\n5. MLP性能分析\n")
        f.write("-" * 40 + "\n")
        if '多层感知机' in results and '逻辑回归' in results:
            mlp_acc = results['多层感知机']['metrics']['accuracy']
            lr_acc = results['逻辑回归']['metrics']['accuracy']
            diff = lr_acc - mlp_acc

            f.write(f"逻辑回归准确率: {lr_acc:.4f}\n")
            f.write(f"MLP准确率: {mlp_acc:.4f}\n")
            f.write(f"差异: {diff:.4f}\n\n")

            f.write("可能原因:\n")
            if diff > 0.05:
                f.write("- 数据量较小，线性模型更稳定\n")
                f.write("- 数据可能近似线性可分\n")
                f.write("- MLP可能过拟合\n")
                f.write("- MLP参数需要调优\n")
            else:
                f.write("- 差异较小，模型性能相近\n")

    print("诊断报告已保存到 diagnostic_report.txt")


if __name__ == "__main__":
    best_model, label_encoder, results, importance_df = main()

    # 保存模型比较结果
    comparison_df = pd.DataFrame({
        model_name: {
            '训练准确率': results[model_name]['metrics']['train_accuracy'],
            '测试准确率': results[model_name]['metrics']['accuracy'],
            '精确率': results[model_name]['metrics']['precision'],
            '召回率': results[model_name]['metrics']['recall'],
            'F1分数': results[model_name]['metrics']['f1_macro'],
            'AUC': results[model_name]['metrics']['auc']
        }
        for model_name in results
    }).T.sort_values('测试准确率', ascending=False)

    comparison_df.to_csv('model_comparison.csv')
    print(f"\n模型比较结果已保存到 model_comparison.csv")