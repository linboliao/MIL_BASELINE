from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import argparse


def Balanced_K_fold_Train_Val(args):
    csv_path = args.csv_path
    k = args.k
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['label'])
    y = df['label']

    feature_columns = X.columns.tolist()

    save_dir = args.save_dir
    dataset_name = args.dataset_name
    skf = StratifiedKFold(n_splits=k, random_state=args.seed, shuffle=True)

    for k_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        max_len = max(len(X_train), len(X_test))

        result_df = pd.DataFrame()

        # 首先添加训练集的所有特征
        for col in feature_columns:
            train_values = X_train[col].tolist()
            train_values += [np.nan] * (max_len - len(train_values))
            result_df[f'train_{col}'] = train_values

        # 然后添加训练集标签
        train_labels = y_train.tolist()
        train_labels += [np.nan] * (max_len - len(train_labels))
        result_df['train_label'] = train_labels

        # 接着添加验证集的所有特征
        for col in feature_columns:
            val_values = X_test[col].tolist()
            val_values += [np.nan] * (max_len - len(val_values))
            result_df[f'val_{col}'] = val_values

        # 然后添加验证集标签
        val_labels = y_test.tolist()
        val_labels += [np.nan] * (max_len - len(val_labels))
        result_df['val_label'] = val_labels

        # 最后添加测试集的所有特征（全部为nan）
        for col in feature_columns:
            result_df[f'test_{col}'] = [np.nan] * max_len

        # 添加测试集标签
        result_df['test_label'] = [np.nan] * max_len

        # 保存结果
        os.makedirs(f'{args.save_dir}/{args.dataset_name}', exist_ok=True)
        result_df.to_csv(f'{save_dir}/{args.dataset_name}/Total_{k}-fold_{dataset_name}_{k_idx + 1}fold.csv', index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--k', type=int, default=3)
    argparser.add_argument('--save_dir', type=str, default='/dir/to/save/dataset/csvs')
    args = argparser.parse_args()
    Balanced_K_fold_Train_Val(args)