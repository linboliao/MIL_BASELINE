import pandas as pd, os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from scipy.optimize import minimize_scalar

ROOT = '/NAS3/lbliao/Code-138/MIL_BASELINE/result/ProstateDiagnosis/DataAnalysis'


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(probs, labels):
    z = logit(probs)

    def nll(T):
        p = sigmoid(z / T)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(0.05, 20), method='bounded')
    return res.x


for version in ['AB_MIL_uni2_5fold_no_extsl', 'AB_MIL_uni2_5fold_sl_dev_only', 'AB_MIL_uni2_5fold_3center']:
    print(f'=== {version}: temperature scaling (fit on internal_test, apply to 301) ===')
    accs_default, accs_calibrated = [], []
    for fold in range(1, 6):
        val_path = os.path.join(ROOT, version, 'internal_test', f'fold_{fold}', 'Infer_Result.csv')
        val_df = pd.read_csv(val_path)
        T = fit_temperature(val_df['prob_1'].values, val_df['label'].values)

        ext_path = os.path.join(ROOT, version, 'external_test', '301', f'fold_{fold}', 'Infer_Result.csv')
        ext_df = pd.read_csv(ext_path)
        p_cal = sigmoid(logit(ext_df['prob_1'].values) / T)

        pred_default = (ext_df['prob_1'] >= 0.5).astype(int)
        pred_cal = (p_cal >= 0.5).astype(int)

        acc_d = accuracy_score(ext_df['label'], pred_default)
        acc_c = accuracy_score(ext_df['label'], pred_cal)
        f1_d = f1_score(ext_df['label'], pred_default, average='macro')
        f1_c = f1_score(ext_df['label'], pred_cal, average='macro')
        bacc_c = balanced_accuracy_score(ext_df['label'], pred_cal)

        accs_default.append(acc_d)
        accs_calibrated.append(acc_c)

        print(f'fold{fold}: T={T:.3f} | acc {acc_d:.4f} -> {acc_c:.4f} | f1 {f1_d:.4f} -> {f1_c:.4f} | bacc_cal={bacc_c:.4f}')
    print(f'  mean acc: {np.mean(accs_default):.4f} -> {np.mean(accs_calibrated):.4f} | '
          f'std: {np.std(accs_default):.4f} -> {np.std(accs_calibrated):.4f}')
    print()
