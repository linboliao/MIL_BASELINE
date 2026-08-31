import pandas as pd, os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

ROOT = '/NAS3/lbliao/Code-138/MIL_BASELINE/result/ProstateDiagnosis/DataAnalysis'


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


for version in ['AB_MIL_uni2_5fold_no_extsl', 'AB_MIL_uni2_5fold_sl_dev_only']:
    print(f'=== {version}: Platt scaling (a*logit+b, fit on internal_test) applied to 301 ===')
    accs_default, accs_cal = [], []
    for fold in range(1, 6):
        val_path = os.path.join(ROOT, version, 'internal_test', f'fold_{fold}', 'Infer_Result.csv')
        val_df = pd.read_csv(val_path)
        z_val = logit(val_df['prob_1'].values).reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(z_val, val_df['label'].values)

        ext_path = os.path.join(ROOT, version, 'external_test', '301', f'fold_{fold}', 'Infer_Result.csv')
        ext_df = pd.read_csv(ext_path)
        z_ext = logit(ext_df['prob_1'].values).reshape(-1, 1)
        p_cal = lr.predict_proba(z_ext)[:, 1]

        pred_default = (ext_df['prob_1'] >= 0.5).astype(int)
        pred_cal = (p_cal >= 0.5).astype(int)

        acc_d = accuracy_score(ext_df['label'], pred_default)
        acc_c = accuracy_score(ext_df['label'], pred_cal)
        f1_d = f1_score(ext_df['label'], pred_default, average='macro')
        f1_c = f1_score(ext_df['label'], pred_cal, average='macro')
        bacc_c = balanced_accuracy_score(ext_df['label'], pred_cal)

        accs_default.append(acc_d)
        accs_cal.append(acc_c)

        print(f'fold{fold}: a={lr.coef_[0][0]:.3f} b={lr.intercept_[0]:.3f} | '
              f'acc {acc_d:.4f} -> {acc_c:.4f} | f1 {f1_d:.4f} -> {f1_c:.4f} | bacc_cal={bacc_c:.4f}')
    print(f'  mean acc: {np.mean(accs_default):.4f} -> {np.mean(accs_cal):.4f} | '
          f'std: {np.std(accs_default):.4f} -> {np.std(accs_cal):.4f}')
    print()
