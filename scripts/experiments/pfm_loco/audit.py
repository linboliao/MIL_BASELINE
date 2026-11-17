"""Read-only PFM LOCO held-out coverage and checkpoint audit."""
import argparse
from collections import Counter
from pathlib import Path
import math
from .prepare_split import read
from ..run_fold import selected_checkpoint


def inspect(root):
    root=Path(root)
    checkpoint=selected_checkpoint(root)
    expected=read(root/'test_dataset.csv')
    observed=read(root/'evaluation/Infer_Result.csv')
    def identities(data,path,label):
        return Counter((r[path],int(r[label])) for r in data)
    if identities(expected,'test_slide_path','test_label')!=identities(observed,'slide_path','label'):
        raise ValueError('Held-out coverage or labels differ')
    for row in observed:
        p=[float(row['prob_0']),float(row['prob_1'])]
        if any(not math.isfinite(v) or not 0<=v<=1 for v in p) or abs(sum(p)-1)>1e-5:
            raise ValueError('Invalid probabilities')
        if int(row['prediction'])!=int(p[1]>p[0]):
            raise ValueError('Prediction and probability disagree')
    return {'checkpoint':str(checkpoint),'predictions':len(observed),'status':'passed'}


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    print(json.dumps(inspect(p.parse_args().output),indent=2))


if __name__=='__main__':
    main()
