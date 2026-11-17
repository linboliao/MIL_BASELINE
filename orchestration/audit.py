"""Check selected checkpoints and one-pass held-out prediction coverage."""
from collections import Counter
from pathlib import Path
import math
from .inputs import read_rows


class CellAudit:
    @staticmethod
    def checkpoint(root):
        logs = list((Path(root)/'training').rglob('Best_Log_*.csv'))
        if len(logs) != 1:
            raise ValueError('Expected exactly one Best_Log, found ' + str(len(logs)))
        rows = read_rows(logs[0])
        if len(rows) != 1:
            raise ValueError('Expected one selected epoch')
        row = rows[0]
        for key in ('val_macro_f1','val_loss'):
            if not math.isfinite(float(row[key])):
                raise ValueError('Invalid checkpoint selection evidence')
        epoch = int(row['epoch'])
        path = logs[0].parent/('Best_EPOCH_'+str(epoch)+'.pth')
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing selected checkpoint')
        return path

    @classmethod
    def inspect(cls, root):
        root = Path(root)
        checkpoint = cls.checkpoint(root)
        expected = read_rows(root/'inputs/test.csv')
        prediction = root/'evaluation/Infer_Result.csv'
        observed = read_rows(prediction)
        identities = lambda rows, path, label: Counter((r[path], int(r[label])) for r in rows)
        if identities(expected,'test_slide_path','test_label') != identities(observed,'slide_path','label'):
            raise ValueError('Held-out coverage or labels differ')
        for row in observed:
            probabilities = [float(row['prob_0']),float(row['prob_1'])]
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError('Invalid probability')
            if abs(sum(probabilities)-1) > 1e-5:
                raise ValueError('Probabilities do not sum to one')
            if int(row['prediction']) != int(probabilities[1] > probabilities[0]):
                raise ValueError('Prediction and probabilities disagree')
        return {'cell':str(root), 'checkpoint':str(checkpoint), 'predictions':len(observed), 'status':'passed'}
