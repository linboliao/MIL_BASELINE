"""Synthetic orchestration contracts; retained fixtures, no training or GPU use."""
import json
import tempfile
import unittest
from pathlib import Path
import yaml
from orchestration.study import Study
from orchestration.inputs import FrozenInputs, write_rows, read_rows
from orchestration.audit import CellAudit


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='orchestration-'))
        source = self.root/'source'
        cell = source/'conch/SL'
        (cell/'configs').mkdir(parents=True)
        (cell/'splits').mkdir()
        cfg = {'General':{'MODEL_NAME':'AB_MIL','seed':42},'Dataset':{},'Logs':{},'Model':{}}
        (cell/'configs/model.yaml').write_text(yaml.safe_dump(cfg))
        for index,role in enumerate(FrozenInputs.roles):
            feature = self.root/(role+'.pt')
            feature.write_bytes(b'synthetic')
            row = {'slide_path':str(feature),'label':str(index%2),
                   'patient_group_id':'patient-'+role,'partition':role,'model':'conch','feature_bytes':'9'}
            write_rows(cell/'splits'/(role+'.csv'),[row],list(row))
        self.study = Study('synthetic',source,self.root/'attempts',('conch',),('SL',))
        self.cell = self.study.cells()[0]

    def test_selection_and_disjointness(self):
        with self.assertRaises(ValueError):
            self.study.cells(['unknown'])
        self.assertEqual(len(self.study.cells()),1)
        rows = read_rows(self.cell.splits/'test.csv')
        rows[0]['patient_group_id']='patient-train'
        # This fixture mutation is intentional; no production artifact is touched.
        replacement=self.cell.splits/'test-conflict.csv'
        write_rows(replacement,rows,list(rows[0]))
        from unittest.mock import patch
        from orchestration.inputs import read_rows as real_read
        with patch('orchestration.inputs.read_rows',side_effect=lambda p: real_read(replacement if p.name=='test.csv' else p)):
            with self.assertRaises(ValueError):
                FrozenInputs(self.cell)

    def prepare(self):
        inputs=FrozenInputs(self.cell,check_features=True)
        out=self.root/'prepared'
        out.mkdir()
        config=inputs.materialize(out)
        return out,yaml.safe_load(config.read_text())

    def test_role_adapter_and_no_clobber(self):
        out,cfg=self.prepare()
        row=read_rows(out/'inputs/train_val.csv')[0]
        self.assertEqual(row['test_slide_path'],'')
        self.assertTrue(row['val_slide_path'].endswith('selection.pt'))
        self.assertTrue(cfg['General']['retain_all_checkpoints'])
        self.assertEqual(cfg['General']['checkpoint_tie_break'],'val_loss')
        self.assertFalse(cfg['Dataset']['preload']['test'])
        with self.assertRaises(FileExistsError):
            FrozenInputs(self.cell).materialize(out)

    def test_feature_drift(self):
        from unittest.mock import patch
        from orchestration.inputs import read_rows as real_read
        def changed(path):
            rows=real_read(path)
            rows[0]['feature_bytes']='999'
            return rows
        with patch('orchestration.inputs.read_rows',side_effect=changed):
            with self.assertRaises(ValueError):
                FrozenInputs(self.cell,check_features=True)

    def test_checkpoint_and_coverage_audit(self):
        out,cfg=self.prepare()
        training=out/'training/run'
        training.mkdir(parents=True)
        write_rows(training/'Best_Log_seed42.csv',[{'epoch':'2','val_macro_f1':'0.8','val_loss':'0.4'}],
                   ['epoch','val_macro_f1','val_loss'])
        (training/'Best_EPOCH_2.pth').write_bytes(b'synthetic-checkpoint')
        (out/'evaluation').mkdir()
        row=read_rows(out/'inputs/test.csv')[0]
        prediction={'slide_path':row['test_slide_path'],'label':row['test_label'],
                    'prob_0':'0.2','prob_1':'0.8','prediction':'1'}
        write_rows(out/'evaluation/Infer_Result.csv',[prediction],list(prediction))
        self.assertEqual(CellAudit.inspect(out)['predictions'],1)
        (training/'Best_EPOCH_3.pth').write_bytes(b'unselected')
        self.assertTrue(str(CellAudit.checkpoint(out)).endswith('Best_EPOCH_2.pth'))

    def test_executor_pipeline_without_training(self):
        from orchestration.execution import LocalExecutor
        executor=LocalExecutor.__new__(LocalExecutor)
        executor.repo=self.root
        executor.host='synthetic'
        executor.gpu='0'
        executor.commit='synthetic'
        stages=[]
        def invoke(args, logfile):
            stages.append(args[0])
            root=Path(logfile).parent
            if args[0]=='train_mil.py':
                training=root/'training/native'
                training.mkdir(parents=True)
                write_rows(training/'Best_Log_seed42.csv',
                           [{'epoch':'1','val_macro_f1':'0.7','val_loss':'0.3'}],
                           ['epoch','val_macro_f1','val_loss'])
                (training/'Best_EPOCH_1.pth').write_bytes(b'synthetic')
            else:
                output=root/'evaluation'
                output.mkdir()
                row=read_rows(root/'inputs/test.csv')[0]
                pred={'slide_path':row['test_slide_path'],'label':row['test_label'],
                      'prob_0':'0.3','prob_1':'0.7','prediction':'1'}
                write_rows(output/'Infer_Result.csv',[pred],list(pred))
        executor._invoke=invoke
        root=executor.run(self.study,[self.cell],'synthetic-r1')
        self.assertEqual(stages,['train_mil.py','test_mil.py'])
        self.assertTrue((root/'completed.json').is_file())
        self.assertTrue((root/'conch/SL/completed.json').is_file())
        with self.assertRaises(FileExistsError):
            executor.run(self.study,[self.cell],'synthetic-r1')

    def test_missing_outputs_fail(self):
        out,cfg=self.prepare()
        with self.assertRaises(ValueError):
            CellAudit.inspect(out)


if __name__=='__main__':
    unittest.main()
