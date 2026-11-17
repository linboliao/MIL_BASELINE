"""Independent script contracts; fixtures retained and cleanup deletion mocked."""
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import yaml
from scripts.experiments.run_fold import prepare as prepare_fold, validate_dataset
from scripts.experiments.pfm_loco.prepare_split import prepare as prepare_roles, write, read
from scripts.experiments.pfm_loco.stage import stage, digest
from scripts.experiments.pfm_loco.clean_stage import deletion_plan, clean
from scripts.experiments.pfm_loco.audit import inspect


class ExperimentScriptTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='experiment-scripts-'))
        self.config=self.root/'anything.yaml'
        self.config.write_text(yaml.safe_dump({'General':{'MODEL_NAME':'NN_MIL','seed':7,
            'best_model_metric':'macro_auc'},'Dataset':{'preload':False},'Logs':{},'Model':{'in_dim':1024}}))
        self.roles={}
        for i,role in enumerate(('train','selection','test','calibration')):
            feature=self.root/(role+'.pt')
            feature.write_bytes(b'feature')
            path=self.root/(role+'.csv')
            row={'slide_path':str(feature),'label':str(i%2),'patient_group_id':role,'partition':role}
            write(path,[row],list(row))
            self.roles[role]=path
        self.prepared=prepare_roles(self.roles['train'],self.roles['selection'],self.roles['test'],
                                    self.root/'prepared',self.roles['calibration'])
        self.dataset=self.prepared/'train_val.csv'
        self.test=self.prepared/'test.csv'

    def test_generic_fold_preserves_scientific_config(self):
        output=self.root/'run'
        prepare_fold(self.config,self.dataset,output,self.test)
        cfg=yaml.safe_load((output/'runtime.yaml').read_text())
        self.assertEqual(cfg['General']['MODEL_NAME'],'NN_MIL')
        self.assertEqual(cfg['General']['seed'],7)
        self.assertEqual(cfg['General']['best_model_metric'],'macro_auc')
        self.assertNotIn('checkpoint_tie_break',cfg['General'])
        self.assertTrue(cfg['General']['retain_all_checkpoints'])
        self.assertFalse(cfg['Dataset']['preload'])
        self.assertEqual(read(output/'dataset.csv'),read(self.dataset))
        with self.assertRaises(FileExistsError):
            prepare_fold(self.config,self.dataset,output,self.test)

    def test_explicit_tie_override(self):
        output=self.root/'tie'
        prepare_fold(self.config,self.dataset,output,self.test,'val_loss')
        self.assertEqual(yaml.safe_load((output/'runtime.yaml').read_text())['General']['checkpoint_tie_break'],'val_loss')

    def test_patient_overlap_rejected(self):
        row=read(self.roles['test'])[0]
        row['patient_group_id']='train'
        conflict=self.root/'conflict.csv'
        write(conflict,[row],list(row))
        with self.assertRaises(ValueError):
            prepare_roles(self.roles['train'],self.roles['selection'],conflict,self.root/'bad')
        self.assertFalse((self.root/'bad').exists())

    def test_staging_is_deduplicated_and_byte_identical(self):
        original=digest(self.dataset)
        manifest=stage([self.dataset,self.dataset],self.root/'cache')
        self.assertEqual(len(manifest['files']),2)
        self.assertEqual(len(manifest['datasets']),2)
        self.assertEqual(original,digest(self.dataset))
        for item in manifest['files']:
            self.assertEqual(digest(item['source']),digest(item['path']))
        staged=self.root/'cache/datasets/0.csv'
        self.assertEqual(validate_dataset(staged),1)
        row=read(staged)[0]
        self.assertEqual(row['train_sample_id'],str(self.root/'train.pt'))
        prepare_fold(self.config,staged,self.root/'staged-run',self.test)
        with self.assertRaises(FileExistsError):
            stage([self.dataset],self.root/'cache')

    def test_independent_test_forbids_epoch_test(self):
        row=read(self.dataset)[0]
        row.update(test_slide_path=str(self.root/'test.pt'),test_label='0')
        path=self.root/'leaky.csv'
        write(path,[row],list(row))
        with self.assertRaises(ValueError):
            validate_dataset(path,forbid_test=True)

    def test_cleanup_rejects_shared_or_wrong_host(self):
        stage([self.dataset],self.root/'shared-cache')
        manifest=self.root/'shared-cache/manifest.json'
        with self.assertRaises(ValueError):
            deletion_plan(manifest)
        self.assertTrue((self.root/'shared-cache/features').is_dir())
        with patch('scripts.experiments.pfm_loco.clean_stage.socket.gethostname',return_value='other-node'):
            with self.assertRaises(ValueError):
                deletion_plan(manifest)

    def test_cleanup_plan_validates_hash_and_lists_only_features(self):
        stage([self.dataset],self.root/'verified-cache')
        manifest=self.root/'verified-cache/manifest.json'
        with patch('scripts.experiments.pfm_loco.clean_stage.LOCAL_CACHE_ROOTS',(self.root,)):
            targets=deletion_plan(manifest)
            self.assertEqual(len(targets),2)
            self.assertTrue(all(p.parent==self.root/'verified-cache/features' for p in targets))
            # Synthetic cache tampering must prevent ALL cleanup, not just one file.
            targets[0].write_bytes(b'changed-fixture')
            with self.assertRaises(ValueError):
                deletion_plan(manifest)
        self.assertTrue(all(p.exists() for p in targets))

    def test_cleanup_only_calls_validated_targets_without_real_deletion(self):
        target=self.root/'train.pt'
        with patch('scripts.experiments.pfm_loco.clean_stage.deletion_plan',return_value=[target]), \
             patch('scripts.experiments.pfm_loco.clean_stage.os.unlink') as remove:
            clean('synthetic-manifest',self.root/'receipt.jsonl')
            remove.assert_called_once_with(target)
        self.assertTrue(target.is_file())
        self.assertIn('completed',(self.root/'receipt.jsonl').read_text())

    def test_audit_coverage(self):
        output=self.root/'audit'
        prepare_fold(self.config,self.dataset,output,self.test)
        training=output/'training/native'
        training.mkdir(parents=True)
        write(training/'Best_Log_seed7.csv',[{'epoch':'2'}],['epoch'])
        (training/'Best_EPOCH_2.pth').write_bytes(b'checkpoint')
        (output/'evaluation').mkdir()
        source=read(self.test)[0]
        pred={'slide_path':source['test_slide_path'],'label':source['test_label'],
              'prob_0':'0.7','prob_1':'0.3','prediction':'0'}
        write(output/'evaluation/Infer_Result.csv',[pred],list(pred))
        self.assertEqual(inspect(output)['predictions'],1)
        (training/'Best_EPOCH_3.pth').write_bytes(b'unselected')
        self.assertTrue(inspect(output)['checkpoint'].endswith('Best_EPOCH_2.pth'))


if __name__=='__main__':
    unittest.main()
