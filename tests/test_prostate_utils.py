"""CPU regression tests; synthetic artifacts are retained under TMPDIR."""
import ast
import copy
import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from addict import Dict

from utils.wsi_utils import (WSI_Dataset, WSI_Coord_Dataset, CDP_MIL_WSI_Dataset,
                             LONG_MIL_WSI_Dataset, SC_MIL_WSI_Dataset)
from utils.data_loading import dataset_options, DataLoader as CacheDataLoader
from utils.loop_utils import val_loop, ac_val_loop, clam_val_loop, ds_val_loop, dtfd_val_loop, cal_scores
from utils.evaluation_utils import evaluate_mil
from utils.nnmil_utils import nnmil_val_loop, fixed_bag_collate
from utils.model_utils import get_criterion, model_select
from utils.repro_utils import write_run_metadata
from utils.prediction_utils import PredictionRecorder
from modules.NN_MIL.nn_mil import NN_MIL


class Toy(nn.Module):
    def forward(self, x, label=None, coords=None, return_WSI_feature=False, return_WSI_attn=False):
        logits = x.mean(dim=1)[:, :2]
        return {'logits': logits, 'max_prediction': logits+.25,
                'instance_loss': logits.sum()*0+.4,
                'sub_preds': logits.repeat_interleave(2, dim=0),
                'attns': torch.ones(len(x), 2, x.shape[1], device=x.device),
                'WSI_feature': x.mean(dim=1), 'WSI_attn': x[..., 0]}


class Attention(nn.Module):
    def forward(self, x):
        return torch.softmax(x[:, 0], dim=0).unsqueeze(0)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x, **kwargs):
        pooled = x.mean(dim=0, keepdim=True)
        return {'logits': self.fc(pooled), 'WSI_feature': pooled,
                'WSI_attn': torch.ones(1, len(x))}


class UtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get('TMPDIR'):
            raise RuntimeError("Set TMPDIR to the project's node-specific scratch directory")
        cls.root = Path(tempfile.mkdtemp(prefix='prostate-utils-', dir=os.environ['TMPDIR']))
        torch.manual_seed(41)
        cls.paths = []
        for i in range(6):
            p = cls.root/f'sample_{i}.pt'
            feat = torch.randn(8, 3)
            feat[:, i % 2] += 2.
            torch.save({'feats': feat, 'coords': torch.arange(16).reshape(8, 2)}, p)
            cls.paths.append(str(p))
        frame = {}
        for split in ('train', 'val', 'test'):
            frame[split+'_slide_path'] = cls.paths
            frame[split+'_label'] = [0, 1, 0, 1, 0, 1]
            frame[split+'_sample_id'] = [f'id-{i}' for i in range(6)]
        cls.csv = cls.root/'splits.csv'
        pd.DataFrame(frame).to_csv(cls.csv, index=False)

    def loader(self, coords=False):
        ds = (WSI_Coord_Dataset if coords else WSI_Dataset)(self.csv, 'test')
        return DataLoader(ds, batch_size=1, shuffle=False)

    def assert_equivalent(self, before, after):
        self.assertAlmostEqual(before[0], after[0], places=6)
        for key in before[1]:
            np.testing.assert_allclose(before[1][key], after[1][key], equal_nan=True)

    def check_loop(self, fn, args, name):
        before = fn(*args)
        path = self.root/f'{name}.csv'
        after = fn(*args, return_predictions=True, prediction_path=path)
        self.assert_equivalent(before, after)
        frame = after[2]
        self.assertEqual(len(frame), 6)
        self.assertEqual(frame.sample_id.tolist(), [f'id-{i}' for i in range(6)])
        self.assertTrue({'slide_id','slide_path','prob_0','prob_1','logit_0','logit_1','prediction'}.issubset(frame.columns))
        np.testing.assert_allclose(frame[['prob_0','prob_1']].sum(axis=1), 1., atol=1e-6)
        with self.assertRaises(FileExistsError):
            fn(*args, prediction_path=path)

    def test_all_shared_evaluation_paths(self):
        device, loss, model = torch.device('cpu'), nn.CrossEntropyLoss(), Toy()
        for fn, extra, name in ((val_loop, (), 'generic'), (ac_val_loop, (2,), 'ac'),
                                (clam_val_loop, (.7,), 'clam'), (ds_val_loop, (), 'ds')):
            with self.subTest(name=name):
                self.check_loop(fn, (device,2,model,self.loader(),loss,*extra), name)
        for distill in ('AFS','MaxS','MaxMinS'):
            modules=[nn.Linear(3,2), Attention(), nn.Identity(), Head()]
            self.check_loop(dtfd_val_loop,
                            (device,2,modules,self.loader(),loss,2,5.,distill,4), 'dtfd_'+distill)
        self.check_loop(lambda *a, **k: evaluate_mil(*a, kind='sc', **k),
                        (device,2,model,self.loader(coords=True),loss), 'sc')
        nnmodel=NN_MIL(in_dim=3, hidden_dim=8, num_classes=2, dropout=0.,
                       feature_select=False, cover_shuffle=False)
        self.check_loop(nnmil_val_loop,(device,2,nnmodel,self.loader(),loss),'nn')

    def test_ce_bce_and_multiclass(self):
        device = torch.device('cpu')
        for criterion in (nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()):
            result=val_loop(device,2,Toy(),self.loader(),criterion,return_predictions=True)
            self.assertEqual(len(result[2]),6)
        p=np.array([[.8,.15,.05],[.1,.8,.1],[.2,.1,.7],[.4,.3,.3],[.3,.4,.3],[.3,.3,.4]])
        y=[0,1,2,0,1,2]
        from sklearn.metrics import roc_auc_score
        self.assertAlmostEqual(cal_scores(p,y,3)['macro_auc'],roc_auc_score(y,p,multi_class='ovr',average='macro'))
        self.assertTrue(np.isnan(cal_scores([[.8,.2]], [0],2)['macro_auc']))
        with self.assertRaises(ValueError):
            cal_scores([[2.,-1.]], [0],2)

    def test_feature_attention_modes(self):
        device, model, loss=torch.device('cpu'),Toy(),nn.CrossEntropyLoss()
        self.assertEqual(val_loop(device,2,model,self.loader(),loss,retrun_WSI_feature=True).shape,(6,3))
        self.assertEqual(len(val_loop(device,2,model,self.loader(),loss,return_WSI_attn=True)),6)
        with self.assertRaises(ValueError):
            val_loop(device,2,model,self.loader(),loss,retrun_WSI_feature=True,return_predictions=True)

    def test_cache_hits_budget_and_multiworker_policy(self):
        raw=WSI_Dataset(self.csv,'train')
        with patch('utils.wsi_utils.torch.load', wraps=torch.load) as load:
            with dataset_options({'preload':True,'preload_mem_gb':.01,'preload_workers':2}):
                ds=WSI_Dataset(self.csv,'train')
            for _ in range(2):
                for i in range(len(ds)):
                    np.testing.assert_allclose(ds[i][0],raw[i][0])
            # raw's own loads account for another 12 calls.
            self.assertEqual(load.call_count,18)
            self.assertTrue(ds.cache_info()['complete'])
            self.assertLessEqual(ds.used_memory,ds.max_memory)
            cached_loads=load.call_count
            list(CacheDataLoader(ds,batch_size=1,num_workers=2))
            self.assertEqual(load.call_count,cached_loads)
        tiny=WSI_Dataset(self.csv,'train',preload=True,preload_mem_gb=1e-12)
        tiny[0]
        self.assertEqual(tiny.used_memory,0)
        self.assertEqual(len(raw[0]),2)
        meta=WSI_Dataset(self.csv,'train',return_metadata=True)
        self.assertEqual(meta[0][2],'id-0')
        # NN_MIL training's pair-only collate remains compatible.
        fixed_bag_collate([raw[0],raw[1]],bag_size=4)

    def test_coordinate_and_legacy_datasets_cache(self):
        import h5py
        hp=self.root/'coords.h5'
        with h5py.File(hp,'w') as f:
            f['features']=np.ones((8,3),np.float32)
            f['coords']=np.arange(16).reshape(8,2)
        index=self.root/'h5index.csv'
        pd.DataFrame({'h5_path':[str(hp)]}).to_csv(index,index=False)
        ds=WSI_Coord_Dataset(self.csv,'test',preload=True,preload_mem_gb=.01)
        self.assertEqual(ds[0][0].shape,(8,5))
        long=LONG_MIL_WSI_Dataset(self.csv,index,'test')
        long.h5_path_list=np.array([str(hp)])
        # Explicit matching stems are required, no silent pseudo-coordinates.
        with self.assertRaises(FileNotFoundError):
            long[0]
        with dataset_options({'preload':True,'preload_mem_gb':.01}):
            sc=SC_MIL_WSI_Dataset(self.csv,group='test')
        self.assertEqual(sc[0][0].shape,(8,5))
        one=self.root/'h5split.csv'
        pd.DataFrame({'test_slide_path':[str(hp)],'test_label':[0]}).to_csv(one,index=False)
        self.assertEqual(WSI_Coord_Dataset(one,'test')[0][0].shape,(8,5))

    def test_identity_and_csv_validation(self):
        ds=WSI_Dataset(self.csv,'test')
        shuffled=DataLoader(ds,batch_size=1,sampler=RandomSampler(ds))
        with self.assertRaises(ValueError):
            val_loop(torch.device('cpu'),2,Toy(),shuffled,nn.CrossEntropyLoss(),return_predictions=True)
        ds=WSI_Dataset(self.csv,'test',return_metadata=True)
        frame=val_loop(torch.device('cpu'),2,Toy(),DataLoader(ds,batch_size=1,sampler=RandomSampler(ds)),nn.CrossEntropyLoss(),return_predictions=True)[2]
        self.assertEqual(set(frame.sample_id),{f'id-{i}' for i in range(6)})
        bad=self.root/'bad.csv'
        pd.DataFrame({'train_slide_path':[self.paths[0]],'train_label':[None]}).to_csv(bad,index=False)
        with self.assertRaises(ValueError):
            WSI_Dataset(bad,'train')

    def test_criterion_selection_and_metadata(self):
        self.assertEqual(get_criterion('ce').label_smoothing,0.)
        self.assertEqual(get_criterion(Dict(loss='ce',label_smoothing=.1)).label_smoothing,.1)
        with self.assertRaises(ValueError):
            get_criterion({'loss':'bce','label_smoothing':.1})
        args=Dict(General={'checkpoint_tie_break':'val_loss'},Logs={'now_log_dir':str(self.root)})
        with patch('utils.model_utils.save_best_model') as save:
            a=model_select(False,args,{}, {'macro_f1':.8},'macro_f1',0.,0,1,val_loss=.5)
            b=model_select(False,args,{}, {'macro_f1':.8},'macro_f1',a[0],1,a[1],val_loss=.4)
            self.assertEqual(b[1],2)
            self.assertEqual(save.call_count,2)
        md=self.root/'metadata'
        write_run_metadata(md, Dict(General={'seed':42}),self.csv)
        with self.assertRaises(FileExistsError):
            write_run_metadata(md, Dict(),self.csv)


    def test_test_cli_saves_predictions_without_clobber(self):
        from types import SimpleNamespace
        import test_mil as entry
        yaml_path=self.root/'test_entry_config.yaml'
        yaml_path.write_text('General: {}')
        weights=self.root/'toy_state.pt'
        torch.save(Toy().state_dict(),weights)
        settings=Dict(General={'MODEL_NAME':'AB_MIL','num_classes':2,'device':0},
                      Dataset={'DATASET_NAME':'synthetic'},Model={'criterion':'ce'})
        args=SimpleNamespace(yaml_path=str(yaml_path),test_dataset_csv=str(self.csv),
                             model_weight_path=str(weights),test_log_dir=str(self.root/'entry_output'),
                             device='cpu')
        with patch.object(entry,'read_yaml',return_value=settings), patch.object(entry,'get_model_from_yaml',return_value=Toy()):
            entry.test(args)
            with self.assertRaises(FileExistsError):
                entry.test(args)
        table=pd.read_csv(self.root/'entry_output/Infer_Result.csv')
        self.assertEqual(len(table),6)
        self.assertIn('logit_1',table)
        self.assertTrue((self.root/'entry_output/run_metadata.json').is_file())

    def test_actual_abmil_forward_and_predictions(self):
        from utils.model_utils import get_model_from_yaml
        config=Dict(General={'MODEL_NAME':'AB_MIL','num_classes':2},
                    Model={'L':8,'D':4,'in_dim':3,'dropout':0.,'act':'relu'})
        model=get_model_from_yaml(config)
        result=val_loop(torch.device('cpu'),2,model,self.loader(),nn.CrossEntropyLoss(),return_predictions=True)
        self.assertEqual(len(result[2]),6)
        self.assertEqual(list(result[2].label),[0,1,0,1,0,1])


    def test_cpu_training_entrypoint_with_preload_and_predictions(self):
        import subprocess, sys, yaml
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'tests/test_nn_mil.py').read_text())
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                        and any(isinstance(x,ast.Name) and x.id=='config' for x in n.targets))
        directory=self.root/'training_entry'
        directory.mkdir()
        namespace={'dataset_path':str(self.csv),'logs':str(directory/'logs')}
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<synthetic-config>','exec'),namespace)
        config=namespace['config']
        config['General'].update(save_predictions=True, checkpoint_tie_break='val_loss')
        config['Dataset'].update(preload=True, preload_mem_gb=.01)
        config['Model']['in_dim']=3
        cfg=directory/'config.yaml'
        cfg.write_text(yaml.safe_dump(config))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        run=subprocess.run([sys.executable,'train_mil.py','--yaml_path',str(cfg)],
                           cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        self.assertEqual(run.returncode,0,run.stdout)
        self.assertEqual(len(list((directory/'logs').rglob('val_epoch_0001.csv'))),1)
        self.assertEqual(len(list((directory/'logs').rglob('test_epoch_0001.csv'))),1)
        self.assertEqual(len(list((directory/'logs').rglob('run_metadata.json'))),1)
        self.assertTrue(list((directory/'logs').rglob('Best_EPOCH_1.pth')))

    def test_all_process_calls_are_wired(self):
        root=Path(__file__).resolve().parents[1]
        total=0
        for path in (root/'process').rglob('*.py'):
            tree=ast.parse(path.read_text())
            for call in ast.walk(tree):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and 'val_loop' in call.func.id:
                    self.assertIn('prediction_path', {k.arg for k in call.keywords},str(path))
                    total+=1
        self.assertGreater(total,180)


if __name__=='__main__':
    unittest.main()
