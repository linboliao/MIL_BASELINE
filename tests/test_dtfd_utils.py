"""DTFD grouping/distillation boundary and backward regression tests."""
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from utils.dtfd_utils import dtfd_groups, dtfd_distill_features
from utils.evaluation_utils import _dtfd_forward
from utils.loop_utils import dtfd_train_loop, dtfd_val_loop


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x).softmax(dim=0).T


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x, **kwargs):
        pooled = x.mean(dim=0, keepdim=True)
        return {'logits': self.fc(pooled), 'WSI_feature': pooled,
                'WSI_attn': torch.ones(1, len(x), device=x.device)}


def models(identity=False):
    return [nn.Linear(3, 2), Attention(),
            nn.Identity() if identity else nn.Linear(3, 3), Head()]


class TestDTFDBoundaries(unittest.TestCase):
    def test_exact_balanced_groups_and_budget_remainders(self):
        bag = torch.arange(33.).reshape(1, 11, 3)
        groups, budgets = dtfd_groups(bag, 5, 7, 'MaxS')
        self.assertEqual([len(x) for x in groups], [3, 2, 2, 2, 2])
        self.assertEqual(budgets, [2, 2, 1, 1, 1])
        self.assertEqual(sum(budgets), 7)
        self.assertTrue(torch.equal(torch.cat(groups), bag[0]))

    def test_afs_does_not_require_instance_budget(self):
        bag = torch.ones(1, 3, 3)
        for budget in (None, 0, 1, 4):
            groups, quotas = dtfd_groups(bag, 5, budget, 'AFS')
            self.assertEqual(len(groups), 3)
            self.assertEqual(quotas, [None]*3)
            self.assertEqual(dtfd_distill_features(models(), bag, 5, budget, 'AFS')[0].shape, (3, 3))

    def test_invalid_parameters_fail_explicitly(self):
        bag = torch.ones(1, 11, 3)
        for group in (0, -1, True, 1.5, None):
            with self.subTest(group=group), self.assertRaises(ValueError):
                dtfd_groups(bag, group, 5, 'AFS')
        for budget in (0, -1, True, 1.5, None, 4):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                dtfd_groups(bag, 5, budget, 'MaxMinS')
        for invalid in (torch.empty(1, 0, 3), torch.ones(2, 11, 3), torch.ones(3)):
            with self.assertRaises(ValueError):
                dtfd_groups(invalid, 5, 5, 'AFS')
        with self.assertRaises(ValueError):
            dtfd_groups(bag, 5, 5, 'unknown')

    def test_small_bags_and_oversized_budgets_never_duplicate_patches(self):
        for n in (1, 2, 3, 7, 11):
            bag = torch.arange(n*3.).reshape(1, n, 3)
            for mode in ('MaxS', 'MaxMinS'):
                for budget in (5, 11, 100):
                    pseudo, sub = dtfd_distill_features(models(identity=True), bag, 5, budget, mode)
                    self.assertGreater(len(pseudo), 0)
                    self.assertLessEqual(len(pseudo), n)
                    self.assertEqual(len(torch.unique(pseudo, dim=0)), len(pseudo))
                    self.assertEqual(sub.shape[0], min(n, 5))

    def test_default_config_and_eval_share_distillation(self):
        from pathlib import Path
        config = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/DTFD_MIL.yaml').read_text())['Model']
        m = models()
        for module in m:
            module.eval()
        bag = torch.randn(1, 13, 3)
        pseudo, _ = dtfd_distill_features(m, bag, config['num_Group'],
                                         config['total_instance'], config['distill'])
        expected = m[3](pseudo)['logits']
        actual = _dtfd_forward(m, bag, config['num_Group'], config['total_instance'], config['distill'])['logits']
        torch.testing.assert_close(expected, actual)

    def test_training_backward_and_validation_for_all_modes_and_small_bags(self):
        for n in (1, 3, 11):
            for mode in ('AFS', 'MaxS', 'MaxMinS'):
                with self.subTest(n=n, mode=mode):
                    torch.manual_seed(42)
                    m = models()
                    a_params = [p for module in m[:3] for p in module.parameters()]
                    a = torch.optim.SGD(a_params, lr=.01)
                    b = torch.optim.SGD(m[3].parameters(), lr=.01)
                    loader = DataLoader(TensorDataset(torch.randn(2, n, 3), torch.tensor([0, 1])), batch_size=1)
                    budget = None if mode == 'AFS' else 5
                    loss, _ = dtfd_train_loop('cpu', m, loader, nn.CrossEntropyLoss(),
                                              [a,b], [None,None], 5, 5., mode, budget)
                    self.assertTrue(torch.isfinite(torch.tensor(loss)))
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for module in m for p in module.parameters()))
                    val, scores = dtfd_val_loop('cpu', 2, m, loader, nn.CrossEntropyLoss(),
                                               5, 5., mode, budget)
                    self.assertTrue(torch.isfinite(torch.tensor(val)))
                    self.assertIn('macro_auc', scores)


if __name__ == '__main__':
    unittest.main()
