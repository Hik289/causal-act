import unittest

import torch
import torch.nn.functional as F

from balanced_followup_worker import weighted_ce, weighted_loss
from forum_expert_iteration import a_loss


class FixedCategory:
    def __init__(self, category):
        self.category = category

    def choice(self, values):
        return self.category


class SmallNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([[.3, -.4, .1, .2, .3, .4]]))

    def forward(self, role, row, category='original'):
        return self.scores + (.1 if category == 'target' else 0.)


class BalancedFollowupTests(unittest.TestCase):
    def test_weight_does_not_cancel_for_a_single_example(self):
        x = torch.tensor([[0., 1.]], requires_grad=True)
        y = torch.tensor([1])
        baseline = F.cross_entropy(x, y)
        weighted = weighted_ce(x, y, [.6, 3.])
        self.assertTrue(torch.allclose(weighted, 3 * baseline))
        weighted.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_realized_training_prior_balances_weighted_mass(self):
        counts = [1587, 333]
        weights = [sum(counts) / (2 * n) for n in counts]
        self.assertAlmostEqual(counts[0] * weights[0], counts[1] * weights[1])
        self.assertAlmostEqual(sum(n * w for n, w in zip(counts, weights)) / sum(counts), 1.)

    def test_unit_weights_preserve_original_losses_for_all_edits(self):
        row = {'size': [100, 100], 'tau': {'dx': -.01, 'dy': .02}, 'variants': {
            'original': {'bbox': [10, 10, 20, 20], 'action': 0},
            'target': {'bbox': [10, 10, 20, 20], 'action': 1},
            'unrelated': {'bbox': [10, 10, 20, 20], 'action': 0},
            'transform': {'bbox': [9, 12, 19, 22], 'action': 0}}}
        categories = ['target', 'unrelated', 'transform']
        net = SmallNet()
        for relations in [False, True]:
            variant = {'relations': relations, 'class_weights': [1., 1.]}
            for category in categories:
                with self.subTest(relations=relations, category=category):
                    old = a_loss(net, row, categories, variant, FixedCategory(category))
                    new = weighted_loss(net, row, categories, variant, FixedCategory(category))
                    self.assertTrue(torch.allclose(old, new))


if __name__ == '__main__':
    unittest.main()
