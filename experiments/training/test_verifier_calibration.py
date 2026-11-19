import unittest

import torch

from verifier_calibration_followup import choose_weight, fused_action
from forum_expert_iteration import causal_report


def report(pair, targeted, unrelated):
    return {'metrics': {k: {'numerator': n, 'denominator': 40}
            for k, n in [('PairCausalAcc', pair), ('TargetedAcc', targeted), ('UnrelatedAcc', unrelated)]}}


class CalibrationProtocolTests(unittest.TestCase):
    def test_primary_pair_metric_requires_unrelated_stability(self):
        rows = [{'id': 'one', 'size': [100, 100], 'variants': {
            'original': {'action': 0, 'bbox': [10, 10, 20, 20]},
            'target': {'action': 1}, 'unrelated': {'action': 0}}}]
        pred = {'one': {'original': {'action': 0, 'bbox': None},
                        'target': {'action': 1}, 'unrelated': {'action': 1}}}
        result = causal_report(rows, pred)['metrics']
        self.assertEqual(result['TargetPairAcc']['numerator'], 1)
        self.assertEqual(result['PairCausalAcc']['numerator'], 0)

    def test_original_weight_reproduces_float32_fusion(self):
        a, v = [-.12345678, -.78901234], [-1.23456, -.04321]
        expected = torch.tensor(a, dtype=torch.float32)
        for i in range(2):
            expected[i] += torch.tensor(v[i], dtype=torch.float32)
        self.assertEqual(fused_action(a, v, 1.), int(expected.argmax()))

    def test_zero_weight_keeps_action_decision(self):
        self.assertEqual(fused_action([-.1, -2.], [-10., 0.], 0.), 0)
        self.assertEqual(fused_action([-.1, -2.], [-10., 0.], 1.), 1)

    def test_validation_gate_rejects_pair_gain_with_stability_loss(self):
        reports = {'1.0': report(2, 8, 30), '0.0': report(7, 15, 10)}
        self.assertIsNone(choose_weight(reports, [0., 1.]))

    def test_gate_requires_strict_pair_improvement_and_uses_frozen_tie_rule(self):
        reports = {'1.0': report(2, 8, 30), '0.0': report(6, 20, 20),
                   '0.25': report(6, 20, 20), '0.5': report(6, 20, 20)}
        self.assertEqual(choose_weight(reports, [0., .25, .5, 1.]), .5)
        reports = {'1.0': report(2, 8, 30), '0.0': report(2, 20, 20)}
        self.assertIsNone(choose_weight(reports, [0., 1.]))


if __name__ == '__main__':
    unittest.main()
