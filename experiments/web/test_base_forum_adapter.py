import unittest

from base_forum_adapter import accepted_evaluation
from seeact_adapter import ModelUnavailable


class Bridge:
    def __init__(self, invalid=False):
        self.invalid = invalid

    def validate_completed_calls(self):
        if self.invalid:
            raise ModelUnavailable('Terminal inference failure')


class OfficialBaseScoreTests(unittest.TestCase):
    def test_native_executor_swallowed_error_is_not_a_scored_failure(self):
        with self.assertRaises(ModelUnavailable):
            accepted_evaluation([], Bridge())

    def test_terminal_inference_error_cannot_turn_into_valid_zero(self):
        with self.assertRaises(ModelUnavailable):
            accepted_evaluation([{'score': 0.0}], Bridge(invalid=True))

    def test_genuine_official_zero_and_fraction_are_preserved(self):
        for score in [0.0, 0.5, 1.0]:
            self.assertEqual(accepted_evaluation([{'score': score}], Bridge())['score'], score)

    def test_nonfinite_or_out_of_range_official_score_is_invalid(self):
        for score in [float('nan'), float('inf'), -1., 2.]:
            with self.assertRaises(ModelUnavailable):
                accepted_evaluation([{'score': score}], Bridge())


if __name__ == '__main__':
    unittest.main()
