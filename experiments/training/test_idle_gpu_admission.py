import unittest

from balanced_followup_idle_controller import admit


class IdleGPUAdmissionTests(unittest.TestCase):
    def test_unrelated_compute_owner_blocks_an_otherwise_roomy_card(self):
        self.assertFalse(admit({}, {}, 0, 1, 45000, 20 * 2**30, active_gpu_pids=[123]))

    def test_inference_on_original_gpu_is_not_mistaken_for_idle_memory(self):
        self.assertFalse(admit({'full': {'status': 'running', 'gpu': 0}}, {}, 0, 0, 41000, 20 * 2**30))

    def test_original_pending_work_keeps_priority(self):
        self.assertFalse(admit({'last': {'status': 'queued'}}, {}, 0, 1, 48000, 20 * 2**30))

    def test_global_worker_limit_includes_side_experiment(self):
        old = {'a': {'status': 'running', 'gpu': 0}, 'b': {'status': 'running', 'gpu': 2}}
        side = {'calibration': {'status': 'running', 'gpu': 3}}
        self.assertTrue(admit(old, side, 0, 1, 48000, 20 * 2**30))
        self.assertFalse(admit(old, side, 1, 1, 48000, 20 * 2**30))

    def test_memory_and_disk_floors(self):
        self.assertFalse(admit({}, {}, 0, 1, 35999, 20 * 2**30))
        self.assertFalse(admit({}, {}, 0, 1, 48000, 9 * 2**30))


if __name__ == '__main__':
    unittest.main()
