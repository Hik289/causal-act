"""The evaluator must wait for positive physical GPU identification."""
import collections
import unittest
from unittest.mock import Mock, patch

import native_route_queue as queue


class ClockStop:
    def __init__(self, cancel_on_wait=False):
        self.now = 0
        self.stopped = False
        self.cancel_on_wait = cancel_on_wait

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.now += seconds
        self.stopped = self.cancel_on_wait
        return self.stopped


class BootDeviceTests(unittest.TestCase):
    def check(self, devices, busy=None, timeout=10, stopped=False, exit_code=None):
        clock = ClockStop(cancel_on_wait=stopped)
        simulator = Mock(pid=101)
        simulator.poll.return_value = exit_code
        occupancy = collections.defaultdict(set, busy or {})
        with patch.object(queue.time, 'monotonic', side_effect=lambda: clock.now), \
             patch.object(queue, 'group_pids', return_value={101, 102}), \
             patch.object(queue, 'rendered_gpu_uuids', side_effect=devices) as probe, \
             patch.object(queue, 'inventory', return_value=([], occupancy)):
            result = queue.wait_simulator_device(simulator, 'GPU-selected', clock, timeout)
        return result, clock, probe

    def test_delayed_registration_requires_positive_match(self):
        result, clock, probe = self.check([set(), set(), {'GPU-selected'}])
        self.assertEqual(result, {'GPU-selected'})
        self.assertEqual(clock.now, 10)
        self.assertEqual(probe.call_count, 3)

    def test_immediate_match_needs_no_additional_delay(self):
        result, clock, probe = self.check([{'GPU-selected'}])
        self.assertEqual(result, {'GPU-selected'})
        self.assertEqual(clock.now, 0)
        self.assertEqual(probe.call_count, 1)

    def test_missing_device_fails_closed_at_deadline(self):
        with self.assertRaisesRegex(queue.DeviceIdentityError, 'boot deadline'):
            self.check([set(), set(), set()])

    def test_wrong_or_additional_device_is_not_a_startup_delay(self):
        for devices in [{'GPU-other'}, {'GPU-selected', 'GPU-other'}]:
            with self.subTest(devices=devices):
                with self.assertRaisesRegex(queue.DeviceIdentityError, 'different physical GPU'):
                    self.check([devices])

    def test_foreign_compute_blocks_both_pending_and_matched_device(self):
        for devices in [set(), {'GPU-selected'}]:
            with self.subTest(devices=devices):
                with self.assertRaisesRegex(RuntimeError, 'another compute job'):
                    self.check([devices], busy={'GPU-selected': {102, 999}})

    def test_owned_simulator_processes_do_not_count_as_foreign(self):
        result, _, _ = self.check([{'GPU-selected'}], busy={'GPU-selected': {101, 102}})
        self.assertEqual(result, {'GPU-selected'})

    def test_shutdown_during_wait_never_admits_unverified_device(self):
        with self.assertRaisesRegex(RuntimeError, 'Controller stop requested'):
            self.check([set()], stopped=True)

    def test_exited_simulator_is_not_retried_inside_boot_wait(self):
        with self.assertRaisesRegex(RuntimeError, 'Simulator exited'):
            self.check([], exit_code=1)


if __name__ == '__main__':
    unittest.main()
