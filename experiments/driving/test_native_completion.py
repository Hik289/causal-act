import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import native_route_queue as nq


class CompletionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw, self.log, self.invalid = [self.root/name for name in
            ['results.json', 'evaluation.log', 'invalid.json']]
        self.route = 'RouteScenario_1_rep0'
        self.record = dict(route_id=self.route, status='Failed - Agent got blocked',
            scores=dict(score_composed=0.), infractions=dict(vehicle_blocked=['blocked']))
        self.actor = SimpleNamespace(pid=200, poll=Mock(return_value=None))
        self.stop = SimpleNamespace(wait=Mock(return_value=False), is_set=Mock(return_value=False))
        self.log.write_text('Game time = 60.05\nRegistering the route statistics\n')
        self.write_complete()
        self.invalid.write_text(json.dumps(dict(valid=False, score=None, error='RuntimeError',
            message='Foreign compute entered allocated GPU; own attempt stopped')))
        for path, ns in [(self.raw, 1000000000), (self.log, 1000000000), (self.invalid, 2000000000)]:
            os.utime(path, ns=(ns, ns))

    def write_complete(self, **changes):
        value=dict(entry_status='Finished', eligible=True,
                   _checkpoint=dict(progress=[1, 1], records=[self.record]))
        value.update(changes);self.raw.write_text(json.dumps(value))

    def pending(self):
        self.raw.write_text(json.dumps(dict(entry_status='Started', eligible=False,
            _checkpoint=dict(progress=[0, 1], records=[]))))

    def wait(self):
        return nq.wait_native_actor(self.actor, 100, 'GPU-own', self.raw, self.log,
                                    self.route, self.stop, 3600)

    def test_finished_zero_precedes_foreign_check_and_is_read_only(self):
        before=self.raw.read_bytes(),self.log.read_bytes(),self.invalid.read_bytes()
        with patch.object(nq, 'inventory') as inventory:
            result=self.wait()
        self.assertEqual(result['scores']['score_composed'], 0.)
        inventory.assert_not_called()
        self.assertEqual((self.raw.read_bytes(),self.log.read_bytes(),self.invalid.read_bytes()),before)

    def test_positive_completion_uses_the_same_rule(self):
        self.record.update(status='Completed',scores=dict(score_composed=100.))
        self.write_complete()
        self.assertEqual(self.wait()['scores']['score_composed'], 100.)

    def test_unfinished_foreign_compute_still_stops_episode(self):
        self.pending()
        with patch.object(nq, 'inventory', return_value=([], {'GPU-own': {300}})), \
             patch.object(nq, 'group_pids', side_effect=lambda pid:{pid}):
            with self.assertRaisesRegex(RuntimeError, 'Foreign compute'):
                self.wait()

    def test_completion_during_inventory_snapshot_is_retained(self):
        self.pending()
        def inventory():
            self.write_complete()
            return [], {'GPU-own': {300}}
        with patch.object(nq, 'inventory', side_effect=inventory), \
             patch.object(nq, 'group_pids', side_effect=lambda pid:{pid}):
            self.assertEqual(self.wait()['scores']['score_composed'], 0.)

    def test_ineligible_or_partial_commit_is_never_finished(self):
        for change in [dict(eligible=False),dict(entry_status='Started'),
                       dict(_checkpoint=dict(progress=[0,1],records=[self.record]))]:
            with self.subTest(change=change):
                self.write_complete(**change)
                self.assertIsNone(nq.finished_native_record(self.raw,self.log,self.route))
        self.raw.write_text('{unfinished JSON')
        self.assertIsNone(nq.finished_native_record(self.raw,self.log,self.route))

    def test_environment_and_wrong_route_records_remain_invalid(self):
        for status in ['Failed - Simulation crashed','Failed - Agent initialization error']:
            self.record['status']=status;self.write_complete()
            self.assertIsNone(nq.finished_native_record(self.raw,self.log,self.route))
        self.record['status']='Completed';self.write_complete()
        self.assertIsNone(nq.finished_native_record(self.raw,self.log,'different-route'))
        self.log.write_text('Traceback (most recent call last):\n')
        self.assertIsNone(nq.finished_native_record(self.raw,self.log,self.route))

    def test_cleanup_may_append_logs_but_must_not_replace_outcome(self):
        before=nq.finished_native_record(self.raw,self.log,self.route)
        after=copy.deepcopy(before);after['log_sha256']='cleanup-log-hash'
        self.assertTrue(nq.same_native_outcome(before,after))
        after['scores']['score_composed']=1.
        self.assertFalse(nq.same_native_outcome(before,after))

    def test_exited_actor_still_requires_strict_finished_record(self):
        self.actor.poll.return_value=0
        self.assertEqual(self.wait()['scores']['score_composed'],0.)
        self.pending();self.assertIsNone(self.wait())

    def test_controller_stop_never_admits_later_cleanup_output(self):
        self.stop.wait.return_value=True;self.stop.is_set.return_value=True
        with self.assertRaisesRegex(RuntimeError,'Controller stop requested'):
            self.wait()

    def test_unscored_stall_keeps_original_window(self):
        self.pending()
        with patch.object(nq, 'inventory', return_value=([], {'GPU-own': {100,200}})), \
             patch.object(nq, 'group_pids', side_effect=lambda pid:{pid}), \
             patch.object(nq.time, 'time', return_value=self.log.stat().st_mtime+3601):
            with self.assertRaisesRegex(RuntimeError,'frozen environment stall window'):
                self.wait()

    def test_earlier_finished_zero_qualifies_without_rewriting_invalid(self):
        before=self.invalid.read_bytes()
        result=nq.recoverable_finished_record(self.raw,self.log,self.invalid,self.route)
        self.assertEqual(result['scores']['score_composed'],0.)
        self.assertEqual(self.invalid.read_bytes(),before)

    def test_recovery_excludes_post_abort_equal_time_and_other_error(self):
        for ns in [2000000000,3000000000]:
            os.utime(self.raw,ns=(ns,ns))
            self.assertIsNone(nq.recoverable_finished_record(self.raw,self.log,self.invalid,self.route))
        os.utime(self.raw,ns=(1000000000,1000000000))
        value=json.loads(self.invalid.read_text());value['message']='Controller stop requested'
        self.invalid.write_text(json.dumps(value));os.utime(self.invalid,ns=(2000000000,2000000000))
        self.assertIsNone(nq.recoverable_finished_record(self.raw,self.log,self.invalid,self.route))

    def test_queue_commits_zero_once_before_post_score_resource_check(self):
        queue=object.__new__(nq.Queue)
        queue.root=self.root/'queue';queue.root.mkdir()
        queue.protocol=dict(attempts_per_route=3,ports=[2010,8010],carla_launcher='CarlaUE4.sh',
            carla_root='/unused',vulkan_identity_tool='identity',evaluator='evaluator.py',
            run_id='fixture',environment_stall_seconds=3600,invalid_backoff_seconds=120)
        queue.protocol_sha='frozen';queue.repair=None;queue.state={};queue.records={}
        queue.stop=Mock();queue.stop.wait.return_value=False;queue.stop.is_set.return_value=False
        queue.verify=Mock();queue.save=Mock()
        queue.free_gpu=Mock(return_value=dict(index=0,uuid='GPU-0011',free=48000))
        sim=SimpleNamespace(pid=100,poll=Mock(return_value=None))
        actor=SimpleNamespace(pid=200,poll=Mock(return_value=None))
        def start(command,**kwargs):
            if command[0]=='CarlaUE4.sh':return sim
            case=queue.root/'cases/vad/1/attempt1'
            (case/'results.json').write_text(json.dumps(dict(entry_status='Finished',eligible=True,
                _checkpoint=dict(progress=[1,1],records=[self.record]))))
            (case/'evaluation.log').write_text('Final scored zero; native actor remains in cleanup\n')
            return actor
        spec=dict(environment={},python='python',entrypoint='agent.py',config='config',checkpoint='weights',cwd='/unused')
        devices=json.dumps([dict(vulkan_index=0,vendor_id=0x10de,device_uuid='0011')])
        with patch.object(nq.subprocess,'check_output',return_value=devices), \
             patch.object(nq.subprocess,'Popen',side_effect=start), \
             patch.object(nq,'rendered_gpu_uuids',return_value={'GPU-0011'}), \
             patch.object(nq,'group_pids',side_effect=lambda pid:{pid}), \
             patch.object(nq,'inventory',return_value=([],{'GPU-0011':{100}})) as inventory, \
             patch.object(nq,'clean_group') as cleanup:
            queue.case('vad',spec,ET.fromstring('<route id="1"/>'))
        self.assertEqual(queue.records['vad/1']['scores']['score_composed'],0.)
        self.assertEqual(inventory.call_count,1)  # Boot check only, no post-score rejection.
        self.assertEqual([call.args[0].pid for call in cleanup.call_args_list],[200,100])
        self.assertFalse((queue.root/'cases/vad/1/attempt2').exists())


if __name__=='__main__':
    unittest.main()
