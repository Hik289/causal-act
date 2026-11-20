import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from forum_pair_queue import runtime_environment


class EndpointTests(unittest.TestCase):
    def fixture(self, folder):
        endpoint = 'http://127.0.0.1:18841/v1/chat/completions'
        receipt = Path(folder)/'receipt.json'
        receipt.write_text(json.dumps(dict(endpoint=endpoint, model='ui-tars-1.5-7b',
            same_input_and_generation_contract=True, transport_checks_passed=True, benchmark_scores=0)))
        protocol = dict(methods=['uitars'], environment={'CAUSALACT_UITARS_ENDPOINT':'original', 'OTHER':'preserved'})
        repair = dict(environment_repair_only=True, uitars_endpoint=endpoint,
            uitars_service_receipt=str(receipt),
            uitars_service_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
        return protocol, repair, receipt

    def test_only_local_endpoint_changes_without_mutating_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            protocol, repair, _ = self.fixture(folder)
            before = json.dumps(protocol)
            env = runtime_environment(protocol, repair)
            self.assertEqual(env['CAUSALACT_UITARS_ENDPOINT'], repair['uitars_endpoint'])
            self.assertEqual(env['OTHER'], 'preserved')
            self.assertEqual(json.dumps(protocol), before)

    def test_other_repairs_preserve_original_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            protocol, _, _ = self.fixture(folder)
            self.assertEqual(runtime_environment(protocol, None), protocol['environment'])
            self.assertEqual(runtime_environment(protocol, {'repair_id':'existing'}), protocol['environment'])

    def test_rejects_external_credentials_and_wrong_routes(self):
        for endpoint in ['https://127.0.0.1:18841/v1/chat/completions',
                         'http://example.com:18841/v1/chat/completions',
                         'http://user@127.0.0.1:18841/v1/chat/completions',
                         'http://127.0.0.1:80/v1/chat/completions',
                         'http://127.0.0.1:18841/other',
                         'http://127.0.0.1:18841/v1/chat/completions?change=1']:
            with self.subTest(endpoint=endpoint), tempfile.TemporaryDirectory() as folder:
                protocol, repair, _ = self.fixture(folder)
                repair['uitars_endpoint'] = endpoint
                with self.assertRaises(RuntimeError):runtime_environment(protocol, repair)

    def test_rejects_another_method_or_non_environment_change(self):
        with tempfile.TemporaryDirectory() as folder:
            protocol, repair, _ = self.fixture(folder)
            protocol['methods'] = ['base','seeact']
            with self.assertRaises(RuntimeError):runtime_environment(protocol, repair)
            protocol['methods'] = ['uitars']
            repair['environment_repair_only'] = False
            with self.assertRaises(RuntimeError):runtime_environment(protocol, repair)

    def test_rejects_modified_or_incomplete_service_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            protocol, repair, receipt = self.fixture(folder)
            value = json.loads(receipt.read_text());value['transport_checks_passed'] = False
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'changed'):runtime_environment(protocol, repair)
            repair['uitars_service_receipt_sha256'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError,'unverified'):runtime_environment(protocol, repair)


if __name__ == '__main__':
    unittest.main()
