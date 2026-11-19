import json
import subprocess
import unittest
from unittest.mock import patch

from prepare_forum_seed import owned_container, verify_source


class PreparationOwnershipTests(unittest.TestCase):
    def test_existing_live_service_cannot_be_claimed_for_cleanup(self):
        result = subprocess.CompletedProcess([], 0, json.dumps([{'Config': {'Labels': {}}}]), '')
        with patch('prepare_forum_seed.command', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'another run'):
                owned_container('forum', 'new-preparation')

    def test_declared_data_volumes_require_separate_snapshot_review(self):
        for source in [{'Config': {'Volumes': {'/data': {}}}},
                       {'Config': {}, 'Mounts': [{'Destination': '/data'}]}]:
            with self.assertRaisesRegex(RuntimeError, 'volume snapshot'):
                verify_source(source, {'Driver': 'overlay2'})

    def test_non_copy_on_write_storage_is_not_admitted(self):
        with self.assertRaisesRegex(RuntimeError, 'copy-on-write'):
            verify_source({'Config': {}}, {'Driver': 'vfs'})


if __name__ == '__main__':
    unittest.main()
