import unittest
from forum_origin_guard import destination


class ForumOriginTests(unittest.TestCase):
    def test_environment_redirect_retains_path_query_and_encoded_values(self):
        self.assertEqual(destination('http://127.0.0.1:9999/f/a?q=x%2Fy#z',
                                    'http://127.0.0.1:9999', 'http://127.0.0.1:9962'),
                         'http://127.0.0.1:9962/f/a?q=x%2Fy#z')

    def test_unrelated_origin_and_redirect_loops_are_rejected(self):
        for url, current in [('http://127.0.0.1:99990/a', 'http://127.0.0.1:9962'),
                             ('http://127.0.0.1:9999/a', 'http://127.0.0.1:9999')]:
            with self.assertRaises(ValueError):
                destination(url, 'http://127.0.0.1:9999', current)


if __name__ == '__main__':
    unittest.main()
