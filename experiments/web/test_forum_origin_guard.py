import unittest
from forum_origin_guard import destination, declared_origin


class ForumOriginTests(unittest.TestCase):
    def test_official_prompt_aliases_map_to_the_same_owned_forum(self):
        for url in ['http://reddit.com/f/a', 'https://www.reddit.com/f/a']:
            source = declared_origin(url, 'http://127.0.0.1:9999')
            self.assertEqual(destination(url, source, 'http://127.0.0.1:9962'), 'http://127.0.0.1:9962/f/a')
        with self.assertRaises(ValueError):
            declared_origin('https://reddit.com.unrelated.example/f/a', 'http://127.0.0.1:9999')

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
