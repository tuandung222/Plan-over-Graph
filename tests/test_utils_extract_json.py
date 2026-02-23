import unittest

from src.utils.utils import extract_json


class ExtractJsonTests(unittest.TestCase):
    def test_parses_fenced_json_list(self):
        text = """```json
        [
          {"name": "Subtask1", "dependencies": []}
        ]
        ```"""

        parsed = extract_json(text)

        self.assertIsInstance(parsed, list)
        self.assertEqual(parsed[0]["name"], "Subtask1")

    def test_parses_plain_json_object(self):
        text = '{"plan": [{"name": "Subtask1", "dependencies": []}]}'

        parsed = extract_json(text)

        self.assertIsInstance(parsed, dict)
        self.assertIn("plan", parsed)
        self.assertEqual(parsed["plan"][0]["name"], "Subtask1")

    def test_raises_on_invalid_json(self):
        with self.assertRaises(ValueError):
            extract_json("not a json payload")


if __name__ == "__main__":
    unittest.main()
