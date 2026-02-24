import json
import unittest
from unittest.mock import patch

from src.agent.module.tooling.registry import ToolRegistry
from src.agent.module.tooling.runtime import ToolRuntime, ToolRuntimeError


class _MockResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class ToolRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry.from_dict(
            {
                "tools": [
                    {
                        "name": "duckduckgo_search",
                        "description": "DDG search",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                    {
                        "name": "calculator",
                        "description": "Calc",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                ]
            }
        )
        self.runtime = ToolRuntime(self.registry)

    @patch("src.agent.module.tooling.runtime.urlopen")
    def test_duckduckgo_search_parses_response(self, mock_urlopen):
        mock_urlopen.return_value = _MockResponse(
            {
                "Heading": "Python",
                "AbstractText": "Python is a programming language.",
                "AbstractURL": "https://example.com/python",
                "RelatedTopics": [
                    {"Text": "Python syntax - quick intro", "FirstURL": "https://example.com/syntax"}
                ],
            }
        )

        output = self.runtime.run(
            "duckduckgo_search",
            {"query": "python language", "max_results": 3},
            timeout_sec=5,
        )

        self.assertEqual(output["tool"], "duckduckgo_search")
        self.assertEqual(output["query"], "python language")
        self.assertGreaterEqual(output["result_count"], 1)
        self.assertIn("results", output)

    def test_calculator_executes_arithmetic(self):
        output = self.runtime.run("calculator", {"expression": "2 + 3 * 4"})
        self.assertEqual(output["value"], 14.0)

    def test_unknown_tool_raises(self):
        with self.assertRaises(ToolRuntimeError):
            self.runtime.run("unknown_tool", {"x": 1})


if __name__ == "__main__":
    unittest.main()
