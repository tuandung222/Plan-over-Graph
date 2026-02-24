import json
import unittest
from unittest.mock import patch

from src.agent.module.tooling.registry import ToolRegistry
from src.agent.module.tooling.runtime import ToolRuntime, ToolRuntimeError


class _MockResponse:
    def __init__(self, payload: dict | str):
        self._payload = payload
        self.headers = {"Content-Type": "application/json"}

    def read(self):
        if isinstance(self._payload, str):
            return self._payload.encode("utf-8")
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
                    {
                        "name": "wikipedia_search",
                        "description": "Wikipedia search",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                    {
                        "name": "arxiv_search",
                        "description": "arXiv search",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                    {
                        "name": "fetch_url",
                        "description": "Fetch URL",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                    {
                        "name": "current_datetime",
                        "description": "Current time",
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

    @patch("src.agent.module.tooling.runtime.urlopen")
    def test_wikipedia_search_parses_response(self, mock_urlopen):
        mock_urlopen.return_value = _MockResponse(
            [
                "python",
                ["Python (programming language)"],
                ["General-purpose programming language"],
                ["https://en.wikipedia.org/wiki/Python_(programming_language)"],
            ]
        )

        output = self.runtime.run(
            "wikipedia_search",
            {"query": "python", "max_results": 2},
            timeout_sec=5,
        )

        self.assertEqual(output["tool"], "wikipedia_search")
        self.assertEqual(output["result_count"], 1)
        self.assertEqual(output["results"][0]["source"], "wikipedia")

    @patch("src.agent.module.tooling.runtime.urlopen")
    def test_arxiv_search_parses_response(self, mock_urlopen):
        atom = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'>
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <published>2024-01-01T00:00:00Z</published>
    <title>Test Paper</title>
    <summary>This is a summary.</summary>
  </entry>
</feed>
"""
        mock_urlopen.return_value = _MockResponse(atom)

        output = self.runtime.run("arxiv_search", {"query": "test"}, timeout_sec=5)

        self.assertEqual(output["tool"], "arxiv_search")
        self.assertEqual(output["result_count"], 1)
        self.assertEqual(output["results"][0]["title"], "Test Paper")

    @patch("src.agent.module.tooling.runtime.urlopen")
    def test_fetch_url_extracts_html_text(self, mock_urlopen):
        html = "<html><head><title>Demo Page</title></head><body><h1>Hello</h1><p>World</p></body></html>"
        response = _MockResponse(html)
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_urlopen.return_value = response

        output = self.runtime.run("fetch_url", {"url": "https://example.com", "max_chars": 1000}, timeout_sec=5)

        self.assertEqual(output["tool"], "fetch_url")
        self.assertEqual(output["title"], "Demo Page")
        self.assertIn("Hello", output["text"])

    def test_current_datetime_returns_utc_fields(self):
        output = self.runtime.run("current_datetime", {}, timeout_sec=5)
        self.assertEqual(output["tool"], "current_datetime")
        self.assertIn("utc_iso", output)
        self.assertIn("unix_ts", output)

    def test_calculator_executes_arithmetic(self):
        output = self.runtime.run("calculator", {"expression": "2 + 3 * 4"})
        self.assertEqual(output["value"], 14.0)

    def test_unknown_tool_raises(self):
        with self.assertRaises(ToolRuntimeError):
            self.runtime.run("unknown_tool", {"x": 1})


if __name__ == "__main__":
    unittest.main()
