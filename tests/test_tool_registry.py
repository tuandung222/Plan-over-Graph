import json
import tempfile
import unittest

from src.agent.module.tooling.registry import ToolRegistry, ToolRegistryError


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "tools": [
                {
                    "name": "search",
                    "description": "Search documents",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
                {
                    "name": "answer",
                    "description": "Return final answer",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
            ]
        }

    def test_load_from_dict(self):
        registry = ToolRegistry.from_dict(self.payload)
        self.assertTrue(registry.has_tool("search"))
        self.assertEqual(sorted(registry.list_tool_names()), ["answer", "search"])

    def test_load_from_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(self.payload, f)
            path = f.name

        registry = ToolRegistry.from_file(path)
        self.assertTrue(registry.has_tool("answer"))

    def test_duplicate_tool_names_raise(self):
        dup_payload = {
            "tools": [
                {
                    "name": "search",
                    "description": "A",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
                {
                    "name": "search",
                    "description": "B",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
            ]
        }
        with self.assertRaises(ToolRegistryError):
            ToolRegistry.from_dict(dup_payload)


if __name__ == "__main__":
    unittest.main()
