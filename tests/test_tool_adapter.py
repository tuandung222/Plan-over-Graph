import unittest

from src.agent.module.tooling.adapter import build_react_handoff
from src.agent.module.tooling.registry import ToolRegistry


class ToolAdapterTests(unittest.TestCase):
    def test_build_react_handoff_includes_tool_specs(self):
        registry = ToolRegistry.from_dict(
            {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search data",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    }
                ]
            }
        )
        payload = {
            "plan": [
                {
                    "name": "Subtask1",
                    "goal": "Collect facts",
                    "dependencies": [],
                    "allowed_tools": ["search"],
                }
            ]
        }

        handoff = build_react_handoff(payload, registry)

        self.assertEqual(len(handoff), 1)
        self.assertEqual(handoff[0]["task_name"], "Subtask1")
        self.assertEqual(handoff[0]["toolbox"][0]["name"], "search")


if __name__ == "__main__":
    unittest.main()
