import unittest

from src.agent.module.tooling.planner_tool_aware import ToolAwarePlanner
from src.agent.module.tooling.registry import ToolRegistry


class FakeModel:
    def __init__(self, response: str):
        self.response = response

    def predict(self, prompt: str) -> str:
        return self.response


class ToolAwarePlannerTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry.from_dict(
            {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search data",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                    {
                        "name": "answer",
                        "description": "Answer",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                ]
            }
        )

    def test_plan_success(self):
        response = """```json
        {
          "plan": [
            {
              "name": "Subtask1",
              "goal": "Collect facts",
              "dependencies": [],
              "allowed_tools": ["search"]
            },
            {
              "name": "Subtask2",
              "goal": "Write answer",
              "dependencies": ["Subtask1"],
              "allowed_tools": ["answer"]
            }
          ]
        }
        ```"""
        planner = ToolAwarePlanner(FakeModel(response), self.registry)

        plan_payload, valid, failed, handoff = planner.plan("dummy prompt", max_retry=1)

        self.assertTrue(valid)
        self.assertEqual(len(failed), 0)
        self.assertEqual(len(plan_payload["plan"]), 2)
        self.assertEqual(handoff[1]["dependencies"], ["Subtask1"])

    def test_plan_fails_validation(self):
        bad_response = """```json
        {"plan": [{"name": "Subtask1", "goal": "X", "dependencies": [], "allowed_tools": ["missing"]}]}
        ```"""
        planner = ToolAwarePlanner(FakeModel(bad_response), self.registry)

        plan_payload, valid, failed, handoff = planner.plan("dummy prompt", max_retry=1)

        self.assertFalse(valid)
        self.assertGreaterEqual(len(failed), 1)
        self.assertEqual(handoff, [])
        self.assertIn("errors", plan_payload)


if __name__ == "__main__":
    unittest.main()
