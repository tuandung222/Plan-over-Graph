import unittest

from src.agent.module.tooling.registry import ToolRegistry
from src.agent.module.tooling.validator import PlanValidationError, validate_tool_aware_plan


class ToolPlanValidatorTests(unittest.TestCase):
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
                        "description": "Final answer",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                ]
            }
        )

    def test_valid_plan_passes(self):
        payload = {
            "plan": [
                {
                    "name": "Subtask1",
                    "goal": "Collect facts",
                    "dependencies": [],
                    "allowed_tools": ["search"],
                },
                {
                    "name": "Subtask2",
                    "goal": "Write answer",
                    "dependencies": ["Subtask1"],
                    "allowed_tools": ["answer"],
                },
            ]
        }

        normalized = validate_tool_aware_plan(payload, self.registry)

        self.assertEqual(len(normalized["plan"]), 2)
        self.assertEqual(normalized["plan"][1]["dependencies"], ["Subtask1"])

    def test_unknown_tool_raises(self):
        payload = {
            "plan": [
                {
                    "name": "Subtask1",
                    "goal": "Collect facts",
                    "dependencies": [],
                    "allowed_tools": ["non_existing_tool"],
                }
            ]
        }

        with self.assertRaises(PlanValidationError):
            validate_tool_aware_plan(payload, self.registry)

    def test_cycle_raises(self):
        payload = {
            "plan": [
                {
                    "name": "A",
                    "goal": "Task A",
                    "dependencies": ["B"],
                    "allowed_tools": ["search"],
                },
                {
                    "name": "B",
                    "goal": "Task B",
                    "dependencies": ["A"],
                    "allowed_tools": ["answer"],
                },
            ]
        }

        with self.assertRaises(PlanValidationError):
            validate_tool_aware_plan(payload, self.registry)


if __name__ == "__main__":
    unittest.main()
