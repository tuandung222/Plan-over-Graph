import unittest

from src.agent.module.tooling.registry import ToolRegistry
from src.agent.module.tooling.worker import ToolAwareWorker


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def run(self, tool_name, tool_input, timeout_sec=30):
        self.calls.append((tool_name, tool_input, timeout_sec))
        return {"tool": tool_name, "ok": True, "input": tool_input}


class ToolWorkerTests(unittest.TestCase):
    def setUp(self):
        registry = ToolRegistry.from_dict(
            {
                "tools": [
                    {
                        "name": "duckduckgo_search",
                        "description": "DDG search",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                    {
                        "name": "final_answer",
                        "description": "Final answer",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                ]
            }
        )
        self.runtime = _FakeRuntime()
        self.worker = ToolAwareWorker(registry, self.runtime)

    def test_execute_handoff_respects_dependencies(self):
        handoff = [
            {
                "task_name": "Subtask1",
                "goal": "Find latest Python release",
                "dependencies": [],
                "inputs_from": [],
                "toolbox": [{"name": "duckduckgo_search"}],
                "budget": {"max_calls": 2, "max_cost": 0.1},
                "timeout_sec": 15,
            },
            {
                "task_name": "Subtask2",
                "goal": "Compose final answer",
                "dependencies": ["Subtask1"],
                "inputs_from": ["Subtask1"],
                "toolbox": [{"name": "final_answer"}],
                "budget": {"max_calls": 1, "max_cost": 0.01},
                "timeout_sec": 10,
            },
        ]

        result = self.worker.execute_handoff(handoff)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["completed_tasks"], 2)
        self.assertEqual(len(self.runtime.calls), 2)
        self.assertEqual(self.runtime.calls[0][0], "duckduckgo_search")
        self.assertEqual(self.runtime.calls[1][0], "final_answer")


if __name__ == "__main__":
    unittest.main()
