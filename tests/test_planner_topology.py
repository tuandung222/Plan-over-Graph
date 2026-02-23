import unittest

from src.agent.module.planner import ParallelPlanner
from src.agent.module.subtask import SubTTNode


class PlannerTopologyTests(unittest.TestCase):
    def setUp(self):
        self.planner = ParallelPlanner(model=None, env=None)

    def test_topological_sort_orders_dependencies_first(self):
        tasks = [
            SubTTNode(
                {
                    "name": "Subtask1",
                    "source": ["N1"],
                    "target": ["N2"],
                    "dependencies": [],
                }
            ),
            SubTTNode(
                {
                    "name": "Subtask2",
                    "source": ["N1"],
                    "target": ["N3"],
                    "dependencies": [],
                }
            ),
            SubTTNode(
                {
                    "name": "Subtask3",
                    "source": ["N2", "N3"],
                    "target": ["N4"],
                    "dependencies": ["Subtask1", "Subtask2"],
                }
            ),
        ]

        sorted_tasks = self.planner.topological_sort(tasks)

        self.assertIsNotNone(sorted_tasks)
        names = [task.name for task in sorted_tasks]
        self.assertEqual(names[-1], "Subtask3")
        self.assertLess(names.index("Subtask1"), names.index("Subtask3"))
        self.assertLess(names.index("Subtask2"), names.index("Subtask3"))

    def test_topological_sort_returns_none_on_cycle(self):
        tasks = [
            SubTTNode(
                {
                    "name": "SubtaskA",
                    "source": ["N1"],
                    "target": ["N2"],
                    "dependencies": ["SubtaskB"],
                }
            ),
            SubTTNode(
                {
                    "name": "SubtaskB",
                    "source": ["N2"],
                    "target": ["N3"],
                    "dependencies": ["SubtaskA"],
                }
            ),
        ]

        sorted_tasks = self.planner.topological_sort(tasks)

        self.assertIsNone(sorted_tasks)


if __name__ == "__main__":
    unittest.main()
