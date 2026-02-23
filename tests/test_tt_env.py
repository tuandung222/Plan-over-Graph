import unittest

from src.agent.module.env.tt_env import TTEnv
from src.agent.module.subtask import SubTTNode


class TTEnvTests(unittest.TestCase):
    def test_valid_subtask_by_source_target_sets_time_and_cost(self):
        env = TTEnv(
            {
                "rules": [
                    {"source": ["N1"], "target": ["N2"], "time": 3, "cost": 2}
                ],
                "initial_source": ["N1"],
                "target": "N2",
            }
        )
        node = SubTTNode(
            {
                "name": "Subtask1",
                "source": ["N1"],
                "target": ["N2"],
                "dependencies": [],
            }
        )

        is_valid = env.is_valid_sub_node(node)

        self.assertTrue(is_valid)
        self.assertEqual(node.time, 3)
        self.assertEqual(node.cost, 2)

    def test_commit_with_rule_index_infers_source_target_and_accumulates_metrics(self):
        env = TTEnv(
            {
                "rules": [
                    {"source": ["N1"], "target": ["N2"], "time": 3, "cost": 1},
                    {"source": ["N1"], "target": ["N3"], "time": 5, "cost": 2},
                    {"source": ["N2", "N3"], "target": ["N4"], "time": 7, "cost": 4},
                ],
                "initial_source": ["N1"],
                "target": "N4",
            }
        )

        step1 = SubTTNode(
            {
                "name": "Subtask1",
                "perform_rule_indx": 0,
                "dependencies": [],
            }
        )
        step2 = SubTTNode(
            {
                "name": "Subtask2",
                "perform_rule_indx": 1,
                "dependencies": [],
            }
        )
        step3 = SubTTNode(
            {
                "name": "Subtask3",
                "perform_rule_indx": 2,
                "dependencies": ["Subtask1", "Subtask2"],
            }
        )

        self.assertEqual(env.commit(step1), 3)
        self.assertEqual(env.commit(step2), 5)
        self.assertEqual(env.commit(step3), 12)

        final_time, total_cost = env.get_final_result()
        self.assertEqual(final_time, 12)
        self.assertEqual(total_cost, 7)

    def test_commit_raises_when_source_not_available(self):
        env = TTEnv(
            {
                "rules": [
                    {"source": ["N2"], "target": ["N3"], "time": 1, "cost": 1}
                ],
                "initial_source": ["N1"],
                "target": "N3",
            }
        )
        node = SubTTNode(
            {
                "name": "SubtaskX",
                "source": ["N2"],
                "target": ["N3"],
                "dependencies": [],
            }
        )

        with self.assertRaises(ValueError):
            env.commit(node)


if __name__ == "__main__":
    unittest.main()
