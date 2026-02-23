import unittest

from template import abstask_plan, specific_task_plan, extract_rules


class TemplateContractTests(unittest.TestCase):
    def test_abstask_template_contains_task_placeholder_and_dependencies(self):
        instruction = abstask_plan.instruction
        self.assertIn("{task}", instruction)
        self.assertIn("dependencies", instruction)

    def test_specific_task_template_mentions_json_plan_fields(self):
        instruction = specific_task_plan.instruction
        for field in ["name", "source", "target", "dependencies"]:
            self.assertIn(field, instruction)

    def test_extract_rules_template_requires_expected_keys(self):
        instruction = extract_rules.instruction
        for key in ["rules", "initial_source", "target"]:
            self.assertIn(key, instruction)


if __name__ == "__main__":
    unittest.main()
