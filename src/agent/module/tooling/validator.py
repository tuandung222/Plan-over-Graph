from __future__ import annotations

from collections import deque
from typing import Any

from src.agent.module.tooling.registry import ToolRegistry


class PlanValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def normalize_tool_aware_plan(payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(payload, dict):
        if "plan" not in payload:
            raise PlanValidationError(["payload object must contain key 'plan'"])
        plan = payload["plan"]
    elif isinstance(payload, list):
        plan = payload
    else:
        raise PlanValidationError(["payload must be a list or an object containing 'plan'"])

    if not isinstance(plan, list):
        raise PlanValidationError(["plan must be a list"])

    normalized_plan = []
    for idx, task in enumerate(plan):
        if not isinstance(task, dict):
            raise PlanValidationError([f"plan[{idx}] must be an object"])

        name = task.get("name")
        goal = task.get("goal")
        dependencies = task.get("dependencies", [])
        allowed_tools = task.get("allowed_tools", [])

        if not isinstance(name, str) or not name.strip():
            raise PlanValidationError([f"plan[{idx}].name must be a non-empty string"])
        if not isinstance(goal, str) or not goal.strip():
            raise PlanValidationError([f"plan[{idx}].goal must be a non-empty string"])
        if not isinstance(dependencies, list) or not all(isinstance(d, str) for d in dependencies):
            raise PlanValidationError([f"plan[{idx}].dependencies must be a list of strings"])
        if not isinstance(allowed_tools, list) or not all(isinstance(t, str) for t in allowed_tools):
            raise PlanValidationError([f"plan[{idx}].allowed_tools must be a list of strings"])

        output_schema = task.get("output_schema", {"type": "object"})
        if not isinstance(output_schema, dict):
            raise PlanValidationError([f"plan[{idx}].output_schema must be an object"])

        budget = task.get("budget", {"max_calls": 0, "max_cost": 0.0})
        if not isinstance(budget, dict):
            raise PlanValidationError([f"plan[{idx}].budget must be an object"])

        timeout_sec = task.get("timeout_sec", 60)
        if not isinstance(timeout_sec, int) or timeout_sec <= 0:
            raise PlanValidationError([f"plan[{idx}].timeout_sec must be a positive integer"])

        success_criteria = task.get("success_criteria", "")
        if not isinstance(success_criteria, str):
            raise PlanValidationError([f"plan[{idx}].success_criteria must be a string"])

        inputs_from = task.get("inputs_from", [])
        if not isinstance(inputs_from, list) or not all(isinstance(x, str) for x in inputs_from):
            raise PlanValidationError([f"plan[{idx}].inputs_from must be a list of strings"])

        normalized_plan.append(
            {
                "name": name.strip(),
                "goal": goal.strip(),
                "dependencies": _dedupe_keep_order(dependencies),
                "allowed_tools": _dedupe_keep_order(allowed_tools),
                "success_criteria": success_criteria.strip(),
                "output_schema": output_schema,
                "budget": budget,
                "timeout_sec": timeout_sec,
                "inputs_from": _dedupe_keep_order(inputs_from),
            }
        )

    return {"plan": normalized_plan}


def validate_tool_aware_plan(
    payload: dict[str, Any] | list[dict[str, Any]],
    registry: ToolRegistry,
) -> dict[str, Any]:
    normalized = normalize_tool_aware_plan(payload)
    plan = normalized["plan"]

    errors: list[str] = []
    name_set: set[str] = set()

    for task in plan:
        if task["name"] in name_set:
            errors.append(f"duplicate task name: {task['name']}")
        else:
            name_set.add(task["name"])

    for task in plan:
        task_name = task["name"]

        if not task["allowed_tools"]:
            errors.append(f"task '{task_name}' must declare at least one allowed tool")

        for tool_name in task["allowed_tools"]:
            if not registry.has_tool(tool_name):
                errors.append(f"task '{task_name}' references unknown tool: {tool_name}")

        for dep in task["dependencies"]:
            if dep not in name_set:
                errors.append(f"task '{task_name}' has unknown dependency: {dep}")
            if dep == task_name:
                errors.append(f"task '{task_name}' cannot depend on itself")

        for dep in task["inputs_from"]:
            if dep not in name_set:
                errors.append(f"task '{task_name}' references unknown inputs_from task: {dep}")

    if _has_cycle(plan):
        errors.append("plan dependencies contain a cycle")

    if errors:
        raise PlanValidationError(errors)

    return normalized


def _has_cycle(plan: list[dict[str, Any]]) -> bool:
    in_degree: dict[str, int] = {task["name"]: 0 for task in plan}
    adjacency: dict[str, list[str]] = {task["name"]: [] for task in plan}

    for task in plan:
        for dep in task["dependencies"]:
            if dep in adjacency:
                adjacency[dep].append(task["name"])
                in_degree[task["name"]] += 1

    queue = deque([name for name, degree in in_degree.items() if degree == 0])
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited != len(plan)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
