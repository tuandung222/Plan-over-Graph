from __future__ import annotations

from typing import Any

from src.agent.module.tooling.registry import ToolRegistry
from src.agent.module.tooling.validator import validate_tool_aware_plan


def build_react_handoff(
    payload: dict[str, Any] | list[dict[str, Any]],
    registry: ToolRegistry,
) -> list[dict[str, Any]]:
    normalized = validate_tool_aware_plan(payload, registry)
    handoff = []

    for task in normalized["plan"]:
        tools = [registry.get_tool(name) for name in task["allowed_tools"]]
        handoff.append(
            {
                "task_name": task["name"],
                "goal": task["goal"],
                "dependencies": task["dependencies"],
                "inputs_from": task["inputs_from"],
                "toolbox": tools,
                "success_criteria": task["success_criteria"],
                "output_schema": task["output_schema"],
                "budget": task["budget"],
                "timeout_sec": task["timeout_sec"],
            }
        )

    return handoff
