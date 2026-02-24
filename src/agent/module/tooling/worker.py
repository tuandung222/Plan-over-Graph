from __future__ import annotations

from typing import Any
import re

from src.agent.module.tooling.runtime import ToolRuntime, ToolRuntimeError
from src.agent.module.tooling.registry import ToolRegistry


class ToolAwareWorker:
    """Execute tool-aware handoff tasks with dependency control."""

    def __init__(self, registry: ToolRegistry, runtime: ToolRuntime):
        self.registry = registry
        self.runtime = runtime

    def execute_handoff(self, handoff: list[dict[str, Any]]) -> dict[str, Any]:
        pending = {task["task_name"]: dict(task) for task in handoff}
        outputs: dict[str, Any] = {}
        traces: list[dict[str, Any]] = []

        while pending:
            ready = [
                name
                for name, task in pending.items()
                if all(dep in outputs for dep in task.get("dependencies", []))
            ]
            if not ready:
                raise ToolRuntimeError("cannot execute handoff: unresolved dependencies or dependency cycle")

            for task_name in ready:
                task = pending.pop(task_name)
                context = {dep: outputs[dep] for dep in task.get("inputs_from", []) if dep in outputs}
                task_result = self._run_task(task, context)
                outputs[task_name] = task_result
                traces.append(
                    {
                        "task_name": task_name,
                        "tool": task_result.get("tool"),
                        "status": "ok",
                    }
                )

        return {
            "status": "success",
            "completed_tasks": len(outputs),
            "outputs": outputs,
            "traces": traces,
        }

    def _run_task(self, task: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        toolbox = task.get("toolbox", [])
        if not toolbox:
            raise ToolRuntimeError(f"task '{task['task_name']}' has empty toolbox")

        errors: list[str] = []
        timeout_sec = int(task.get("timeout_sec", 30))

        for tool in toolbox:
            tool_name = tool.get("name")
            tool_input = self._build_tool_input(tool_name, task, context)
            try:
                output = self.runtime.run(tool_name, tool_input, timeout_sec=timeout_sec)
                return {
                    "tool": tool_name,
                    "input": tool_input,
                    "output": output,
                }
            except Exception as exc:
                errors.append(f"{tool_name}: {exc}")

        raise ToolRuntimeError(
            f"all tools failed for task '{task['task_name']}': " + "; ".join(errors)
        )

    def _build_tool_input(self, tool_name: str, task: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        goal = task.get("goal", "")
        budget = task.get("budget", {})
        max_calls = budget.get("max_calls", 5)
        if not isinstance(max_calls, int) or max_calls <= 0:
            max_calls = 5

        if tool_name in ("duckduckgo_search", "web_search"):
            context_hint = _context_to_text(context)
            query = goal if not context_hint else f"{goal}. Context: {context_hint}"
            return {
                "query": query.strip(),
                "max_results": min(max_calls, 10),
            }

        if tool_name in ("wikipedia_search", "arxiv_search"):
            context_hint = _context_to_text(context)
            query = goal if not context_hint else f"{goal}. Context: {context_hint}"
            return {
                "query": query.strip(),
                "max_results": min(max_calls, 10),
            }

        if tool_name == "fetch_url":
            url = _extract_first_url_from_context(context)
            if not url:
                url = _extract_first_url_from_text(goal)
            if not url:
                raise ToolRuntimeError(
                    f"task '{task['task_name']}' needs fetch_url but no URL found in goal/context"
                )
            return {
                "url": url,
                "max_chars": 6000,
            }

        if tool_name == "current_datetime":
            return {}

        if tool_name == "calculator":
            return {"expression": goal}

        if tool_name == "final_answer":
            return {"content": {"goal": goal, "context": context}}

        return {"goal": goal, "context": context}


def _context_to_text(context: dict[str, Any]) -> str:
    chunks = []
    for key, value in context.items():
        output = value.get("output", value) if isinstance(value, dict) else value
        chunks.append(f"{key}={str(output)[:200]}")
    return " | ".join(chunks)


def _extract_first_url_from_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"https?://[^\s)]+", text)
    return match.group(0) if match else None


def _extract_first_url_from_context(context: dict[str, Any]) -> str | None:
    for _, value in context.items():
        output = value.get("output", value) if isinstance(value, dict) else value
        if isinstance(output, dict):
            if isinstance(output.get("url"), str):
                return output.get("url")
            results = output.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and isinstance(item.get("url"), str):
                        return item["url"]
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and isinstance(item.get("url"), str):
                    return item["url"]
    return None
