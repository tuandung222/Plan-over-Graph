from __future__ import annotations

import ast
import json
from urllib.parse import urlencode
from urllib.request import urlopen
from typing import Any

from src.agent.module.tooling.registry import ToolRegistry


class ToolRuntimeError(RuntimeError):
    pass


class ToolRuntime:
    """Runtime executor for real tool calls used by tool-aware workers."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(self, tool_name: str, tool_input: dict[str, Any], timeout_sec: int = 30) -> dict[str, Any]:
        if not self.registry.has_tool(tool_name):
            raise ToolRuntimeError(f"unknown tool in runtime: {tool_name}")

        if not isinstance(tool_input, dict):
            raise ToolRuntimeError("tool_input must be a dictionary")

        if tool_name in ("duckduckgo_search", "web_search"):
            query = tool_input.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolRuntimeError("duckduckgo_search requires non-empty 'query' string")
            max_results = tool_input.get("max_results", 5)
            if not isinstance(max_results, int) or max_results <= 0:
                max_results = 5
            return self._duckduckgo_search(query=query.strip(), max_results=max_results, timeout_sec=timeout_sec)

        if tool_name == "calculator":
            expression = tool_input.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                raise ToolRuntimeError("calculator requires non-empty 'expression' string")
            value = _safe_eval_arithmetic(expression)
            return {"tool": "calculator", "expression": expression, "value": value}

        if tool_name == "final_answer":
            content = tool_input.get("content")
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            if not isinstance(content, str):
                content = str(content)
            return {"tool": "final_answer", "answer": content}

        raise ToolRuntimeError(f"unsupported tool execution path: {tool_name}")

    def _duckduckgo_search(self, query: str, max_results: int, timeout_sec: int) -> dict[str, Any]:
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1",
        }
        url = f"https://api.duckduckgo.com/?{urlencode(params)}"

        try:
            with urlopen(url, timeout=timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:
            raise ToolRuntimeError(f"duckduckgo request failed: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolRuntimeError(f"duckduckgo response is not valid json: {exc}") from exc

        results: list[dict[str, Any]] = []
        abstract_text = payload.get("AbstractText")
        abstract_url = payload.get("AbstractURL")
        heading = payload.get("Heading") or query

        if abstract_text:
            results.append(
                {
                    "title": heading,
                    "snippet": abstract_text,
                    "url": abstract_url,
                    "source": "duckduckgo_abstract",
                }
            )

        related = payload.get("RelatedTopics", [])
        flattened_topics = _flatten_related_topics(related)
        for topic in flattened_topics:
            if len(results) >= max_results:
                break
            text = topic.get("Text")
            first_url = topic.get("FirstURL")
            if not text:
                continue
            results.append(
                {
                    "title": text.split(" - ")[0],
                    "snippet": text,
                    "url": first_url,
                    "source": "duckduckgo_related",
                }
            )

        return {
            "tool": "duckduckgo_search",
            "query": query,
            "result_count": len(results),
            "results": results[:max_results],
        }


def _flatten_related_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for topic in topics:
        if "Topics" in topic and isinstance(topic["Topics"], list):
            flattened.extend(_flatten_related_topics(topic["Topics"]))
        elif isinstance(topic, dict):
            flattened.append(topic)
    return flattened


def _safe_eval_arithmetic(expression: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Mod,
        ast.Constant,
        ast.Load,
        ast.FloorDiv,
    )
    parsed = ast.parse(expression, mode="eval")
    for node in ast.walk(parsed):
        if not isinstance(node, allowed_nodes):
            raise ToolRuntimeError(f"unsupported expression element: {type(node).__name__}")
    return float(eval(compile(parsed, filename="<calc>", mode="eval"), {"__builtins__": {}}, {}))
