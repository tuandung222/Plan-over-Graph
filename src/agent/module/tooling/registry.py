import copy
import json
from typing import Any


class ToolRegistryError(ValueError):
    pass


class ToolRegistry:
    """In-memory registry for tool metadata used by tool-aware planning."""

    REQUIRED_FIELDS = {
        "name",
        "description",
        "input_schema",
        "output_schema",
    }

    def __init__(self, tools: list[dict[str, Any]]):
        if not isinstance(tools, list):
            raise ToolRegistryError("tools must be a list")

        self._tools: dict[str, dict[str, Any]] = {}
        for tool in tools:
            validated = self._validate_tool(tool)
            name = validated["name"]
            if name in self._tools:
                raise ToolRegistryError(f"duplicate tool name: {name}")
            self._tools[name] = validated

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | list[dict[str, Any]]) -> "ToolRegistry":
        if isinstance(payload, list):
            tools = payload
        elif isinstance(payload, dict) and "tools" in payload:
            tools = payload["tools"]
        else:
            raise ToolRegistryError("registry payload must be a list or a dict with key 'tools'")
        return cls(tools)

    @classmethod
    def from_file(cls, path: str) -> "ToolRegistry":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def list_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> dict[str, Any]:
        if name not in self._tools:
            raise ToolRegistryError(f"unknown tool: {name}")
        return copy.deepcopy(self._tools[name])

    def to_prompt_block(self) -> str:
        return json.dumps({"tools": list(self._tools.values())}, ensure_ascii=False, indent=2)

    def _validate_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tool, dict):
            raise ToolRegistryError("each tool must be a dictionary")

        missing = self.REQUIRED_FIELDS - set(tool.keys())
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ToolRegistryError(f"tool missing required fields: {missing_fields}")

        name = tool["name"]
        if not isinstance(name, str) or not name.strip():
            raise ToolRegistryError("tool.name must be a non-empty string")

        description = tool["description"]
        if not isinstance(description, str) or not description.strip():
            raise ToolRegistryError(f"tool.description must be a non-empty string: {name}")

        if not isinstance(tool["input_schema"], dict):
            raise ToolRegistryError(f"tool.input_schema must be an object: {name}")

        if not isinstance(tool["output_schema"], dict):
            raise ToolRegistryError(f"tool.output_schema must be an object: {name}")

        normalized = copy.deepcopy(tool)
        normalized["name"] = name.strip()
        normalized["description"] = description.strip()
        return normalized
