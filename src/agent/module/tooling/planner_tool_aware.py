from __future__ import annotations

from typing import Any

from src.agent.module.planner import Planner
from src.agent.module.tooling.adapter import build_react_handoff
from src.agent.module.tooling.registry import ToolRegistry
from src.agent.module.tooling.validator import PlanValidationError, validate_tool_aware_plan
from src.utils.logger_config import logger, COLOR_CODES, RESET
from src.utils.utils import extract_json


class ToolAwarePlanner(Planner):
    """Planner that emits tool-aware DAG tasks for downstream ReAct workers."""

    def __init__(self, model, registry: ToolRegistry):
        super().__init__(model=model, env=None)
        self._name = "ToolAwarePlanner"
        self.registry = registry

    def decompose_task(self, prompt: str, max_retry: int = 3) -> tuple[dict[str, Any], bool, list[Any]]:
        failed_plans: list[Any] = []
        errors: list[str] = []

        retry_count = 0
        while retry_count < max_retry:
            try:
                response = self.model.predict(prompt)
                parsed = extract_json(response)
                validated = validate_tool_aware_plan(parsed, self.registry)
                return validated, True, failed_plans
            except (ValueError, PlanValidationError) as exc:
                failed_plans.append(response if "response" in locals() else None)
                errors.append(str(exc))
                retry_count += 1
                logger.info(
                    f"Tool-aware planning failed ({retry_count}/{max_retry}): "
                    f"{COLOR_CODES['RED']}{exc}{RESET}"
                )
                prompt = (
                    "Your previous response failed strict schema validation. "
                    "Return only valid JSON in the required format.\n\n"
                    + prompt
                )

        return {"plan": [], "errors": errors}, False, failed_plans

    def plan(
        self,
        prompt: str,
        max_retry: int = 3,
    ) -> tuple[dict[str, Any], bool, list[Any], list[dict[str, Any]]]:
        plan_payload, valid, failed_plans = self.decompose_task(prompt, max_retry=max_retry)
        if not valid:
            return plan_payload, False, failed_plans, []

        handoff = build_react_handoff(plan_payload, self.registry)
        return plan_payload, True, failed_plans, handoff
