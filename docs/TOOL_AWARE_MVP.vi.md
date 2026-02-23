# Tool-Aware Planning MVP

Tài liệu này mô tả phần mở rộng `tool-aware` đã thêm theo hướng không phá legacy flow.

## 1) Mục tiêu

- Thêm planner aware toàn bộ tool catalog.
- Planner xuất `plan` có `allowed_tools` để feed cho ReAct workers.
- Giữ nguyên mode cũ (`legacy`) làm mặc định.

## 2) Thành phần mới

- Registry:
  - `src/agent/module/tooling/registry.py`
- Validator + schema normalization:
  - `src/agent/module/tooling/validator.py`
- Adapter sang payload cho ReAct worker:
  - `src/agent/module/tooling/adapter.py`
- Planner mới:
  - `src/agent/module/tooling/planner_tool_aware.py`
- Prompt template:
  - `template/tool_aware_plan.py`

## 3) Feature flags trong CLI

Đã thêm vào `src/agent/main.py`:
- `--planner_mode legacy|tool_aware` (default: `legacy`)
- `--tool_registry <path>` (bắt buộc khi `tool_aware`)
- `--worker_mode simulate|react_handoff` (default: `simulate`)

## 4) Tool registry mẫu

- `examples/tool_registry.example.json`

## 5) Chạy mẫu

Script mẫu:
- `script/test_tool_aware.sh`

Lưu ý:
- Script vẫn dùng model runtime, nên cần môi trường model/API phù hợp.
- Trong mode `tool_aware`, output lưu `react_handoff` payload để đưa sang worker layer.

## 6) Contract output (tool-aware)

Top-level JSON:
```json
{
  "plan": [
    {
      "name": "Subtask1",
      "goal": "...",
      "dependencies": [],
      "allowed_tools": ["tool_name"],
      "success_criteria": "...",
      "output_schema": {"type": "object"},
      "budget": {"max_calls": 3, "max_cost": 0.1},
      "timeout_sec": 60,
      "inputs_from": []
    }
  ]
}
```

## 7) Regression

Full test suite chạy bằng:
```bash
./script/phase0_regression.sh
```

Suite hiện bao gồm test legacy + test tool-aware modules.
