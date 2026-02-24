#!/usr/bin/env bash
set -euo pipefail

test_case="10-1-100-s"

python -m src.agent.main \
  --task specific_task \
  --template tool_aware_plan \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --scheduler parallel \
  --planner_mode tool_aware \
  --tool_registry "examples/tool_registry.example.json" \
  --worker_mode react_execute \
  --max_retry 2 \
  --test_case "${test_case}" \
  --output_dir "data/result/tool-aware-plan"
