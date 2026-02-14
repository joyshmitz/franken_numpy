#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
LOG_PATH="${1:-$ROOT_DIR/artifacts/logs/workflow_scenario_e2e_${TS}.jsonl}"

cd "$ROOT_DIR"

echo "[workflow-scenario-gate] root=$ROOT_DIR"
echo "[workflow-scenario-gate] workflow_log=$LOG_PATH"

cargo run -p fnp-conformance --bin run_workflow_scenario_gate -- --log-path "$LOG_PATH"

echo "[workflow-scenario-gate] completed"
