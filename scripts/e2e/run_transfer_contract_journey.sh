#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
LOG_PATH="${1:-$ROOT_DIR/artifacts/logs/workflow_scenario_packet003_e2e_${TS}.jsonl}"
REPORT_PATH="${2:-${FNP_TRANSFER_WORKFLOW_RELIABILITY_REPORT:-$ROOT_DIR/artifacts/logs/workflow_scenario_packet003_reliability_${TS}.json}}"
ARTIFACT_INDEX_PATH="${3:-${FNP_TRANSFER_WORKFLOW_ARTIFACT_INDEX:-$ROOT_DIR/artifacts/logs/workflow_scenario_packet003_artifact_index_${TS}.json}}"
RETRIES="${FNP_TRANSFER_WORKFLOW_RETRIES:-${FNP_WORKFLOW_RETRIES:-1}}"
FLAKE_BUDGET="${FNP_TRANSFER_WORKFLOW_FLAKE_BUDGET:-${FNP_WORKFLOW_FLAKE_BUDGET:-0}}"
COVERAGE_FLOOR="${FNP_TRANSFER_WORKFLOW_COVERAGE_FLOOR:-${FNP_WORKFLOW_COVERAGE_FLOOR:-1.0}}"

cd "$ROOT_DIR"

echo "[transfer-contract-journey] root=$ROOT_DIR"
echo "[transfer-contract-journey] workflow_log=$LOG_PATH"
echo "[transfer-contract-journey] reliability_report=$REPORT_PATH"
echo "[transfer-contract-journey] artifact_index=$ARTIFACT_INDEX_PATH"
echo "[transfer-contract-journey] retries=$RETRIES flake_budget=$FLAKE_BUDGET coverage_floor=$COVERAGE_FLOOR"

rch exec -- cargo run -p fnp-conformance --bin run_workflow_scenario_gate -- \
  --log-path "$LOG_PATH" \
  --artifact-index-path "$ARTIFACT_INDEX_PATH" \
  --report-path "$REPORT_PATH" \
  --retries "$RETRIES" \
  --flake-budget "$FLAKE_BUDGET" \
  --coverage-floor "$COVERAGE_FLOOR"

echo "[transfer-contract-journey] completed"
