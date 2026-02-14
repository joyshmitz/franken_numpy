#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
LOG_PATH="${1:-$ROOT_DIR/artifacts/logs/test_contract_e2e_${TS}.jsonl}"

cd "$ROOT_DIR"

echo "[test-contract-gate] root=$ROOT_DIR"
echo "[test-contract-gate] runtime_policy_log=$LOG_PATH"

cargo run -p fnp-conformance --bin run_test_contract_gate -- --log-path "$LOG_PATH"

echo "[test-contract-gate] completed"
