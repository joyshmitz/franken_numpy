#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"

echo "[ci-topology] root=$ROOT_DIR"

echo "[ci-topology] G1 fmt+lint"
rch exec -- cargo fmt --check
rch exec -- cargo clippy --workspace --all-targets -- -D warnings

echo "[ci-topology] G2 unit+property"
rch exec -- cargo test --workspace --lib

echo "[ci-topology] G3 differential"
rch exec -- cargo run -p fnp-conformance --bin run_ufunc_differential

echo "[ci-topology] G4 adversarial+security"
scripts/e2e/run_security_policy_gate.sh

echo "[ci-topology] G5 test-contract"
scripts/e2e/run_test_contract_gate.sh

echo "[ci-topology] G6 workflow+forensics"
scripts/e2e/run_workflow_scenario_gate.sh

echo "[ci-topology] G7 performance-budget"
scripts/e2e/run_performance_budget_gate.sh

echo "[ci-topology] G8 durability+decode-proof"
scripts/e2e/run_raptorq_gate.sh

echo "[ci-topology] completed"
