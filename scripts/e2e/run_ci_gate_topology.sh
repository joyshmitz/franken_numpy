#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORACLE_OUTPUT_PATH="$ROOT_DIR/crates/fnp-conformance/fixtures/oracle_outputs/ufunc_oracle_output.json"

cd "$ROOT_DIR"

resolve_require_real_oracle() {
  if [[ -n "${FNP_REQUIRE_REAL_NUMPY_ORACLE:-}" ]]; then
    printf '%s' "$FNP_REQUIRE_REAL_NUMPY_ORACLE"
    return
  fi

  if [[ "${CI:-}" == "1" || "${GITHUB_ACTIONS:-}" == "true" ]]; then
    printf '1'
  else
    printf '0'
  fi
}

read_oracle_source() {
  python3 - "$ORACLE_OUTPUT_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)
print(payload.get("oracle_source", ""))
PY
}

echo "[ci-topology] root=$ROOT_DIR"

echo "[ci-topology] G1 fmt+lint"
rch exec -- cargo fmt --check
rch exec -- cargo clippy --workspace --all-targets -- -D warnings

echo "[ci-topology] G2 unit+property"
rch exec -- cargo test --workspace --lib

echo "[ci-topology] G3 differential"
rch exec -- cargo run -p fnp-conformance --bin capture_numpy_oracle
rch exec -- cargo run -p fnp-conformance --bin run_ufunc_differential
REQUIRE_REAL_ORACLE="$(resolve_require_real_oracle)"
ORACLE_SOURCE="$(read_oracle_source)"
echo "[ci-topology] G3 oracle_source=$ORACLE_SOURCE require_real_oracle=$REQUIRE_REAL_ORACLE"
if [[ "$REQUIRE_REAL_ORACLE" == "1" && "$ORACLE_SOURCE" == "pure_python_fallback" ]]; then
  echo "[ci-topology] G3 failure: oracle capture used pure_python_fallback; configure NumPy-backed interpreter via FNP_ORACLE_PYTHON" >&2
  exit 1
fi

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
