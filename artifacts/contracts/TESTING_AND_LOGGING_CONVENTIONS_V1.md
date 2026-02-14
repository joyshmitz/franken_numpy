# Testing and Logging Conventions V1

Contract ID: `test-logging-contract-v1`  
Schema Version: `1`

This document is the human-readable companion to `artifacts/contracts/test_logging_contract_v1.json`.
It defines mandatory unit/property testing conventions, fixture naming policy, structured logging rules, and gate commands.

## 1. Mandatory Invariant Families

Every conformance expansion must preserve and extend these invariant families:

1. `shape_stride_legality`
2. `dtype_promotion_table`
3. `runtime_policy_fail_closed`
4. `ufunc_metamorphic_relations`
5. `ufunc_adversarial_error_surface`

Current suite anchors:

- `shape_stride_legality` -> `run_shape_stride_suite`
- `dtype_promotion_table` -> `run_dtype_promotion_suite`
- `runtime_policy_fail_closed` -> `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`
- `ufunc_metamorphic_relations` -> `run_ufunc_metamorphic_suite`
- `ufunc_adversarial_error_surface` -> `run_ufunc_adversarial_suite`

## 2. Property Testing + Shrink Requirements

Property/adversarial fixtures must carry deterministic replay context and shrink metadata.

Mandatory rules:

1. Deterministic replay seed is required.
2. Shrinking order is `shape` minimization before `value` minimization.
3. `max_shrink_steps` is bounded (currently `32`).
4. Shrink-related failures must include a `reason_code`.

## 3. Structured Log Contract

Required structured log fields:

1. `fixture_id`
2. `seed`
3. `mode`
4. `env_fingerprint`
5. `artifact_refs`
6. `reason_code`

Replay semantics:

1. Property/adversarial runs are deterministic by `seed`.
2. Policy/adversarial fixtures require non-empty `reason_code`.
3. Policy/adversarial fixtures require non-empty `artifact_refs`.

## 4. Fixture Naming Convention

Fixture IDs must satisfy:

1. lowercase only
2. start with alphanumeric
3. characters limited to `a-z0-9_`

Examples:

- valid: `sum_axis_out_of_bounds`
- invalid: `_bad_prefix`
- invalid: `BadFixture`
- invalid: `bad-hyphen`

## 5. Canonical Test Helper APIs

The contract currently requires these helper API anchors:

1. `fnp_conformance::run_all_core_suites`
2. `fnp_conformance::test_contracts::run_test_contract_suite`
3. `fnp_conformance::set_runtime_policy_log_path`

## 6. Enforcement Gates

Contract enforcement commands:

1. `cargo test -p fnp-conformance test_contract_suite_is_green -- --nocapture`
2. `cargo run -p fnp-conformance --bin run_test_contract_gate`
3. `scripts/e2e/run_test_contract_gate.sh`

Gate behavior:

1. Missing contract fields or invalid schema fails the suite.
2. Missing seed/reason_code/artifact_refs in required fixture classes fails the suite.
3. Missing required gate script fails the suite.

## 7. Method-Stack Mapping (Alien/Optimization/Durability/Compatibility)

Mapped graveyard sections:

1. `alien_cs_graveyard.md` §6.12 (`Property-Based Testing with Shrinking`)
2. `alien_cs_graveyard.md` §0.19 (`Evidence Ledger Schema`)
3. `alien_cs_graveyard.md` §0.3 (`Isomorphism Proof Block`)
4. `alien_cs_graveyard.md` §0.4 (`Decision-Theoretic Runtime Contracts`)
5. `alien_cs_graveyard.md` §1.1 (`RaptorQ Fountain Codes`)

Mapped FrankenSuite summary sections:

1. `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12 (`Evidence Ledger Schema + CI Gates`)
2. `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.13 (`Archetype Playbooks for FrankenSuite`)
3. `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.19 (`Policy-as-Data`)

RaptorQ applicability note:

- Long-lived bundles (conformance fixtures, benchmark baselines, reproducibility ledgers) require sidecar/scrub/decode-proof artifacts.
- Ephemeral local gate logs are out of RaptorQ scope unless explicitly promoted to durable artifacts.
