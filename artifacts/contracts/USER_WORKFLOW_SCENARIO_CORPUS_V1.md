# User Workflow Scenario Corpus V1

Contract ID: `workflow-scenario-corpus-v1`  
Schema Version: `1`

This document defines the user-centric golden journey corpus enforced by:

- `crates/fnp-conformance/fixtures/workflow_scenario_corpus.json`
- `cargo run -p fnp-conformance --bin run_workflow_scenario_gate`
- `scripts/e2e/run_workflow_scenario_gate.sh`

## Required Scenario Categories

The corpus must contain at least one scenario for each category:

1. `high_frequency`
2. `high_risk`
3. `adversarial`

## Required Replay Metadata Per Scenario

Every scenario must include:

1. `id`
2. `seed`
3. `env_fingerprint`
4. `artifact_refs`
5. `reason_code`
6. `strict.expected_status`
7. `hardened.expected_status`

## Step Types

Supported step kinds:

1. `ufunc_input`
2. `runtime_policy`
3. `runtime_policy_wire`

All steps must carry deterministic IDs and fixture references that resolve to tracked conformance fixture collections.

## Linkage Requirements

Every scenario must link to:

1. Differential fixture IDs in `ufunc_input_cases.json`
2. E2E script paths present in the repository
3. Non-empty prioritized gap entries (`bead_id`, `owner`, `priority`, `description`)

## Current Gap Priorities (Initial V1 Set)

1. `bd-23m.12.7` owner `bd-23m.12.7` — reshape-heavy end-to-end workflows
2. `bd-23m.13.7` owner `bd-23m.13.7` — mixed-dtype pipeline workflows
3. `bd-23m.22` owner `bd-23m.22` — failure forensics UX improvements
4. `bd-23m.23` owner `bd-23m.23` — reliability/flake budgets for scenario replay
5. `bd-23m.20.7` owner `bd-23m.20.7` — malformed NPY/NPZ adversarial workflows
6. `bd-23m.17.7` owner `bd-23m.17.7` — hostile stride-trick replay workflows
