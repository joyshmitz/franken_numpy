# EXHAUSTIVE_LEGACY_ANALYSIS.md — FrankenNumPy

Date: 2026-02-13  
Method stack: `$porting-to-rust` Phase-2 Deep Extraction + `$alien-artifact-coding` + `$extreme-software-optimization` + RaptorQ durability + frankenlibc/frankenfs strict/hardened doctrine.

## 0. Mission and Completion Criteria

This document defines exhaustive legacy extraction for FrankenNumPy. Phase-2 is complete only when each scoped subsystem has:
1. explicit invariants,
2. explicit crate ownership,
3. explicit oracle families,
4. explicit strict/hardened policy behavior,
5. explicit performance and durability gates.

## 1. Source-of-Truth Crosswalk

Legacy corpus:
- `/data/projects/franken_numpy/legacy_numpy_code/numpy`
- Upstream oracle: `numpy/numpy`

Project contracts:
- `/data/projects/franken_numpy/COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1.md` (sections 14-21)
- `/data/projects/franken_numpy/EXISTING_NUMPY_STRUCTURE.md`
- `/data/projects/franken_numpy/PLAN_TO_PORT_NUMPY_TO_RUST.md`
- `/data/projects/franken_numpy/PROPOSED_ARCHITECTURE.md`
- `/data/projects/franken_numpy/FEATURE_PARITY.md`

## DOC-PASS-00 Baseline Gap Matrix + Quantitative Expansion Targets

Snapshot baseline (2026-02-14):

| Document | Baseline lines | Target lines | Expansion multiplier | Anchor requirement |
|---|---:|---:|---:|---|
| `EXHAUSTIVE_LEGACY_ANALYSIS.md` | 275 | 3300 | 12.0x | Every new assertion links to legacy path and/or executable artifact |
| `EXISTING_NUMPY_STRUCTURE.md` | 62 | 992 | 16.0x | Every new subsystem map row links to crate owner + conformance anchor |

Pass-1 domain gap matrix (traceable to legacy anchors and executable evidence):

| Domain | Legacy anchors | Current depth (0-5) | Target multiplier | Unit/Property implications | Differential implications | E2E implications | Structured logging implications | Evidence anchors |
|---|---|---:|---:|---|---|---|---|---|
| Shape/stride legality | `numpy/_core/src/multiarray/shape.c`, `shape.h` | 3 | 3.0x | partial (`shape_stride_cases.json`) | partial (`run_ufunc_differential`) | missing dedicated journey | partial (runtime policy only) | `crates/fnp-conformance/fixtures/shape_stride_cases.json`, `crates/fnp-conformance/src/lib.rs` |
| Dtype promotion/casting | `dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | 2 | 4.0x | partial (`dtype_promotion_cases.json`) | missing cast-matrix differential | missing dedicated journey | missing per-cast reason codes | `crates/fnp-conformance/fixtures/dtype_promotion_cases.json` |
| Transfer/alias semantics | `dtype_transfer.c`, `lowlevel_strided_loops.c.src` | 1 | 6.0x | missing overlap property suite | missing overlap differential | missing overlap journey | missing overlap log taxonomy | legacy anchors only (no Rust artifact yet) |
| Ufunc dispatch/override | `umath/ufunc_object.c`, dispatch loops | 3 | 3.0x | partial (metamorphic/adversarial corpus) | partial (`ufunc_differential`) | missing override journey | partial (`fixture_id`,`seed` in fixtures) | `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/fixtures/ufunc_*` |
| NDIter traversal | `multiarray/nditer*` | 1 | 6.0x | missing | missing | missing | missing | `artifacts/phase2c/FNP-P2C-006/legacy_anchor_map.md`, `artifacts/phase2c/FNP-P2C-006/behavior_extraction_ledger.md` |
| Random subsystem | `numpy/random/*.pyx`, `random/src/*` | 1 | 5.0x | missing deterministic stream properties | missing RNG differential | missing seed/state journey | missing RNG log taxonomy | `crates/fnp-random/src/lib.rs` (stub) |
| Linalg subsystem | `numpy/linalg/lapack_lite/*` | 1 | 5.0x | missing solver invariants | missing solver differential | missing solver journey | missing linalg reason codes | `crates/fnp-linalg/src/lib.rs` (stub) |
| IO subsystem | `numpy/lib/format.py`, npy/npz paths | 2 | 4.0x | missing malformed-header property coverage | missing io round-trip differential | missing io journey | missing parser reason-code families | `crates/fnp-io/src/lib.rs` (stub) |

Contradictions/unknowns register (must be closed during doc-overhaul passes):

| ID | Contradiction or unknown | Source/evidence anchor | Owner | Risk | Closure criteria |
|---|---|---|---|---|---|
| `DOC-C001` | `EXISTING_NUMPY_STRUCTURE.md` still states a reduced-scope “V1 extraction boundary” which conflicts with full drop-in parity doctrine. | `EXISTING_NUMPY_STRUCTURE.md` section 6 vs `COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1.md` sections 0 and 2 | `bd-23m.24.2` | critical | Replace scope-cut language with parity-debt sequencing language and explicit ownership table. |
| `DOC-C002` | Verification/logging implication mapping is now documented in DOC-PASS-03, but multiple packet domains remain missing/deferred. | `EXHAUSTIVE_LEGACY_ANALYSIS.md` DOC-PASS-03.4 + `EXISTING_NUMPY_STRUCTURE.md` DOC-PASS-03.4 | `bd-23m.24.4` | high | Keep pass-03 matrices synchronized with packet execution until unresolved domains are implementation-backed and evidence-complete. |
| `DOC-C003` | Packet ownership crosswalk for unresolved domains is now documented in DOC-PASS-02.4, but executable crate boundaries remain unimplemented for those domains. | `EXHAUSTIVE_LEGACY_ANALYSIS.md` DOC-PASS-02.4 + crate stubs (`fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`) | `bd-23m.24.3` | high | Keep crosswalk synchronized with packet execution and close only when packet owners replace stubs with production surfaces and full E/F/G evidence. |
| `DOC-C004` | Structured logging contract was added but not yet threaded into all subsystem extraction sections. | `artifacts/contracts/test_logging_contract_v1.json`, `artifacts/contracts/TESTING_AND_LOGGING_CONVENTIONS_V1.md` | `bd-23m.24.10` | medium | Each packet section names required log fields and reason-code taxonomy coverage status. |

## DOC-PASS-01 Full Module/Package Cartography with Ownership and Boundaries

### DOC-PASS-01.1 Workspace ownership map (crate -> behavior contract)

| Crate | Ownership boundary (Rust anchors) | Legacy anchor families | Executable evidence anchors | Boundary contract |
|---|---|---|---|---|
| `fnp-dtype` | `crates/fnp-dtype/src/lib.rs` (`DType`, `promote`, `can_cast_lossless`) | `numpy/_core/src/multiarray/dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | `crates/fnp-conformance/fixtures/dtype_promotion_cases.json`, `crates/fnp-conformance/src/lib.rs` (`run_dtype_promotion_suite`) | Owns deterministic dtype identity/promotion/cast primitives only; does not own array layout, iteration, or execution kernels. |
| `fnp-ndarray` | `crates/fnp-ndarray/src/lib.rs` (`broadcast_shape`, `broadcast_shapes`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout`) | `numpy/_core/src/multiarray/shape.c`, `shape.h`, stride legality loci | `crates/fnp-conformance/fixtures/shape_stride_cases.json`, `run_shape_stride_suite` | Owns shape/stride legality and contiguous layout calculus; does not own dtype promotion policy or numeric kernels. |
| `fnp-ufunc` | `crates/fnp-ufunc/src/lib.rs` (`UFuncArray`, `BinaryOp`, `elementwise_binary`, `reduce_sum`) | `numpy/_core/src/umath/ufunc_object.c`, reduction/dispatch loop families | `crates/fnp-conformance/src/ufunc_differential.rs`, `ufunc_*` fixtures, `run_ufunc_*` suites | Owns broadcasted elementwise/reduction execution semantics for scoped ops; relies on `fnp-dtype` + `fnp-ndarray` for promotion/layout decisions. |
| `fnp-runtime` | `crates/fnp-runtime/src/lib.rs` (`decide_compatibility*`, `evaluate_policy_override`, `EvidenceLedger`) | Runtime policy doctrine from strict/hardened contract set (fail-closed matrix) | `runtime_policy_cases.json`, `runtime_policy_adversarial_cases.json`, workflow scenario suite | Owns strict/hardened compatibility decisions, fail-closed wire decoding, and policy/audit evidence records; does not own numerical semantics. |
| `fnp-conformance` | `crates/fnp-conformance/src/lib.rs` + submodules (`contract_schema`, `security_contracts`, `test_contracts`, `workflow_scenarios`, `benchmark`, `raptorq_artifacts`, `ufunc_differential`) | Oracle/tests from `legacy_numpy_code/numpy`, packet artifact topology contracts | `run_all_core_suites`, contract/test/security/workflow suites, RaptorQ sidecar/scrub/decode proof tooling | Owns verification and evidence production/validation pipeline; must not redefine core semantics owned by execution crates. |
| `fnp-iter` | `crates/fnp-iter/src/lib.rs` (`TransferSelectorInput`, `select_transfer_class`, `overlap_copy_policy`, `validate_nditer_flags`, `FlatIterIndex`, `TransferLogRecord`) | `numpy/_core/src/multiarray/dtype_transfer.c`, `lowlevel_strided_loops.c.src`, `numpy/_core/src/multiarray/nditer*` | crate unit/property tests + packet refs (`artifacts/phase2c/FNP-P2C-003/fixture_manifest.json`, `parity_gate.yaml`) | Owns transfer selector/overlap policy/flatiter contract checks and packet-003 reason-code taxonomy; packet-local differential/e2e gates are still pending. |
| `fnp-random` | `crates/fnp-random/src/lib.rs` (`DeterministicRng`, `RandomError`, `RandomLogRecord`, bounded/float/state APIs) | `numpy/random/_generator.pyx`, `numpy/random/src/*` | crate unit/property tests + packet refs (`artifacts/phase2c/FNP-P2C-007/fixture_manifest.json`, `parity_gate.yaml`) | Owns deterministic stream/state constructor contracts and replay-log schema for packet-007 first wave; oracle differential and workflow lanes are still pending. |
| `fnp-linalg` | `crates/fnp-linalg/src/lib.rs` (`LinAlgError`, `QrMode`, `solve_2x2`, `svd_output_shapes`, `lstsq_output_shapes`, policy validators, `LinAlgLogRecord`) | `numpy/linalg/lapack_lite/*`, high-level wrappers in `numpy/linalg/*` | crate unit/property tests + packet refs (`artifacts/phase2c/FNP-P2C-008/contract_table.md`, `unit_property_evidence.json`) | Owns linalg shape/solver/mode/tolerance/backend fail-closed contracts for packet-008 first wave; broader parity matrix and differential/e2e lanes remain open. |
| `fnp-io` | `crates/fnp-io/src/lib.rs` (`NpyHeader`, `IOError`, magic/header/descriptor/memmap/load/npz validators, `IOLogRecord`) | `numpy/lib/format.py`, npy/npz IO paths | crate unit/property tests + packet refs (`artifacts/phase2c/FNP-P2C-009/contract_table.md`, `unit_property_evidence.json`) | Owns NPY/NPZ contract validation, bounded parser budgets, and fail-closed metadata policy for packet-009 first wave; full oracle/e2e parity lanes remain open. |

### DOC-PASS-01.2 Dependency direction and layering constraints

Workspace dependency graph (from `Cargo.toml` manifests):

1. Foundation leaves: `fnp-dtype`, `fnp-ndarray`, `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`.
2. Execution layer: `fnp-ufunc` -> (`fnp-dtype`, `fnp-ndarray`).
3. Policy layer: `fnp-runtime` (optional feature links: `asupersync`, `ftui`).
4. Verification layer: `fnp-conformance` -> (`fnp-dtype`, `fnp-ndarray`, `fnp-ufunc`, `fnp-runtime`) + evidence dependencies (`serde*`, `sha2`, `base64`, `asupersync`).

Layering law (non-negotiable for this pass):

- Core semantics flow upward: dtype/shape -> execution -> runtime policy -> conformance/evidence.
- Verification modules may call execution/policy crates, but execution crates must not depend on conformance crates.
- Stub crates (`fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`) are reserved ownership slots; promotion to active owners requires packet A->I evidence artifacts.

Potentially invalid edges to reject:

| Forbidden edge | Why forbidden | Detection/closure |
|---|---|---|
| `fnp-dtype` -> `fnp-ufunc`/`fnp-conformance` | Inverts semantic hierarchy and couples primitives to higher layers | Enforce via manifest review in packet D-stage and contract-schema checks. |
| `fnp-ndarray` -> `fnp-runtime` | Layout legality must remain policy-neutral | Reject in implementation plans (`artifacts/phase2c/*/implementation_plan.md`). |
| Any crate -> `fnp-conformance` (except tests/tools) | Would mix production semantics with oracle/harness code | Gate via Cargo review + `cargo check --workspace --all-targets`. |

### DOC-PASS-01.3 Module hot paths, adapters, and compatibility pockets

| Path | Role type | Why it is a topology hotspot |
|---|---|---|
| `crates/fnp-ndarray/src/lib.rs` (`broadcast_shape`, `fix_unknown_dimension`, `contiguous_strides`) | Hot path | Central SCE legality kernel; every broadcast/reshape legality decision funnels here. |
| `crates/fnp-ufunc/src/lib.rs` (`UFuncArray::elementwise_binary`, `reduce_sum`) | Hot path | Numeric execution path that combines broadcast mapping and reduction index traversal. |
| `crates/fnp-runtime/src/lib.rs` (`decide_compatibility_from_wire`, `decide_and_record_with_context`) | Policy adapter | Converts wire metadata into strict/hardened decisions and audit ledger events. |
| `crates/fnp-conformance/src/ufunc_differential.rs` (`PY_CAPTURE_SCRIPT`, `capture_numpy_oracle`, `compare_against_oracle`) | Oracle adapter/shim | Bridges Rust execution to legacy/system NumPy (or pure fallback) for parity comparisons. |
| `crates/fnp-conformance/src/contract_schema.rs` | Artifact contract validator | Enforces packet artifact topology (`phase2c-contract-v1`) before packet promotion. |
| `crates/fnp-conformance/src/workflow_scenarios.rs` | Integration shim | Links fixture-level checks to E2E scenario corpus and replay logging expectations. |
| `crates/fnp-conformance/src/raptorq_artifacts.rs` | Durability adapter | Generates/validates sidecars, scrub reports, and decode proofs for long-lived evidence bundles. |
| `crates/fnp-iter/src/lib.rs` (`select_transfer_class`, `overlap_copy_policy`, flatiter validators) | Transfer/index hotspot | Encodes packet-003 transfer legality and overlap/broadcast policy checks that future NDIter and stride-tricks paths depend on. |
| `crates/fnp-random/src/lib.rs` (`DeterministicRng::{next_u64,next_f64,bounded_u64,state}`) | RNG hotspot | Encodes deterministic-seed/state semantics and bounded-generation policy for packet-007. |
| `crates/fnp-linalg/src/lib.rs` (shape/solver/qr/svd/lstsq/policy validators) | Linalg contract hotspot | Captures packet-008 fail-closed shape and backend policy semantics that gate solver behavior. |
| `crates/fnp-io/src/lib.rs` (magic/header/descriptor/memmap/load/npz validators) | IO security hotspot | Captures packet-009 parser-budget and contract checks for NPY/NPZ boundary hardening. |
| `crates/fnp-conformance/src/bin/*.rs` gate binaries | Workflow orchestration layer | Operational entrypoints (`run_security_gate`, `run_test_contract_gate`, `run_workflow_scenario_gate`, etc.) that decide what evidence actually ships in CI/gate runs. |

### DOC-PASS-01.4 Verification implications by ownership boundary (covered/missing/deferred)

| Ownership boundary | Unit/Property | Differential | E2E | Structured logging | Status + owner |
|---|---|---|---|---|---|
| `fnp-ndarray` shape/stride legality | covered (crate tests + shape/stride fixtures) | covered (indirect via ufunc differential shape checks), still partial | deferred to packet scenario layers | partial (runtime/workflow logs exist; shape-specific reason taxonomy incomplete) | partial; packet owner `bd-23m.12` |
| `fnp-dtype` promotion/cast primitives | covered (crate tests + promotion fixture suite) | missing full cast-matrix differential | missing packet-level journey | missing per-cast reason-code taxonomy | missing/partial; packet owner `bd-23m.13` |
| `fnp-ufunc` scoped ops (`add/sub/mul/div/sum`) | covered (crate tests + metamorphic/adversarial suites) | covered for scoped fixtures; broader NumPy surface missing | deferred (workflow corpus links exist, packet-local journeys incomplete) | covered at fixture/scenario level | partial; packet owner `bd-23m.17` + `bd-23m.19` |
| `fnp-runtime` strict/hardened policy | covered (crate tests + policy suites) | not applicable as numerical diff; policy-wire adversarial coverage is covered | covered via workflow scenario suite | covered (`fixture_id`,`seed`,`mode`,`env_fingerprint`,`artifact_refs`,`reason_code`) | covered for current surface; continuing under foundation beads `bd-23m.6`, `bd-23m.23` |
| `fnp-iter` transfer/index contracts | covered (unit + property grids in crate tests) | missing packet-local oracle differential lane | missing packet-local journey | covered at record-schema level; not yet gate-enforced | partial; packet owners `bd-23m.14` + `bd-23m.17` |
| `fnp-random` deterministic stream/state | covered (determinism/state/bounded/range tests) | missing dedicated RNG oracle differential lane | missing seed/state replay journey | covered by `RandomLogRecord` schema tests; not yet gate-enforced | partial; packet owner `bd-23m.18` |
| `fnp-linalg` solver/shape/backend contracts | covered (unit/property contract tests) | missing dedicated linalg oracle differential lane | missing solver workflow replay journey | covered by `LinAlgLogRecord` schema checks; not yet gate-enforced | partial; packet owner `bd-23m.19` |
| `fnp-io` NPY/NPZ parser/writer contracts | covered (unit/property parser/budget/policy tests) | missing dedicated IO round-trip differential lane | missing IO replay journey | covered by `IOLogRecord` schema checks; not yet gate-enforced | partial; packet owner `bd-23m.20` |

### DOC-PASS-01.5 Contradictions and unknowns register (topology-specific)

| ID | Contradiction / unknown | Evidence anchors | Risk | Owner | Closure criteria |
|---|---|---|---|---|---|
| `TOPO-C001` | Packet-D boundaries are implemented for `fnp-iter`/`fnp-random`/`fnp-linalg`/`fnp-io`, but `run_all_core_suites` does not yet execute packet-local F/G/H/I lanes for those domains, so topology ownership is only partially exercised in gates. | `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs`, `crates/fnp-conformance/src/lib.rs` (`run_all_core_suites`) | high | `bd-23m.17`, `bd-23m.18`, `bd-23m.19`, `bd-23m.20` | Add packet-local differential/e2e/gate wiring (F/G) and include those suites in operational gate paths before claiming full boundary closure. |
| `TOPO-C002` | Differential harness is currently concentrated in ufunc/policy surfaces; packet-specific differential lanes for transfer, RNG, linalg, and IO are not wired. | `crates/fnp-conformance/src/lib.rs`, fixture inventory under `crates/fnp-conformance/fixtures` | high | packet F-stage beads (`bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6`) | Add packet-local fixture manifests, oracle capture paths, and differential report outputs with reason codes. |
| `TOPO-C003` | Layering constraints are documented but not machine-enforced as a workspace contract test. | Root `Cargo.toml` + crate manifests | medium | foundation gate `bd-23m.23` | Add automated dependency-direction check to reliability gate outputs and fail CI on forbidden edges. |
| `TOPO-C004` | Runtime/workflow structured logs are optional via path configuration, allowing silent no-op when env/config is absent. | `set_runtime_policy_log_path`, `set_workflow_scenario_log_path`, `maybe_append_runtime_policy_log` | medium | foundation orchestration `bd-23m.6` | Gate runs must set explicit log paths and fail when required log artifacts are absent. |

## DOC-PASS-02 Symbol/API Census and Surface Classification

### DOC-PASS-02.1 Crate-level public symbol census

| Crate | Public symbol families (Rust anchors) | Classification | Legacy correspondence |
|---|---|---|---|
| `fnp-dtype` | `DType` enum; `promote`; `can_cast_lossless` (`crates/fnp-dtype/src/lib.rs`) | Compatibility-kernel API (Tier A) | `dtypemeta.c`, `descriptor.c`, `can_cast_table.h` promotion/cast families |
| `fnp-ndarray` | `MemoryOrder`, `ShapeError`, `broadcast_*`, `element_count`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout` (`crates/fnp-ndarray/src/lib.rs`) | Compatibility-kernel API (Tier A) | `shape.c`, `shape.h`, stride legality/broadcast shape semantics |
| `fnp-ufunc` | `BinaryOp`, `UFuncArray` constructors/accessors/ops, `UFuncError` (`crates/fnp-ufunc/src/lib.rs`) | Execution API (Tier A) | `umath/ufunc_object.c` dispatch/reduction semantics |
| `fnp-runtime` | mode/class/action enums; decision and ledger APIs; override gate; posterior/expected-loss helpers (`crates/fnp-runtime/src/lib.rs`) | Policy/audit API (Tier A) | Strict/hardened compatibility doctrine and fail-closed runtime matrix |
| `fnp-conformance` | `HarnessConfig`, `SuiteReport`, all `run_*` suites + module exports in `benchmark`, `contract_schema`, `raptorq_artifacts`, `security_contracts`, `test_contracts`, `ufunc_differential`, `workflow_scenarios` | Verification/tooling API (Tier B) | Legacy oracle capture/comparison and packet artifact contract validation |
| `fnp-iter` | transfer/index APIs (`TransferSelectorInput`, `TransferError`, `select_transfer_class`, overlap/flatiter validators, `TransferLogRecord`) (`crates/fnp-iter/src/lib.rs`) | Packet-local first-wave API (Tier C) | `dtype_transfer.c`/`nditer*` transfer and overlap policy families |
| `fnp-random` | `DeterministicRng`, `RandomError`, `RandomLogRecord`, stream/state/bounded sampling APIs (`crates/fnp-random/src/lib.rs`) | Packet-local first-wave API (Tier C) | `numpy/random/_generator.pyx` constructor/stream semantics |
| `fnp-linalg` | `LinAlgError`, `QrMode`, solver/shape/policy validators, `LinAlgLogRecord` (`crates/fnp-linalg/src/lib.rs`) | Packet-local first-wave API (Tier C) | `numpy/linalg/lapack_lite/*` and wrapper behavior contracts |
| `fnp-io` | `NpyHeader`, `IOError`, NPY/NPZ boundary validators, `IOLogRecord` (`crates/fnp-io/src/lib.rs`) | Packet-local first-wave API (Tier C) | `numpy/lib/format.py` and npy/npz dispatch/parser contracts |

### DOC-PASS-02.2 Call/usage context graph (operator paths -> API surfaces)

| Entry path | Immediate API calls | Downstream semantic surfaces |
|---|---|---|
| `capture_numpy_oracle` bin | `ufunc_differential::capture_numpy_oracle` | Legacy/system NumPy capture into fixture oracle JSON |
| `run_ufunc_differential` bin | `compare_against_oracle`, `write_differential_report` | `execute_input_case` -> `UFuncArray::{new,elementwise_binary,reduce_sum}` + dtype parsing |
| `run_security_gate` bin | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, `security_contracts::run_security_contract_suite` | `decide_and_record_with_context`, `decide_compatibility_from_wire`, runtime ledger/log contract checks |
| `run_test_contract_gate` bin | `test_contracts::run_test_contract_suite` + runtime suites + log schema validation | fixture-shape/metadata contracts and structured logging field enforcement |
| `run_workflow_scenario_gate` bin | `run_user_workflow_scenario_suite` | mixed pipeline: ufunc execution + runtime policy + scenario log emission (`fixture_id`,`seed`,`mode`,`env_fingerprint`,`artifact_refs`,`reason_code`) |
| `generate_benchmark_baseline` bin | `benchmark::generate_benchmark_baseline` | controlled workload timing and baseline artifact emission |
| `generate_raptorq_sidecars` bin | `raptorq_artifacts::generate_bundle_sidecar_and_reports` | sidecar generation, scrub report, decode-proof workflow |
| `validate_phase2c_packet` bin | `contract_schema::validate_phase2c_packet`, `write_packet_readiness_report` | packet artifact completeness and schema/token checks |

### DOC-PASS-02.3 Stability and user-visibility classification

| Tier | Definition | Current symbol groups | Risk if treated as stable too early |
|---|---|---|---|
| Tier A | Implemented compatibility-kernel/public semantics used by harness execution | `fnp-dtype`, `fnp-ndarray`, `fnp-ufunc`, `fnp-runtime` public APIs | Medium: behavior is first-wave complete, not full-NumPy complete; parity drift remains possible outside scoped fixtures. |
| Tier B | Operational tooling APIs for CI/gates/artifact lifecycle | `fnp-conformance` public modules and binaries | Medium: interfaces are workflow-stable but still evolving with packet schema/logging requirements. |
| Tier C | Packet-local first-wave boundaries with incomplete gate integration | `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io` public contract APIs | High: APIs exist, but full packet-level differential/e2e/optimization evidence is incomplete, so production parity claims remain invalid. |

### DOC-PASS-02.4 Packet ownership index for unresolved behavior domains (`DOC-C003`)

| Behavior domain | Placeholder surface now | Packet owner | Required closure gates (minimum) | Evidence anchors |
|---|---|---|---|---|
| NDIter traversal/index semantics | transfer/index boundary APIs in `fnp-iter` (selector, overlap, flatiter validators, logs) | `bd-23m.17` / `FNP-P2C-006` | Extend from packet-D boundary into packet-F/G oracle + replay lanes and integrate into conformance gate workflows | `artifacts/phase2c/FNP-P2C-006/*`, `crates/fnp-iter/src/lib.rs` |
| RNG streams + state schema | deterministic RNG/state APIs in `fnp-random` | `bd-23m.18` / `FNP-P2C-007` | Add packet-F/G oracle differential + replay suites and gate wiring; keep deterministic-seed witness artifacts mandatory | `artifacts/phase2c/FNP-P2C-007/*`, `crates/fnp-random/src/lib.rs` |
| Linalg adapters/solver contracts | shape/solver/policy contracts in `fnp-linalg` | `bd-23m.19` / `FNP-P2C-008` | Add packet-F/G linalg differential + replay lanes and packet-H optimization/isomorphism evidence | `artifacts/phase2c/FNP-P2C-008/*`, `crates/fnp-linalg/src/lib.rs` |
| NPY/NPZ parser-writer contracts | parser/writer boundary validators in `fnp-io` | `bd-23m.20` / `FNP-P2C-009` | Add packet-F/G IO round-trip/adversarial lanes and gate-enforced logging/durability links | `artifacts/phase2c/FNP-P2C-009/*`, `crates/fnp-io/src/lib.rs` |

### DOC-PASS-02.5 Verification implications by symbol family

| Symbol family | Unit/Property | Differential | E2E | Structured logging | Assessment |
|---|---|---|---|---|---|
| Shape/stride symbols (`fnp-ndarray`) | covered (crate tests + `shape_stride_cases.json`) | partial through ufunc differential shape validation | deferred packet-local scenarios | partial | parity debt remains for richer alias/transfer semantics |
| Dtype symbols (`fnp-dtype`) | covered for promotion/cast primitives | partial (promotion diff exists, full cast matrix absent) | deferred | missing cast taxonomy | medium risk |
| Ufunc symbols (`fnp-ufunc`) | covered for scoped ops | covered for scoped fixture corpus | partial via workflow suite | covered at fixture/scenario level | medium risk |
| Runtime policy symbols (`fnp-runtime`) | covered (unit + policy fixtures) | adversarial wire-class differential equivalent covered | covered in workflow scenarios | covered with required fields | low-medium risk |
| Verification/tooling symbols (`fnp-conformance`) | covered by module tests and gate execution | N/A | covered by gate binaries | covered by contract suites | medium operational risk (schema evolution) |
| Packet-local first-wave symbols (`fnp-iter`/`fnp-random`/`fnp-linalg`/`fnp-io`) | covered (crate unit/property contract suites) | missing packet-local oracle differential lanes | missing packet-local workflow replay lanes | covered at record-schema level; not yet enforced in gate outputs | high parity debt (integration/evidence gap) |

### DOC-PASS-02.6 Symbol-layer contradictions and unknowns

| ID | Contradiction/unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `SYM-C001` | Tier-C packet-local APIs now exist for iter/random/linalg/io, but gate orchestration still under-represents them compared with ufunc/runtime lanes. | high | `bd-23m.17`/`18`/`19`/`20` | Add packet-local suites to conformance gate execution paths and require parity artifacts before promoting these APIs beyond packet scope. |
| `SYM-C002` | Runtime and workflow log appenders are path-config driven; absent config can silently skip log emission. | medium | `bd-23m.6` | Require explicit log path in orchestrator/gates and fail when expected logs are absent. |
| `SYM-C003` | Differential harness is strongest for ufunc/runtime but weaker for cast matrix, transfer/alias, RNG, linalg, and IO. | high | packet F-stage beads (`bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6`) | Add packet-specific fixture and oracle diff lanes with failure taxonomy and reproducible artifacts. |

## DOC-PASS-03 Data Model, State, and Invariant Mapping

### DOC-PASS-03.1 Canonical data model map (state-bearing types)

| Model family | Concrete types (anchors) | State schema | Ownership |
|---|---|---|---|
| Dtype identity/promotion | `DType` + `promote` + `can_cast_lossless` (`crates/fnp-dtype/src/lib.rs`) | finite scalar domain + deterministic promotion/cast relation | `fnp-dtype` |
| Shape/stride legality | `MemoryOrder`, `ShapeError`, `NdLayout` + legality functions (`crates/fnp-ndarray/src/lib.rs`) | shape vectors, stride vectors, item-size arithmetic, legality error taxonomy | `fnp-ndarray` |
| Numeric execution payload | `UFuncArray`, `BinaryOp`, `UFuncError` (`crates/fnp-ufunc/src/lib.rs`) | shape/value/dtype tuple with broadcasted binary and reduction transitions | `fnp-ufunc` |
| Runtime policy/audit | `RuntimeMode`, `CompatibilityClass`, `DecisionAction`, `DecisionAuditContext`, `DecisionEvent`, `OverrideAuditEvent`, `EvidenceLedger` (`crates/fnp-runtime/src/lib.rs`) | mode/class/risk/threshold input state -> action + posterior/loss + audit metadata and append-only ledger | `fnp-runtime` |
| Conformance fixtures/reports | `HarnessConfig`, `SuiteReport`, `UFuncInputCase`, `UFuncDifferentialReport`, workflow scenario structs (`crates/fnp-conformance/src/lib.rs`, `ufunc_differential.rs`, `workflow_scenarios.rs`) | corpus-driven execution state, case counters, failure diagnostics | `fnp-conformance` |
| Artifact contract/durability | `PacketReadinessReport`, `MissingField` (`contract_schema.rs`), `RaptorQSidecar`, `ScrubReport`, `DecodeProofArtifact` (`raptorq_artifacts.rs`) | packet readiness state + integrity/durability metadata | `fnp-conformance` |
| Packet-local first-wave ownership models | transfer/index contracts (`fnp-iter`), deterministic RNG/state (`fnp-random`), linalg shape/solver policies (`fnp-linalg`), NPY/NPZ boundary validators (`fnp-io`) | domain-specific contract models with reason-code vocabularies and replay-log records; not yet fully gate-integrated | packet owners `bd-23m.17`/`18`/`19`/`20` |

### DOC-PASS-03.2 State transition ledger (input class -> transition -> terminal)

| Transition class | Entry state | Transition function(s) | Success terminal | Failure terminal |
|---|---|---|---|---|
| Broadcast merge | `(lhs_shape, rhs_shape)` | `broadcast_shape`, `broadcast_shapes` | deterministic merged shape | `ShapeError::IncompatibleBroadcast` |
| Reshape inference | `(new_shape spec, old_count)` | `fix_unknown_dimension` | resolved concrete shape | `MultipleUnknownDimensions` / `InvalidDimension` / `IncompatibleElementCount` / `Overflow` |
| Layout derivation | `(shape, item_size, order)` | `contiguous_strides`, `NdLayout::contiguous` | legal contiguous layout | `InvalidItemSize` / `Overflow` |
| Ufunc binary op | `(lhs UFuncArray, rhs UFuncArray, op)` | `elementwise_binary` | new output `UFuncArray` with promoted dtype | shape or input-length error via `UFuncError` |
| Ufunc reduction | `(UFuncArray, axis, keepdims)` | `reduce_sum` | reduced output `UFuncArray` | `AxisOutOfBounds` / shape-derived errors |
| Runtime policy decision | `(mode,class,risk,threshold)` | `decide_compatibility`, `decide_compatibility_from_wire` | `allow` or `full_validate` or `fail_closed` | unknown mode/class intentionally collapses to `fail_closed` |
| Runtime event capture | `(ledger, decision context)` | `decide_and_record_with_context` | appended `DecisionEvent` | normalized default tokens when context fields are empty |
| Workflow scenario execution | `(scenario fixture, mode)` | `run_user_workflow_scenario_suite`, `execute_mode` | pass/fail status with step-level evidence | deterministic failure records for missing fixtures/scripts or expectation mismatch |
| Packet readiness gate | `(packet artifact dir)` | `validate_phase2c_packet` | ready report | non-ready report with missing artifact/field/parse diagnostics |
| Durability pipeline | `(bundle files)` | `generate_bundle_sidecar_and_reports`, scrub/decode helpers | sidecar + scrub + decode proof | integrity mismatch/decode failure surfaces |

### DOC-PASS-03.3 Mutability boundaries and trust zones

| Zone | Mutable surface | Guard rail | Residual risk |
|---|---|---|---|
| Compatibility kernel data | derived vectors/buffers during computation | constructor + precondition checks (shape length, axis bounds, overflow guards) | public struct fields in some models can be mutated after construction if misused |
| Runtime policy ledger | `EvidenceLedger.events` append path | append-only API (`record`) + context normalization | downstream consumers must treat missing logs as gate failure, not optional telemetry |
| File-backed audit logs | runtime/workflow JSONL append | contract suites validate required fields | if log path not configured outside gates, append may no-op |
| Artifact outputs | readiness reports + RaptorQ artifacts | schema/token/path validation + hash checks | stale artifacts can drift without periodic gate enforcement |
| Placeholder crates | trivial function bodies | explicit documentation + packet ownership mapping | accidental reliance on placeholders as real APIs |

### DOC-PASS-03.4 Invariant obligations with evidence status

| Invariant | Expression | Evidence status | Logging/e2e implication |
|---|---|---|---|
| `INV-SHAPE-001` | element counts and stride arithmetic never overflow into unchecked state | covered by `fnp-ndarray` unit tests + shape fixtures | errors should map to stable reason taxonomy in packet logs (partial today) |
| `INV-SHAPE-002` | broadcast legality is deterministic and fail-fast on incompatible dimensions | covered for current shape corpus | dedicated broadcast-focused e2e journey still deferred |
| `INV-RESHAPE-001` | reshape `-1` inference is unique and count-preserving | covered by unit tests | packet-local reason-code coverage pending |
| `INV-UFUNC-001` | ufunc construction enforces value length == shape element count | covered by unit tests + differential path | workflow logs capture case/step context for failures |
| `INV-UFUNC-002` | reduction axis must be in bounds and output shape semantics respect `keepdims` | covered by unit tests + adversarial fixtures | scenario logs include pass/fail + detail per step |
| `INV-POLICY-001` | unknown/incompatible wire semantics must fail closed | covered (runtime adversarial suite + workflow scenarios) | required structured log fields are enforced by test/security contract gates |
| `INV-AUDIT-001` | decision events always carry non-empty audit identity fields | covered via context normalization + log validators | missing field is a contract/gate failure |
| `INV-CONTRACT-001` | packet readiness requires mandatory files/tokens/paths | covered by `validate_phase2c_packet` | readiness reports are explicit triage artifacts |
| `INV-DURABILITY-001` | sidecar/scrub/decode artifacts preserve payload integrity | covered for generated bundles | decode proof artifacts are mandatory for durable claim |
| `INV-OWNERSHIP-001` | unresolved domains must expose real packet-scoped models before parity claims | missing (stubs remain) | packet owners must add unit/differential/e2e/logging trails before closure |

### DOC-PASS-03.5 Data-model contradictions and closure backlog

| ID | Contradiction/unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `MODEL-C001` | `NdLayout` exposes public fields, so post-construction mutation can violate assumptions outside constructor checks. | medium | `bd-23m.12` | Decide encapsulation policy (public-by-contract vs encapsulated mutators) and enforce consistently. |
| `MODEL-C002` | Runtime/workflow logging optionality can weaken forensic guarantees when not running gates. | medium | `bd-23m.6` | Require explicit log path and gate-level assertion of produced log artifacts in orchestrator flows. |
| `MODEL-C003` | Packet-local state models exist for iter/random/linalg/io, but those models are not yet fully exercised by packet-local differential and workflow replay gates. | high | `bd-23m.17`/`18`/`19`/`20` | Wire packet-local differential/e2e suites into gates and require packet-level readiness artifacts before treating models as parity-complete. |
| `MODEL-C004` | Cast-matrix and transfer/alias invariants are not yet represented as first-class state contracts in code. | high | `bd-23m.13`/`bd-23m.14` | Add explicit state/invariant tables and executable fixtures for cast matrix and overlap/alias transitions. |

## DOC-PASS-04 Execution-Path Tracing and Control-Flow Narratives

### DOC-PASS-04.1 Canonical execution workflows (entry -> branch -> terminal)

| Workflow ID | Entry anchor | Primary branch points (ordered) | Terminal outcomes | Legacy/evidence anchors | Ownership |
|---|---|---|---|---|---|
| `EP-001` ufunc differential | `crates/fnp-conformance/src/bin/run_ufunc_differential.rs` | (1) fixture/oracle load success; (2) dtype parse success; (3) `UFuncArray::new` shape/value gate; (4) op dispatch (`add/sub/mul/div/sum`) | pass report / mismatch report / fixture parse failure | `crates/fnp-conformance/src/ufunc_differential.rs`, `fixtures/ufunc_input_cases.json`, `fixtures/oracle_outputs/ufunc_oracle_output.json` | `bd-23m.16`, `bd-23m.17` |
| `EP-002` security gate | `crates/fnp-conformance/src/bin/run_security_gate.rs` | (1) runtime policy suite; (2) adversarial policy suite; (3) security contract suite; (4) gate aggregate pass/fail | gate pass / gate fail with reason taxonomy | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, `security_contracts::run_security_contract_suite` | foundation `bd-23m.6`, packet owners |
| `EP-003` workflow scenario replay | `crates/fnp-conformance/src/bin/run_workflow_scenario_gate.rs` | (1) scenario fixture parse; (2) strict/hardened mode execution; (3) expectation comparison; (4) scenario log append | scenario pass / deterministic failure record | `workflow_scenarios::run_user_workflow_scenario_suite`, `artifacts/logs/workflow_scenario_e2e_*.jsonl` | foundation + packet G beads |
| `EP-004` packet readiness contract | `crates/fnp-conformance/src/bin/validate_phase2c_packet.rs` | (1) required file presence; (2) schema token checks; (3) JSON/YAML parse checks; (4) readiness summarization | ready / non-ready with missing-field diagnostics | `contract_schema::validate_phase2c_packet`, `packet_readiness_report.json` | packet I beads |
| `EP-005` IO load dispatch | `crates/fnp-io/src/lib.rs` (`classify_load_dispatch`) | (1) NPZ magic check; (2) NPY magic check; (3) pickle branch only if `allow_pickle`; else fail-closed | `LoadDispatch::{Npz,Npy,Pickle}` / `IOError::LoadDispatchInvalid` | `numpy/lib/format.py`, `IO_PACKET_REASON_CODES` | `bd-23m.20` |
| `EP-006` RNG bounded generation | `crates/fnp-random/src/lib.rs` (`bounded_u64`) | (1) upper-bound non-zero; (2) rejection-threshold loop; (3) modulo projection | bounded sample / `RandomError::InvalidUpperBound` | `numpy/random/_generator.pyx`, `RANDOM_PACKET_REASON_CODES` | `bd-23m.18` |
| `EP-007` linalg branch family | `crates/fnp-linalg/src/lib.rs` (`qr_output_shapes`, `svd_output_shapes`, `validate_policy_metadata`) | (1) rank/shape preflight; (2) mode token parse; (3) convergence gate; (4) fail-closed unknown metadata | shape tuple outputs / `LinAlgError::*` fail-closed terminals | `numpy/linalg/lapack_lite/*`, `LINALG_PACKET_REASON_CODES` | `bd-23m.19` |
| `EP-008` transfer selector and overlap policy | `crates/fnp-iter/src/lib.rs` (`select_transfer_class`, `overlap_copy_policy`, flatiter validators) | (1) context sanity checks; (2) stride-multiple checks; (3) overlap direction choice; (4) read/write arity checks | transfer class + copy action / `TransferError::*` | `dtype_transfer.c`, `lowlevel_strided_loops.c.src`, `TRANSFER_PACKET_REASON_CODES` | `bd-23m.14`, `bd-23m.17` |

### DOC-PASS-04.2 Branch ordering and fallback law (deterministic precedence)

| Branch family | Deterministic ordering rule | Fallback / fail-closed behavior | Why ordering matters |
|---|---|---|---|
| IO payload dispatch (`EP-005`) | Evaluate NPZ magic first, then NPY magic, then pickle (only with explicit allow), else reject | Unknown/forbidden prefixes always map to `io_load_dispatch_invalid` | Prevent ambiguous decode paths and policy bypass via permissive probing. |
| Runtime mode/class decode (`EP-002`, `EP-003`) | Normalize wire mode/class; unknown tokens do not branch into permissive defaults | Unknown metadata maps to fail-closed action and logged reason code | Preserves strict/hardened doctrine under malformed metadata. |
| Ufunc case execution (`EP-001`) | Validate fixture + shape/value consistency before numeric op dispatch | Any precondition failure short-circuits with deterministic mismatch/error output | Avoids undefined behavior from malformed fixtures and preserves oracle comparability. |
| RNG bounded sampling (`EP-006`) | Reject invalid bound first; only then run rejection sampling loop | `upper_bound == 0` hard-fails with stable reason code | Prevents silent modulo-by-zero style divergence and keeps replay deterministic. |
| Linalg mode/convergence (`EP-007`) | Shape/rank preflight -> mode token parse -> convergence predicate | Non-convergence/invalid mode/unknown metadata fail with explicit packet reason codes | Prevents accidental fallback to undefined solver branches. |
| Transfer overlap and flatiter writes (`EP-008`) | Validate context -> determine overlap direction -> enforce index/value arity | Contract violation yields stable transfer reason code and no write admission | Protects alias-sensitive transfer semantics and write-lane safety. |

### DOC-PASS-04.3 Coverage implications by workflow

| Workflow ID | Unit/Property | Differential | E2E | Structured logging | Assessment |
|---|---|---|---|---|---|
| `EP-001` | covered (ufunc unit/metamorphic/adversarial suites) | covered for scoped ops | partial via workflow corpus | covered at fixture/scenario level | partial parity debt outside scoped op families |
| `EP-002` | covered (runtime policy tests) | covered (policy-wire adversarial classes) | covered in gate execution | covered (required contract fields) | strong for policy surface |
| `EP-003` | covered by scenario tests | N/A (orchestration path) | covered | covered when log path configured | medium operational risk due optional log path configuration |
| `EP-004` | covered by contract-schema tests | N/A | covered via packet validator binary | readiness reports are structured outputs | strong for artifact topology checks |
| `EP-005` | covered (io contract tests) | missing packet-local oracle differential | missing packet-local replay lane | covered by `IOLogRecord` schema tests | partial, gate integration still pending |
| `EP-006` | covered (determinism/bounded/range/state tests) | missing packet-local oracle differential | missing packet-local replay lane | covered by `RandomLogRecord` schema tests | partial, gate integration still pending |
| `EP-007` | covered (linalg contract tests) | missing packet-local oracle differential | missing packet-local replay lane | covered by `LinAlgLogRecord` schema tests | partial, gate integration still pending |
| `EP-008` | covered (transfer/index contract tests) | missing packet-local oracle differential | missing packet-local replay lane | covered by `TransferLogRecord` schema tests | partial, gate integration still pending |

### DOC-PASS-04.4 Control-flow contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `FLOW-C001` | Packet-local execution paths (`EP-005`..`EP-008`) have deterministic contract tests but are not yet first-class suites in `run_all_core_suites`. | high | `bd-23m.18`, `bd-23m.19`, `bd-23m.20`, `bd-23m.14`/`17` | Add packet-local suites and include them in gate orchestrators. |
| `FLOW-C002` | Workflow/security log appenders are path-config dependent; absent paths can downgrade forensic coverage despite successful test execution. | medium | foundation `bd-23m.6` | Make log path configuration mandatory in gate invocations and fail when expected logs are absent. |
| `FLOW-C003` | Differential harness precedence is ufunc-centric; equivalent branch-level oracle replay is missing for random/linalg/io/transfer paths. | high | packet F-stage beads (`bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6`, `bd-23m.14.6`) | Implement packet-local oracle differentials with per-branch reason-code reporting. |
| `FLOW-C004` | Packet readiness validation (`EP-004`) checks artifact presence/schema but cannot prove runtime branch reachability without correlated replay logs. | medium | packet I beads + doc pass-10 | Add branch-reachability evidence links (`fixture_id`,`reason_code`) to readiness report conventions. |

## DOC-PASS-05 Complexity, Performance, and Memory Characterization

### DOC-PASS-05.1 Complexity classes by canonical execution path

| Workflow ID | Dominant operations | Time complexity (current implementation) | Space complexity (current implementation) | Anchors |
|---|---|---|---|---|
| `EP-001` ufunc differential | fixture parse + per-case execution + oracle compare | `O(C * N)` where `C` is case count and `N` is per-case element count | `O(N)` per case for output buffers + report accumulation | `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-ufunc/src/lib.rs` |
| `EP-002` security gate | finite policy/security suite execution | `O(C)` over configured security/policy fixtures | `O(C)` for suite report/log aggregation | `run_security_gate.rs`, runtime suite functions in `crates/fnp-conformance/src/lib.rs` |
| `EP-003` workflow scenario replay | scenario script execution and expectation checks | `O(S * K)` where `S` is scenario count and `K` is steps per scenario | `O(S + log_entries)` | `workflow_scenarios.rs`, `run_workflow_scenario_gate.rs` |
| `EP-004` packet readiness validation | file existence/schema parse checks | `O(F)` where `F` is required artifact count | `O(F)` for missing-field/readiness structures | `contract_schema.rs`, `validate_phase2c_packet.rs` |
| `EP-005` IO dispatch/validation | magic checks, header/shape validation, budget checks | dispatch is `O(1)`; shape/header checks are `O(rank)`; payload footprint checks are `O(1)` arithmetic | `O(rank)` for copied shape/header descriptors | `crates/fnp-io/src/lib.rs` |
| `EP-006` RNG deterministic generation | splitmix step + optional rejection loop | `next_u64`/`next_f64` are `O(1)`; `bounded_u64` has expected `O(1)` with probabilistic retries | `O(1)` per sample; `fill_u64(len)` is `O(len)` | `crates/fnp-random/src/lib.rs` |
| `EP-007` linalg branch family | shape/mode validation + small-kernel helpers | validators are `O(rank)` or `O(1)`; `solve_2x2` is `O(1)` | mostly `O(1)`; shape outputs are `O(rank)` | `crates/fnp-linalg/src/lib.rs` |
| `EP-008` transfer/index legality | selector/overlap checks + index scan | selector/overlap are `O(1)`; fancy/mask index validation is `O(index_len)`/`O(len)` | `O(1)` except `Fancy`/`BoolMask` index storage already provided by caller | `crates/fnp-iter/src/lib.rs` |

### DOC-PASS-05.2 Memory-growth and allocation behavior

| Path family | Allocation model | Growth driver | Boundedness controls | Residual risk |
|---|---|---|---|---|
| `fnp-ufunc` elementwise/reduction | allocates output value vectors; keeps input vectors immutable | output element count and dtype width | shape legality and element-count admission checks | high memory pressure for very large broadcasted outputs if fixture caps are weak |
| `fnp-ndarray` layout calculus | shape/stride vectors and small metadata structs | rank and reshape/broadcast rank expansions | overflow checks + rank-aware validation | large-rank adversarial inputs still require explicit fixture budgets |
| `fnp-conformance` reports/artifacts | suite reports + mismatch vectors + artifact JSON outputs | fixture corpus size and number of gate runs | harness config + packet contract validation | artifact accumulation can grow quickly without retention policy |
| `fnp-random` stream generation | constant-state RNG; vector allocation only in `fill_u64` | requested fill length | caller-controlled `len`; deterministic state model | large requested fills can dominate memory unless bounded by caller/gate fixture contracts |
| `fnp-linalg` contract checks | mostly scalar validation and small shape vectors | matrix rank metadata and output-shape tuples | bounded shape checks (`MAX_BATCH_SHAPE_CHECKS`, etc.) | full-matrix numerical kernels (future packet stages) will alter memory profile materially |
| `fnp-io` parser/dispatch contracts | shape/header copies, archive-member name vectors | shape rank, archive member count, metadata size | `MAX_HEADER_BYTES`, `MAX_ARCHIVE_MEMBERS`, byte-budget constants | decoded payload memory growth requires additional end-to-end budget enforcement in replay gates |
| Transfer/index validation (`fnp-iter`) | mostly zero-allocation checks, except caller-provided index structures | mask/fancy index lengths | strict contract validation and fail-fast checks | large index vectors still impose caller-side memory pressure |

### DOC-PASS-05.3 Hotspot inventory and optimization-governance readiness

| Hotspot family | Why hotspot | Baseline/profile artifact status | EV/one-lever governance status | Owner |
|---|---|---|---|---|
| Broadcasted ufunc loops | `O(N)` numeric kernel path across many fixtures | baseline present (`artifacts/baselines/ufunc_benchmark_baseline.json`) | governance scaffold present; continued per-lever isomorphism artifacts required | `bd-23m.16` |
| Differential harness orchestration | multiplies execution cost by fixture corpus size and oracle bridge overhead | partial (differential report artifacts present; profiler artifacts not standardized) | needs standardized profile capture attachment in packet H/I outputs | packet F/H beads |
| Runtime/security/workflow gates | repeated suite execution with log + summary generation | partial (gate summaries/logs exist; cross-gate cost dashboards absent) | add periodic gate-cost snapshots and budget alarms | foundation `bd-23m.8` |
| RNG/linalg/io/transfer packet-local paths | emerging hotspots as packet-local suites scale | missing dedicated benchmark baselines | EV gate cannot be applied consistently until per-path baselines/profiles exist | `bd-23m.14`/`18`/`19`/`20` |
| Artifact durability pipeline | sidecar/scrub/decode proofs scale with bundle size/count | partial (sidecar/scrub/decode artifacts generated; throughput trends not tracked) | needs throughput and storage-growth baselines in optimization reports | packet I + durability owners |

### DOC-PASS-05.4 Verification and logging implications for performance changes

| Domain | Unit/Property guardrails | Differential guardrails | E2E guardrails | Logging guardrails | Coverage status |
|---|---|---|---|---|---|
| Ufunc hotspot optimization | covered (ufunc unit/metamorphic/adversarial) | covered for scoped corpus | partial | covered | partial outside scoped ops |
| Runtime gate optimization | covered | policy differential equivalent covered | covered | covered | strong |
| RNG/linalg/io/transfer optimization | covered at unit/property level | missing packet-local differentials | missing packet-local replay lanes | schema-level only | missing/partial |
| Artifact pipeline optimization | contract and durability checks covered | N/A | gate runs available | artifact metadata logged | partial (throughput trend logs missing) |

### DOC-PASS-05.5 Complexity/performance contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `PERF-C001` | EV-gated optimization doctrine is documented, but non-ufunc packet paths lack standardized benchmark baselines/profiler artifacts. | high | `bd-23m.8`, packet H beads | Add baseline/profile artifacts for iter/random/linalg/io paths and require them in packet optimization reports. |
| `PERF-C002` | `bounded_u64` uses rejection sampling with no explicit retry cap; expected `O(1)` but worst-case retries are unbounded. | medium | `bd-23m.18` | Document/implement bounded retry policy (or explicit rationale) and add adversarial witness tests. |
| `PERF-C003` | Gate/logging costs are observable per run but not tracked as longitudinal metrics, making regressions hard to detect early. | medium | foundation `bd-23m.8` | Emit periodic gate-cost trend artifacts with environment fingerprints and threshold alerts. |
| `PERF-C004` | Packet-local memory budgets are encoded in crate constants, but cross-packet aggregate memory envelope is undocumented. | medium | `bd-23m.24.6` follow-on + packet owners | Define and publish cross-packet memory budget matrix with enforcement hooks in gate scripts. |

## 2. Quantitative Legacy Inventory (Measured)

- Total files: `2326`
- Python: `493`
- Native: `c=180`, `cpp=29`, `h=227`, `hpp=16`
- Cython: `pyx=13`, `pxd=6`
- Test-like files: `530`

High-density zones:
- `numpy/_core/src` (402 files)
- `numpy/typing/tests` (149)
- `numpy/f2py/tests` (132)
- `numpy/_core/tests` (100)
- `numpy/linalg/lapack_lite` (26)
- `numpy/random/src` (37)

## 3. Subsystem Extraction Matrix (Legacy -> Rust)

| Legacy locus | Non-negotiable behavior to preserve | Target crates | Primary oracles | Phase-2 extraction deliverables |
|---|---|---|---|---|
| `numpy/_core/src/multiarray/shape.c` + `shape.h` | shape legality, stride legality, reshape semantics | `fnp-ndarray` | `_core/tests/test_shape_base.py`, `test_nditer.py` | shape/stride state machine, invalid-transition matrix |
| `dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | promotion and cast determinism | `fnp-dtype` | `_core/tests/test_dtype.py`, `test_casting*` | promotion matrix, cast allow/deny table |
| `dtype_transfer.c`, `lowlevel_strided_loops.c.src` | assignment loop correctness under strides | `fnp-iter`, `fnp-ufunc` | `_core/tests/test_mem_overlap.py` | alias/overlap safety matrix, loop invariants |
| `umath/ufunc_object.c`, dispatch loop sources | ufunc dispatch and broadcasting precedence | `fnp-ufunc` | `_core/tests/test_ufunc.py`, `test_overrides.py` | dispatch precedence graph, override semantics spec |
| `multiarray/nditer*` | iterator consistency across order/layout | `fnp-iter` | `_core/tests/test_nditer.py` | traversal equivalence ledger |
| `numpy/random/*.pyx` + `random/src/*` | RNG determinism and state serialization | `fnp-random` | `random/tests/*` | RNG stream fixture corpus + state schema |
| `numpy/linalg/lapack_lite/*` | solver shape/tolerance/return semantics | `fnp-linalg` | `linalg/tests/test_linalg.py` | BLAS/LAPACK boundary contract table |
| `numpy/lib/stride_tricks.py` + IO paths | view tricks and npy/npz metadata semantics | `fnp-ndarray`, `fnp-io` | `lib/tests/test_stride_tricks.py`, IO tests | metadata parser/state machine spec |

## 4. Alien-Artifact Invariant Ledger (Formal Obligations)

- `FNP-I1` Shape legality closure: all produced array states satisfy dimension/stride legality.
- `FNP-I2` Promotion determinism: scoped dtype pairs map to a single canonical result dtype.
- `FNP-I3` Broadcast determinism: result shape is deterministic and independent of iteration schedule.
- `FNP-I4` Alias safety: write operations do not violate overlap semantics.
- `FNP-I5` RNG reproducibility: fixed seed and algorithm produce stable output stream.

Required proof artifacts per implemented slice:
1. invariant statement,
2. executable witness fixtures,
3. failing counterexample archive (if any),
4. remediation proof.

## 5. Native/C/Fortran Boundary Register

| Boundary | Files | Risk | Mandatory mitigation |
|---|---|---|---|
| ndarray metadata core | `multiarray/*.c` | critical | legality checks first, then loop execution |
| ufunc dispatch core | `umath/ufunc_object.c` | high | precedence fixtures + override differential tests |
| lapack-lite bridge | `linalg/lapack_lite/*` | high | solver IO/tolerance conformance fixtures |
| RNG backend | `random/src/*`, `random/*.pyx` | high | deterministic stream and state round-trip checks |

## 6. Compatibility and Security Doctrine (Mode-Split)

Decision law (runtime):
`mode + metadata_contract + risk_score + budget -> allow | full_validate | fail_closed`

| Threat | Strict mode | Hardened mode | Required ledger artifact |
|---|---|---|---|
| malformed npy/npz metadata | fail-closed | fail-closed with bounded diagnostics | parser incident ledger |
| shape bomb payload | execute in scoped limits | tighter admission caps | admission guard report |
| unsafe cast attempt | reject unscoped cast | allow only policy-allowlisted cast | cast decision ledger |
| stride alias confusion | reject invalid transitions | reject invalid transitions | stride invariant report |
| unknown incompatible metadata version | fail-closed | fail-closed | compatibility drift report |

## 7. Conformance Program (Exhaustive First Wave)

### 7.1 Fixture families

1. Shape/reshape fixtures
2. Stride/view and overlap fixtures
3. Broadcast fixtures
4. Ufunc arithmetic/reduction fixtures
5. Dtype promotion fixtures
6. RNG determinism fixtures
7. NPY/NPZ IO fixtures

### 7.2 Differential harness outputs (fnp-conformance)

Each run emits:
- machine-readable parity report,
- mismatch class histogram,
- minimized repro fixture bundle,
- strict/hardened divergence report.

Release gate rule: critical-family drift => hard fail.

## 8. Extreme Optimization Program

Primary hotspots:
- ufunc dispatch path
- non-contiguous iteration loops
- dtype conversion hot loops
- reduction kernels

Budgets (from spec section 17):
- broadcast p95 <= 180 ms
- reductions p95 <= 210 ms
- reshape/view p95 <= 40 ms
- npy parse throughput >= 400 MB/s
- p99 regression <= +7%, peak RSS regression <= +8%

Optimization governance:
1. baseline,
2. profile,
3. single optimization lever,
4. conformance proof,
5. budget gate,
6. evidence commit.

## 9. RaptorQ-Everywhere Artifact Contract

Durable artifacts requiring RaptorQ sidecars:
- conformance fixture bundles,
- benchmark baselines,
- promotion/cast ledgers,
- risk/proof ledgers.

Required envelope fields:
- source hash,
- symbol manifest,
- scrub status,
- decode proof chain.

## 10. Phase-2 Execution Backlog (Concrete)

1. Extract shape/stride legality rules from `shape.c`.
2. Extract promotion/cast tables from dtype sources.
3. Extract overlap/alias rules from low-level loop code.
4. Extract ufunc dispatch precedence and override rules.
5. Extract NDIter traversal invariants.
6. Extract RNG state schema and deterministic stream behavior.
7. Extract linalg solver input/output and tolerance behavior.
8. Extract npy/npz metadata parsing state machine.
9. Build first differential fixture corpus for items 1-8.
10. Implement mismatch taxonomy in `fnp-conformance`.
11. Add strict/hardened divergence reporting.
12. Attach RaptorQ sidecar generation and decode-proof validation.

Definition of done for Phase-2:
- each row in section 3 has extraction artifacts,
- all seven fixture families are runnable,
- G1-G6 gates from comprehensive spec map to concrete harness outputs.

## 11. Residual Gaps and Risks

- `PROPOSED_ARCHITECTURE.md` crate list contains literal `\n` separators; normalize before automation.
- high-risk native boundaries (multiarray/umath/lapack-lite) require wider differential corpus before aggressive optimization.
- RNG and cast semantics are common silent-regression vectors and require explicit release blockers.

## 12. Deep-Pass Hotspot Inventory (Measured)

Measured from `/data/projects/franken_numpy/legacy_numpy_code/numpy`:
- file count: `2326`
- highest concentration: `numpy/_core` (`604` files), `numpy/f2py` (`177`), `numpy/typing` (`152`), `numpy/random` (`101`), `numpy/lib` (`97`)

Top source hotspots by line count (first-wave extraction anchors):
1. `numpy/linalg/lapack_lite/f2c_d_lapack.c` (`41864`)
2. `numpy/linalg/lapack_lite/f2c_s_lapack.c` (`41691`)
3. `numpy/linalg/lapack_lite/f2c_z_lapack.c` (`29996`)
4. `numpy/linalg/lapack_lite/f2c_c_lapack.c` (`29861`)
5. `numpy/linalg/lapack_lite/f2c_blas.c` (`21603`)
6. `numpy/_core/src/umath/ufunc_object.c` (`6796`)

Interpretation:
- low-level shape/ufunc/cast paths must be extracted before high-level wrappers,
- lapack-lite bridge dominates numerical backend complexity,
- iterator/transfer semantics remain highest silent-regression risk.

## 13. Phase-2C Extraction Payload Contract (Per Ticket)

Each `FNP-P2C-*` ticket MUST produce:
1. type inventory (dtype/array/iterator structs and fields),
2. shape-stride legality rules with exact conditions,
3. cast/promotion rule tables (including defaults),
4. error/diagnostic contract map,
5. alias/overlap/memory-order behavior ledger,
6. strict/hardened mode split,
7. explicit exclusion ledger,
8. fixture mapping manifest,
9. optimization candidate notes + isomorphism risk,
10. RaptorQ artifact declarations.

Artifact location (normative):
- `artifacts/phase2c/FNP-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FNP-P2C-00X/contract_table.md`
- `artifacts/phase2c/FNP-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FNP-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FNP-P2C-00X/risk_note.md`

### 13.1 Packet `FNP-P2C-006` extraction status (`A` stage)

Completed in this pass:
- `artifacts/phase2c/FNP-P2C-006/legacy_anchor_map.md`
- `artifacts/phase2c/FNP-P2C-006/behavior_extraction_ledger.md`
- `artifacts/phase2c/FNP-P2C-006/contract_table.md`
- `artifacts/phase2c/FNP-P2C-006/risk_note.md`
- `artifacts/phase2c/FNP-P2C-006/implementation_plan.md`

Anchor families locked for packet-A:
- `numpy/lib/_stride_tricks_impl.py` (`as_strided`, `_broadcast_to`, `broadcast_to`, `_broadcast_shape`, `broadcast_shapes`, `broadcast_arrays`)
- `numpy/_core/src/multiarray/nditer_constr.c` (`npyiter_fill_axisdata`, `broadcast_error`, `operand_different_than_broadcast`)
- `numpy/_core/src/multiarray/nditer_api.c` (`NpyIter_GetShape`, `NpyIter_CreateCompatibleStrides`)
- oracle tests: `numpy/lib/tests/test_stride_tricks.py`, `numpy/_core/tests/test_nditer.py`

Verification hooks registered:
- unit/property expansion target: `bd-23m.17.5`
- differential/metamorphic/adversarial expansion target: `bd-23m.17.6`
- e2e replay/logging target: `bd-23m.17.7`
- strict/hardened threat closure target: `bd-23m.17.3`

Contract-table status:
- strict/hardened invariants, pre/postconditions, failure semantics, and packet reason-code vocabulary are defined in `artifacts/phase2c/FNP-P2C-006/contract_table.md`
- every contract row links planned unit/property IDs, differential IDs, and e2e IDs for packet follow-on stages

Threat-model status:
- hostile input classes, fail-closed envelope, hardened recovery constraints, adversarial fixture lanes, and residual-risk closures are documented in `artifacts/phase2c/FNP-P2C-006/risk_note.md`

Implementation-plan status:
- crate/module boundary skeleton, packet D->I implementation sequence, instrumentation insertion points, and compile-safe validation path are documented in `artifacts/phase2c/FNP-P2C-006/implementation_plan.md`

## 14. Strict/Hardened Compatibility Drift Budgets

Packet acceptance budgets:
- strict mode critical drift budget: `0`
- strict mode non-critical drift budget: `<= 0.10%`
- hardened mode divergence budget: `<= 1.00%` and only allowlisted defensive cases
- unknown dtype/layout/metadata behavior: fail-closed

Required per-packet report fields:
- `strict_parity`,
- `hardened_parity`,
- `divergence_classes`,
- `cast_drift_summary`,
- `compatibility_drift_hash`.

## 15. Extreme-Software-Optimization Execution Law

Mandatory loop:
1. baseline,
2. profile,
3. one lever,
4. conformance + invariant proof,
5. re-baseline.

Primary sentinel workloads:
- reshape/broadcast-heavy traces (`FNP-P2C-001`, `FNP-P2C-006`),
- mixed-dtype cast pipelines (`FNP-P2C-002`, `FNP-P2C-003`),
- ufunc dispatch stress (`FNP-P2C-005`),
- RNG reproducibility workloads (`FNP-P2C-007`).

Optimization scoring gate:
`score = (impact * confidence) / effort`, merge only if `score >= 2.0`.

## 16. RaptorQ Evidence Topology and Recovery Drills

Durable artifacts requiring sidecars:
- parity reports,
- fixture corpora,
- cast/promotion ledgers,
- benchmark baselines,
- strict/hardened divergence logs.

Naming convention:
- payload: `packet_<id>_<artifact>.json`
- sidecar: `packet_<id>_<artifact>.raptorq.json`
- proof: `packet_<id>_<artifact>.decode_proof.json`

Decode-proof failure policy: hard blocker for packet promotion.

## 17. Phase-2C Exit Checklist (Operational)

Phase-2C is complete only when:
1. `FNP-P2C-001..009` artifacts exist and pass schema validation.
2. Every packet has strict and hardened fixture coverage.
3. Drift budgets from section 14 are met.
4. At least one hotspot optimization proof exists for each high-risk packet.
5. RaptorQ sidecars + decode proofs are scrub-clean.
6. Remaining risks are explicit with owners and next actions.
