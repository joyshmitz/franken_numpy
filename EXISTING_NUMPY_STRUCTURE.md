# EXISTING_NUMPY_STRUCTURE

## DOC-PASS-00 Baseline Gap Matrix + Quantitative Expansion Targets

Snapshot baseline (2026-02-14):

| Document | Baseline lines | Target lines | Expansion multiplier | Status |
|---|---:|---:|---:|---|
| `EXISTING_NUMPY_STRUCTURE.md` | 62 | 992 | 16.0x | baseline pass complete (`bd-23m.24.1` closed) |
| `EXHAUSTIVE_LEGACY_ANALYSIS.md` | 275 | 3300 | 12.0x | baseline pass complete (`bd-23m.24.1` closed) |

Gap matrix for this document (pass-1 planning):

| Area | Current state | Missing for parity-grade docs | Coverage implications to record |
|---|---|---|---|
| Subsystem map detail | coarse bullets | per-package ownership + boundary invariants + packet IDs | unit/property + differential + e2e + structured logging per subsystem |
| Semantic hotspots | short list | explicit legality formulas and tie-break rules | metamorphic/property families and adversarial classes |
| Compatibility-critical behavior | high-level statements | strict/hardened split with drift budget and fail-closed triggers | runtime-policy reason-code matrix and replay fields |
| Risk areas | broad categories | threat-to-control-to-test mappings | parser fuzz corpus, crash triage IDs, forensic log fields |
| Conformance families | list only | fixture IDs, oracle anchors, and closure gates | coverage ledger with covered/missing/deferred status |

Pass-1 explicit coverage/traceability matrix (covered, missing, deferred):

| Subsystem | Legacy anchors | Executable evidence anchors | Unit/Property | Differential | E2E | Structured logging |
|---|---|---|---|---|---|---|
| Shape/stride | `numpy/_core/src/multiarray/shape.c`, `shape.h` | `crates/fnp-conformance/fixtures/shape_stride_cases.json` | covered (partial) | covered (partial via ufunc diff) | missing | covered (partial via runtime-policy logs) |
| Dtype promotion/cast | `numpy/_core/src/multiarray/dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | `crates/fnp-conformance/fixtures/dtype_promotion_cases.json` | covered (partial) | missing (cast-matrix diff missing) | missing | missing |
| Ufunc dispatch | `numpy/_core/src/umath/ufunc_object.c` | `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/fixtures/ufunc_*` | covered (partial metamorphic/adversarial) | covered (partial) | missing | covered (fixture-level fields) |
| Transfer/alias | `numpy/_core/src/multiarray/dtype_transfer.c`, `lowlevel_strided_loops.c.src` | none yet (legacy only) | missing | missing | missing | missing |
| NDIter / transfer semantics | `numpy/_core/src/multiarray/nditer*`, `dtype_transfer.c`, `lowlevel_strided_loops.c.src` | `crates/fnp-iter/src/lib.rs`, packet docs in `artifacts/phase2c/FNP-P2C-003/*` and `FNP-P2C-006/*` | covered (unit/property contract tests) | missing packet-local differential lane | missing packet-local journey | covered at record-schema level (not yet gate-enforced) |
| Random | `numpy/random/*.pyx`, `numpy/random/src/*` | `crates/fnp-random/src/lib.rs`, packet docs in `artifacts/phase2c/FNP-P2C-007/*` | covered (deterministic/state/bounded tests) | missing packet-local differential lane | missing seed/state journey | covered at record-schema level (not yet gate-enforced) |
| Linalg | `numpy/linalg/lapack_lite/*` | `crates/fnp-linalg/src/lib.rs`, packet docs in `artifacts/phase2c/FNP-P2C-008/*` | covered (shape/solver/policy contract tests) | missing packet-local differential lane | missing solver journey | covered at record-schema level (not yet gate-enforced) |
| IO | `numpy/lib/format.py`, npy/npz handling paths | `crates/fnp-io/src/lib.rs`, packet docs in `artifacts/phase2c/FNP-P2C-009/*` | covered (parser/budget/policy contract tests) | missing packet-local differential lane | missing io journey | covered at record-schema level (not yet gate-enforced) |

Traceability anchors:

- Legacy source roots: `/data/projects/franken_numpy/legacy_numpy_code/numpy`
- Machine contracts: `artifacts/contracts/test_logging_contract_v1.json`, `artifacts/contracts/phase2c_contract_schema_v1.json`
- Gate artifacts: `crates/fnp-conformance/src/test_contracts.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `scripts/e2e/run_test_contract_gate.sh`

Contradictions/unknowns register (for closure in doc passes 01-10):

| ID | Item | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `DOC-C001` | Section 6 carries historical reduced-scope language incompatible with full drop-in parity doctrine. | critical | `bd-23m.24.2` | Replace with parity-debt sequencing model and explicit owner/blocker table. |
| `DOC-C002` | Verification/logging implications are now mapped in DOC-PASS-03.4, but several packet domains remain explicitly missing/deferred. | high | `bd-23m.24.4` | Keep DOC-PASS-03.4 synchronized with packet implementation progress until missing lanes are closed. |
| `DOC-C003` | Ownership for NDIter/RNG/linalg/IO extraction is mapped and packet-local APIs now exist, but those APIs are underrepresented in conformance gate execution and packet F/G evidence. | high | `bd-23m.24.3` | Keep ownership/gate mapping synchronized until packet-local differential + replay suites are integrated and packet closure artifacts are complete. |

## DOC-PASS-01 Full Module/Package Cartography with Ownership and Boundaries

### DOC-PASS-01.1 Workspace ownership map (crate -> behavior contract)

| Crate | Ownership boundary (Rust anchors) | Legacy anchor families | Executable evidence anchors | Boundary contract |
|---|---|---|---|---|
| `fnp-dtype` | `DType`, `promote`, `can_cast_lossless` (`crates/fnp-dtype/src/lib.rs`) | `dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | `dtype_promotion_cases.json`, `run_dtype_promotion_suite` | Owns deterministic dtype identity/promotion/cast primitives only. |
| `fnp-ndarray` | `broadcast_shape`, `broadcast_shapes`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout` (`crates/fnp-ndarray/src/lib.rs`) | `shape.c`, `shape.h`, stride/broadcast legality paths | `shape_stride_cases.json`, `run_shape_stride_suite` | Owns shape/stride legality and layout calculus only. |
| `fnp-ufunc` | `UFuncArray`, `BinaryOp`, `elementwise_binary`, `reduce_sum` (`crates/fnp-ufunc/src/lib.rs`) | `umath/ufunc_object.c` dispatch/reduction families | `ufunc_differential.rs`, `run_ufunc_*` suites | Owns scoped ufunc execution semantics. |
| `fnp-runtime` | strict/hardened decision APIs + `EvidenceLedger` (`crates/fnp-runtime/src/lib.rs`) | strict/hardened policy doctrine and fail-closed matrix | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, workflow scenarios | Owns compatibility policy decisions and audit records. |
| `fnp-conformance` | suite runners + artifact/security/workflow modules (`crates/fnp-conformance/src/lib.rs`, `src/bin/*.rs`) | legacy oracle tests and packet artifact topology contracts | gate binaries + packet readiness reports + sidecar/scrub/decode proof artifacts | Owns verification/evidence orchestration and contract enforcement. |
| `fnp-iter` | transfer/index APIs (`select_transfer_class`, overlap/flatiter validators, `TransferLogRecord`) (`crates/fnp-iter/src/lib.rs`) | `dtype_transfer.c`, `lowlevel_strided_loops.c.src`, `nditer*` | crate unit/property suites + `artifacts/phase2c/FNP-P2C-003/*` | Owns first-wave transfer/overlap/flatiter contract checks. |
| `fnp-random` | `DeterministicRng`, bounded/float/state APIs, `RandomLogRecord` (`crates/fnp-random/src/lib.rs`) | `numpy/random/_generator.pyx`, `numpy/random/src/*` | crate unit/property suites + `artifacts/phase2c/FNP-P2C-007/*` | Owns first-wave deterministic stream/state contract checks. |
| `fnp-linalg` | shape/solver/qr/svd/lstsq/policy validators + `LinAlgLogRecord` (`crates/fnp-linalg/src/lib.rs`) | `numpy/linalg/lapack_lite/*`, wrappers in `numpy/linalg/*` | crate unit/property suites + `artifacts/phase2c/FNP-P2C-008/*` | Owns first-wave linalg boundary and policy contract checks. |
| `fnp-io` | magic/header/descriptor/memmap/load/npz validators + `IOLogRecord` (`crates/fnp-io/src/lib.rs`) | `numpy/lib/format.py`, npy/npz paths | crate unit/property suites + `artifacts/phase2c/FNP-P2C-009/*` | Owns first-wave NPY/NPZ parser/writer boundary and policy checks. |

### DOC-PASS-01.2 Dependency direction and layering constraints

Dependency direction (from Cargo manifests):

1. Foundation semantics: `fnp-dtype`, `fnp-ndarray`.
2. Packet-local contract layers: `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`.
3. Execution layer: `fnp-ufunc` depends on `fnp-dtype` + `fnp-ndarray`.
4. Policy layer: `fnp-runtime` (optional runtime integrations).
5. Verification/gate layer: `fnp-conformance` depends on semantic/execution/policy crates and owns binaries.

Layering constraints for this pass:

- Semantic crates must not depend on conformance/tooling crates.
- Packet-local crates must remain policy-neutral (no runtime decision coupling in core contract logic).
- Conformance binaries are orchestration-only and must not redefine numerical semantics.
- Any new dependency edge that inverts this direction is a gate-level defect.

### DOC-PASS-01.3 Module and binary entrypoint cartography

| Path | Role class | Topology implication |
|---|---|---|
| `crates/fnp-ndarray/src/lib.rs` | SCE legality kernel | Broadcast/reshape/stride legality funnel. |
| `crates/fnp-ufunc/src/lib.rs` | Execution kernel | Broadcasted binary/reduction operation semantics. |
| `crates/fnp-runtime/src/lib.rs` | Policy kernel | strict/hardened fail-closed decisioning and ledger recording. |
| `crates/fnp-iter/src/lib.rs` | Transfer/index contract kernel | overlap/broadcast/flatiter legality for transfer semantics. |
| `crates/fnp-random/src/lib.rs` | RNG contract kernel | deterministic constructor/state/stream/bounded sampling semantics. |
| `crates/fnp-linalg/src/lib.rs` | Linalg contract kernel | shape/solver/mode/backend policy contract enforcement. |
| `crates/fnp-io/src/lib.rs` | IO boundary kernel | NPY/NPZ parser/dispatch/budget and fail-closed policy checks. |
| `crates/fnp-conformance/src/bin/run_security_gate.rs` | Security gate orchestrator | Security + runtime policy contract enforcement path. |
| `crates/fnp-conformance/src/bin/run_test_contract_gate.rs` | Logging contract gate | Structured logging schema and runtime suite coupling. |
| `crates/fnp-conformance/src/bin/run_workflow_scenario_gate.rs` | E2E replay gate | Scenario-level replay/logging checks. |
| `crates/fnp-conformance/src/bin/validate_phase2c_packet.rs` | Packet readiness gate | Packet artifact closure and schema validation. |

### DOC-PASS-01.4 Verification and logging implications by ownership boundary

| Ownership boundary | Unit/Property | Differential | E2E | Structured logging | Status |
|---|---|---|---|---|---|
| `fnp-dtype` | covered | partial (promotion-focused) | missing packet-local | partial | partial |
| `fnp-ndarray` | covered | partial | deferred packet-local | partial | partial |
| `fnp-ufunc` | covered | covered for scoped ops | deferred packet-local | covered | partial |
| `fnp-runtime` | covered | covered for policy-wire classes | covered via workflow suite | covered | covered for scoped surface |
| `fnp-iter` | covered | missing packet-local lane | missing packet-local | covered at schema level | partial |
| `fnp-random` | covered | missing packet-local lane | missing packet-local | covered at schema level | partial |
| `fnp-linalg` | covered | missing packet-local lane | missing packet-local | covered at schema level | partial |
| `fnp-io` | covered | missing packet-local lane | missing packet-local | covered at schema level | partial |

### DOC-PASS-01.5 Topology contradictions and closure backlog

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `TOPO-C001` | Packet-local crates (`fnp-iter`/`fnp-random`/`fnp-linalg`/`fnp-io`) now implement first-wave contract APIs, but conformance gate orchestration still prioritizes ufunc/runtime lanes. | high | `bd-23m.17`/`bd-23m.18`/`bd-23m.19`/`bd-23m.20` | Integrate packet-local differential + replay suites into operational gate flows. |
| `TOPO-C002` | Dependency direction is documented, but no automated check currently enforces forbidden edges. | medium | `bd-23m.23` | Add machine-enforced dependency-direction contract check and fail on violations. |
| `TOPO-C003` | Logging contracts exist for packet-local crates, but enforcement is mostly schema-level tests and not fully represented in gate outputs. | medium | packet G/I beads | Promote packet-local structured-log checks into gate assertions and readiness reports. |

## DOC-PASS-02 Symbol/API Census and Surface Classification

### DOC-PASS-02.1 Public symbol surface by crate

| Crate | Public symbol/API surface (Rust anchors) | Visibility class | Primary usage contexts |
|---|---|---|---|
| `fnp-dtype` | `DType::{name,item_size,parse}`, `promote`, `can_cast_lossless` (`crates/fnp-dtype/src/lib.rs`) | core semantic contract | Called by `fnp-ufunc` execution (`crates/fnp-ufunc/src/lib.rs`) and conformance promotion suite (`run_dtype_promotion_suite` in `crates/fnp-conformance/src/lib.rs`). |
| `fnp-ndarray` | `MemoryOrder`, `ShapeError`, `can_broadcast`, `broadcast_shape`, `broadcast_shapes`, `element_count`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout::{contiguous,nbytes}` (`crates/fnp-ndarray/src/lib.rs`) | core semantic contract | Used by ufunc execution shape checks and by conformance shape/stride suite (`run_shape_stride_suite`). |
| `fnp-ufunc` | `BinaryOp`, `UFuncArray::{new,scalar,shape,values,dtype,elementwise_binary,reduce_sum}`, `UFuncError` (`crates/fnp-ufunc/src/lib.rs`) | execution contract | Called from differential/metamorphic/adversarial harness via `execute_input_case` (`crates/fnp-conformance/src/ufunc_differential.rs`). |
| `fnp-runtime` | `RuntimeMode`, `CompatibilityClass`, `DecisionAction`, `DecisionLossModel`, `DecisionAuditContext`, `DecisionEvent`, `OverrideAuditEvent`, `EvidenceLedger`, `decide_*`, `evaluate_policy_override`, `posterior_incompatibility`, `expected_loss_for_action` (`crates/fnp-runtime/src/lib.rs`) | policy/audit contract | Used by runtime policy suites and workflow scenario execution in `fnp-conformance`. |
| `fnp-conformance` | `HarnessConfig`, `HarnessReport`, `SuiteReport`, `run_*` suites, plus module exports (`benchmark`, `contract_schema`, `raptorq_artifacts`, `security_contracts`, `test_contracts`, `ufunc_differential`, `workflow_scenarios`) (`crates/fnp-conformance/src/lib.rs`) | verification/tooling contract | Entry point for all gate binaries and packet-readiness tooling. |
| `fnp-iter` | `TransferClass`, `OverlapAction`, `TransferSelectorInput`, `FlatIterIndex`, `TransferError`, transfer/overlap/flatiter validators, `TransferLogRecord` (`crates/fnp-iter/src/lib.rs`) | packet-local first-wave contract | Transfer semantics foundation for packet-003/006 domains; currently unit/property-backed only. |
| `fnp-random` | `DeterministicRng`, `RandomError`, `RandomRuntimeMode`, `RandomLogRecord`, stream/state/bounded APIs (`crates/fnp-random/src/lib.rs`) | packet-local first-wave contract | Deterministic RNG boundary for packet-007; currently unit/property-backed only. |
| `fnp-linalg` | `LinAlgError`, `QrMode`, shape/solver/spectral/lstsq/policy validators, `LinAlgLogRecord` (`crates/fnp-linalg/src/lib.rs`) | packet-local first-wave contract | Linalg boundary for packet-008; currently unit/property-backed only. |
| `fnp-io` | `IOError`, `IOSupportedDType`, `MemmapMode`, `LoadDispatch`, NPY/NPZ validators, `IOLogRecord` (`crates/fnp-io/src/lib.rs`) | packet-local first-wave contract | IO boundary for packet-009; currently unit/property-backed only. |

### DOC-PASS-02.2 Operator entry points and call graph anchors

| Operator entry | Calls into | Main outputs/artifacts |
|---|---|---|
| `capture_numpy_oracle` (`crates/fnp-conformance/src/bin/capture_numpy_oracle.rs`) | `ufunc_differential::capture_numpy_oracle` | `crates/fnp-conformance/fixtures/oracle_outputs/ufunc_oracle_output.json` |
| `run_ufunc_differential` (`crates/fnp-conformance/src/bin/run_ufunc_differential.rs`) | `compare_against_oracle`, `write_differential_report` | `crates/fnp-conformance/fixtures/oracle_outputs/ufunc_differential_report.json` |
| `run_security_gate` (`crates/fnp-conformance/src/bin/run_security_gate.rs`) | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, `security_contracts::run_security_contract_suite` | `artifacts/logs/runtime_policy_e2e_*.jsonl` + gate summary JSON |
| `run_test_contract_gate` (`crates/fnp-conformance/src/bin/run_test_contract_gate.rs`) | `test_contracts::run_test_contract_suite` + runtime suites + runtime-log validation | `artifacts/logs/test_contract_e2e_*.jsonl` + gate summary JSON |
| `run_workflow_scenario_gate` (`crates/fnp-conformance/src/bin/run_workflow_scenario_gate.rs`) | `workflow_scenarios::run_user_workflow_scenario_suite` | `artifacts/logs/workflow_scenario_e2e_*.jsonl` |
| `generate_benchmark_baseline` (`crates/fnp-conformance/src/bin/generate_benchmark_baseline.rs`) | `benchmark::generate_benchmark_baseline` | `artifacts/baselines/ufunc_benchmark_baseline.json` |
| `generate_raptorq_sidecars` (`crates/fnp-conformance/src/bin/generate_raptorq_sidecars.rs`) | `raptorq_artifacts::generate_bundle_sidecar_and_reports` | `artifacts/raptorq/*.sidecar.json`, `*.scrub_report.json`, `*.decode_proof.json` |
| `validate_phase2c_packet` (`crates/fnp-conformance/src/bin/validate_phase2c_packet.rs`) | `contract_schema::validate_phase2c_packet`, `write_packet_readiness_report` | `artifacts/phase2c/<packet>/packet_readiness_report.json` |

### DOC-PASS-02.3 Stability and user-visibility tiers

| Symbol tier | Included surfaces | Stability assessment | User visibility |
|---|---|---|---|
| Tier A: compatibility-kernel contracts | `fnp-dtype`, `fnp-ndarray`, `fnp-ufunc`, `fnp-runtime` public APIs | medium: implemented and tested for first-wave scope, still parity-incomplete vs full NumPy | direct to internal consumers and conformance harnesses; not yet published as stable external API |
| Tier B: verification/tooling contracts | `fnp-conformance` suite/module public functions and binaries | medium: operationally stable for current packet workflow | primarily developer/CI/operator-facing |
| Tier C: reserved ownership placeholders | `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io` `add()` stubs | low: placeholders only; incompatible with parity claims | should be treated as non-contractual internals until packet implementation lands |

### DOC-PASS-02.4 Verification and logging implications by symbol family

| Symbol family | Unit/Property | Differential | E2E | Structured logging | Status |
|---|---|---|---|---|---|
| `fnp-dtype` promotion/cast APIs | covered (crate tests + promotion fixtures) | partial (promotion only; cast-matrix diff missing) | missing | missing dedicated cast reason-code taxonomy | partial |
| `fnp-ndarray` shape/stride APIs | covered (crate tests + shape_stride fixtures) | partial via ufunc differential shape validation | deferred | partial via runtime/workflow logs | partial |
| `fnp-ufunc` execution APIs | covered (crate tests + metamorphic/adversarial fixtures) | covered for scoped ops (`add/sub/mul/div/sum`) | deferred packet scenarios | covered at scenario/log-entry level | partial |
| `fnp-runtime` policy APIs | covered (crate tests + policy suites) | policy-wire adversarial coverage (non-numeric) | covered via workflow scenario suite | covered (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`) | covered for current scope |
| `fnp-conformance` contract/sidecar APIs | covered via module tests and gate binaries | N/A | covered through gate binaries | covered through gate log validation | covered/ongoing |
| Stub crate symbols (`add`) | trivial | none | none | none | missing (intentional parity debt) |

### DOC-PASS-02.5 Packet ownership and closure gates for unresolved domains (`DOC-C003`)

| Unresolved behavior family | Current placeholder crate API | Packet owner bead | Required closure gates |
|---|---|---|---|
| NDIter traversal/index semantics | transfer/index contract APIs in `fnp-iter` | `bd-23m.17` (`FNP-P2C-006`) | Expand into packet F/G/H/I lanes and wire into conformance gates. |
| RNG deterministic streams/state schema | deterministic stream/state APIs in `fnp-random` | `bd-23m.18` (`FNP-P2C-007`) | Add oracle differential + replay lanes and promote deterministic-seed witness artifacts into packet gates. |
| Linalg adapter semantics | shape/solver/policy contract APIs in `fnp-linalg` | `bd-23m.19` (`FNP-P2C-008`) | Add linalg differential + replay + optimization evidence lanes. |
| NPY/NPZ parser/writer hardening | parser/writer boundary validators in `fnp-io` | `bd-23m.20` (`FNP-P2C-009`) | Add IO differential + replay lanes and gate-enforced durability/logging links. |

## DOC-PASS-03 Data Model, State, and Invariant Mapping

### DOC-PASS-03.1 Canonical data models (state-carrying structs/enums)

| Model | Rust anchor | State carried | Mutability boundary |
|---|---|---|---|
| DType taxonomy | `fnp_dtype::DType` (`crates/fnp-dtype/src/lib.rs`) | discrete scalar domain (`Bool`, `I32`, `I64`, `F32`, `F64`) | immutable value enum; transitions only via pure functions (`parse`, `promote`, cast checks). |
| Shape/stride layout model | `fnp_ndarray::NdLayout` + `MemoryOrder` + `ShapeError` (`crates/fnp-ndarray/src/lib.rs`) | `shape`, `strides`, `item_size`, legality error class | constructor enforces legality (`NdLayout::contiguous`); consumers can read fields but validity hinges on checked constructors/functions. |
| Ufunc execution payload | `fnp_ufunc::UFuncArray` + `BinaryOp` + `UFuncError` (`crates/fnp-ufunc/src/lib.rs`) | logical shape, value buffer, dtype identity | creation gated by `UFuncArray::new`; operations produce new arrays (non-in-place state transitions). |
| Runtime policy event model | `RuntimeMode`, `CompatibilityClass`, `DecisionAction`, `DecisionAuditContext`, `DecisionEvent`, `OverrideAuditEvent`, `EvidenceLedger` (`crates/fnp-runtime/src/lib.rs`) | decision inputs, posterior/expected-loss outputs, audit metadata, append-only event sequence | state mutation isolated to `EvidenceLedger::record` and audit-context normalization. |
| Conformance fixture/report models | `HarnessConfig`, `SuiteReport`, `UFunc*Case`/`UFuncDifferentialReport` (`crates/fnp-conformance/src/lib.rs`, `ufunc_differential.rs`) | fixture inputs, pass/fail counters, mismatch diagnostics | suites are functional over fixture files; report structs are produced as outputs. |
| Artifact durability/contract models | `PacketReadinessReport`, `RaptorQSidecar`, `ScrubReport`, `DecodeProofArtifact` (`contract_schema.rs`, `raptorq_artifacts.rs`) | packet completeness state, symbol/hash metadata, scrub/decode status | generated artifacts are immutable outputs of validator/generator pipelines. |

### DOC-PASS-03.2 State transitions and lifecycle edges

| Transition | Source state | Operation edge | Target state | Fail-closed/error path |
|---|---|---|---|---|
| reshape inference | `(new_shape spec, old element count)` | `fix_unknown_dimension` | resolved `Vec<usize>` | `MultipleUnknownDimensions`, `InvalidDimension`, `IncompatibleElementCount`, `Overflow` |
| broadcast legality | `(lhs shape, rhs shape)` | `broadcast_shape` / `broadcast_shapes` | merged output shape | `ShapeError::IncompatibleBroadcast` |
| contiguous layout derivation | `(shape, item_size, order)` | `contiguous_strides` / `NdLayout::contiguous` | legal stride vector/layout | `InvalidItemSize`, `Overflow` |
| ufunc binary execution | `(lhs UFuncArray, rhs UFuncArray, BinaryOp)` | `elementwise_binary` | new `UFuncArray` output | shape error or input-length rejection via `UFuncError` |
| ufunc reduction | `(UFuncArray, axis, keepdims)` | `reduce_sum` | reduced-shape `UFuncArray` | axis bounds and shape-derived failures (`AxisOutOfBounds`, shape errors) |
| runtime policy decision | `(mode,class,risk,threshold)` | `decide_compatibility*` | `DecisionAction` | unknown wire mode/class => `FailClosed` |
| runtime audit append | `(ledger, decision context)` | `decide_and_record_with_context` | appended `DecisionEvent` | context normalization to non-empty defaults for required audit fields |
| workflow scenario replay | `(scenario corpus + fixtures + mode)` | `run_user_workflow_scenario_suite` | per-scenario pass/fail report + JSONL entries | missing fixtures/scripts/required fields produce deterministic failure records |
| packet readiness validation | `(packet dir artifacts)` | `validate_phase2c_packet` | `PacketReadinessReport` | missing files/fields parse errors force non-ready status |

### DOC-PASS-03.3 Mutability boundaries and write surfaces

| Boundary | Writable state | Protection mechanism |
|---|---|---|
| Numeric core objects (`DType`, `UFuncArray` outputs) | only through constructors/ops that enforce invariants | pure/persistent-style API: operations return new values instead of mutating shared global state |
| Layout legality (`fnp-ndarray`) | derived vectors (`shape`, `strides`) | overflow and legality checks before any layout object is returned |
| Runtime policy ledger | append-only event vector (`EvidenceLedger`) | controlled via `record`; audit context normalized to required default tokens |
| Gate/workflow logs | filesystem JSONL append (`runtime_policy` and `workflow_scenarios`) | explicit schema field checks in contract suites and gate binaries |
| Artifact contracts | packet readiness and sidecar outputs | schema/token/path validation and hash-based scrub/decode proof generation |

### DOC-PASS-03.4 Invariant obligations and verification/logging implications

| Invariant family | Contract expression | Evidence (covered/missing/deferred) | Logging implication |
|---|---|---|---|
| Shape arithmetic safety | `element_count` and stride derivation must not overflow | covered (unit + shape fixtures) | failures should carry stable reason taxonomy in packet logs (partial) |
| Broadcast determinism | broadcast merge independent of operand order alignment rules | covered (unit + shape fixtures), broader high-arity differentials deferred | scenario logs carry fixture IDs but no dedicated broadcast reason-code matrix yet |
| Reshape `-1` legality | at most one unknown dim and exact element-count preservation | covered (unit tests) | error family present; packet-level reason-code mapping still partial |
| Ufunc input/output consistency | input length == element count, axis bounds legal, dtype promotion deterministic | covered for scoped ops (unit + metamorphic/adversarial + differential) | workflow logs include required fields and per-step pass/fail detail |
| Runtime fail-closed doctrine | unknown/incompatible wire semantics must map to `fail_closed` | covered (policy/adversarial suites + workflow scenarios) | explicitly logged via required fields (`fixture_id`,`seed`,`mode`,`env_fingerprint`,`artifact_refs`,`reason_code`) |
| Packet artifact completeness | required files/tokens/JSON/YAML paths must be present | covered by contract schema validator | readiness report captures missing artifacts/fields/parse errors |
| Durability integrity | sidecar symbols/hash scrub/decode proof must be coherent | covered for generated bundles | scrub/decode artifacts become mandatory audit trail pointers |
| Unresolved domain executability | NDIter/RNG/linalg/IO crates must expose real domain APIs, not stubs | missing (currently placeholders) | requires packet-local reason-code vocab once implemented |

### DOC-PASS-03.5 Data/invariant contradictions and unknowns

| ID | Contradiction or unknown | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `STATE-C001` | `NdLayout` fields are public, so post-construction mutation can bypass constructor-time legality assumptions. | medium | `bd-23m.12` | Decide/encode whether layout fields remain public by design or become encapsulated with invariant-preserving mutators only. |
| `STATE-C002` | Runtime/workflow logging can no-op when no path is configured, weakening forensic guarantees outside gate runs. | medium | `bd-23m.6` | Require explicit log paths in orchestrator contexts and fail on missing required logs. |
| `STATE-C003` | Packet-local data models now exist for NDIter/RNG/linalg/IO, but their packet-local differential/e2e evidence lanes are not fully integrated into gate workflows. | high | `bd-23m.17`/`bd-23m.18`/`bd-23m.19`/`bd-23m.20` | Integrate packet-local differential/replay suites into gate outputs and packet readiness closure artifacts. |

## DOC-PASS-04 Execution-Path Tracing and Control-Flow Narratives

### DOC-PASS-04.1 Major workflow narratives (entrypoint -> branch -> terminal)

| Workflow ID | Entry anchor | Ordered branch decisions | Terminal outcomes | Evidence anchors |
|---|---|---|---|---|
| `EP-001` Ufunc differential replay | `crates/fnp-conformance/src/bin/run_ufunc_differential.rs` | fixture/oracle load -> case parse -> `UFuncArray` shape/value admission -> op dispatch -> compare | pass report / mismatch report / parse or admission failure | `crates/fnp-conformance/src/ufunc_differential.rs`, `fixtures/ufunc_*` |
| `EP-002` Security gate | `crates/fnp-conformance/src/bin/run_security_gate.rs` | runtime policy suite -> adversarial policy suite -> security-contract suite -> aggregate gate result | gate pass / gate fail with reason taxonomy | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, `security_contracts` |
| `EP-003` Workflow scenario gate | `crates/fnp-conformance/src/bin/run_workflow_scenario_gate.rs` | fixture parse -> strict/hardened execution -> expectation compare -> log append | scenario pass / deterministic failure record | `workflow_scenarios::run_user_workflow_scenario_suite`, `artifacts/logs/workflow_scenario_e2e_*.jsonl` |
| `EP-004` Packet readiness validator | `crates/fnp-conformance/src/bin/validate_phase2c_packet.rs` | required file check -> schema/token check -> parse check -> readiness summarization | ready / non-ready diagnostic report | `contract_schema::validate_phase2c_packet`, `packet_readiness_report.json` |
| `EP-005` IO dispatch and validation | `crates/fnp-io/src/lib.rs` (`classify_load_dispatch`, header/payload validators) | NPZ magic -> NPY magic -> pickle allowed? -> fail-closed invalid prefix | dispatch branch / stable IO error reason code | `IO_PACKET_REASON_CODES`, `artifacts/phase2c/FNP-P2C-009/*` |
| `EP-006` RNG deterministic generation | `crates/fnp-random/src/lib.rs` (`DeterministicRng`, `bounded_u64`) | upper-bound gate -> rejection sampling threshold loop -> modulo projection | bounded value / stable RNG error reason code | `RANDOM_PACKET_REASON_CODES`, `artifacts/phase2c/FNP-P2C-007/*` |
| `EP-007` Linalg branch family | `crates/fnp-linalg/src/lib.rs` (shape preflight, QR/SVD/LSTSQ/policy validators) | rank/shape preflight -> mode token parse -> convergence gate -> policy metadata gate | output-shape contracts / stable linalg reason codes | `LINALG_PACKET_REASON_CODES`, `artifacts/phase2c/FNP-P2C-008/*` |
| `EP-008` Transfer/index legality | `crates/fnp-iter/src/lib.rs` (transfer selector, overlap policy, flatiter validators) | context checks -> stride multiple checks -> overlap direction -> index/value arity checks | transfer class/copy action / stable transfer reason codes | `TRANSFER_PACKET_REASON_CODES`, `artifacts/phase2c/FNP-P2C-003/*` |

### DOC-PASS-04.2 Control-flow ordering and fallback law

| Branch family | Ordering rule | Fallback semantics |
|---|---|---|
| Dispatch classification (`EP-005`) | NPZ prefix first, then NPY prefix, then pickle only with explicit policy allow | Unknown/forbidden prefixes fail closed (`io_load_dispatch_invalid`) |
| Runtime mode/class decode (`EP-002`/`EP-003`) | normalize + classify known tokens before action selection | unknown metadata resolves to fail-closed and logged reason code |
| Ufunc admission (`EP-001`) | validate fixtures and shape/value/dtype preconditions before numeric dispatch | precondition failures short-circuit with deterministic diagnostics |
| RNG bounded sampling (`EP-006`) | reject `upper_bound == 0` before sampling loop | invalid bound hard-fails with stable reason code |
| Linalg mode/convergence (`EP-007`) | shape/rank gate precedes mode/convergence checks | invalid mode/non-convergence/unknown metadata fail closed |
| Transfer overlap/indexing (`EP-008`) | context sanity then overlap/index checks before allowing lane operations | contract violations emit stable transfer reason codes and reject operation |

### DOC-PASS-04.3 Verification/logging implications by execution path

| Workflow ID | Unit/Property | Differential | E2E | Structured logging | Status |
|---|---|---|---|---|---|
| `EP-001` | covered | covered for scoped ufunc corpus | partial | covered | partial parity debt outside scoped op set |
| `EP-002` | covered | covered for policy-wire classes | covered | covered | strong for current policy scope |
| `EP-003` | covered | N/A | covered | covered when path configured | medium operational risk |
| `EP-004` | covered | N/A | covered | structured readiness reports | strong artifact-topology coverage |
| `EP-005` | covered | missing packet-local differential lane | missing packet-local replay lane | schema-level coverage exists | partial |
| `EP-006` | covered | missing packet-local differential lane | missing packet-local replay lane | schema-level coverage exists | partial |
| `EP-007` | covered | missing packet-local differential lane | missing packet-local replay lane | schema-level coverage exists | partial |
| `EP-008` | covered | missing packet-local differential lane | missing packet-local replay lane | schema-level coverage exists | partial |

### DOC-PASS-04.4 Control-flow contradictions and closure backlog

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `FLOW-C001` | Packet-local paths (`EP-005`..`EP-008`) are implemented and tested but not yet promoted to first-class suites in core gate orchestration. | high | `bd-23m.14`/`17`/`18`/`19`/`20` | Integrate packet-local differential/replay suites into gate binaries and `run_all_core_suites`. |
| `FLOW-C002` | Runtime/workflow log append behavior is configuration-dependent, creating potential forensic blind spots. | medium | `bd-23m.6` | Require explicit log path configuration and fail when expected logs are absent. |
| `FLOW-C003` | Packet readiness reports validate artifact topology but do not yet encode branch-reachability proof links for packet-local paths. | medium | packet I beads + DOC pass-10 | Add branch-level replay evidence references into readiness report conventions. |

## DOC-PASS-05 Complexity, Performance, and Memory Characterization

### DOC-PASS-05.1 Complexity map by workflow

| Workflow ID | Dominant operation class | Time complexity (current wave) | Space complexity (current wave) | Anchors |
|---|---|---|---|---|
| `EP-001` | fixture replay + ufunc execution + oracle compare | `O(C * N)` (cases × per-case element count) | `O(N)` per-case buffers + report growth | `ufunc_differential.rs`, `fnp-ufunc/src/lib.rs` |
| `EP-002` | runtime/security suite execution | `O(C)` over policy/security fixtures | `O(C)` reports/log metadata | `run_security_gate.rs`, runtime suites |
| `EP-003` | scenario replay engine | `O(S * K)` (scenarios × steps) | `O(S + log_entries)` | `workflow_scenarios.rs`, workflow gate |
| `EP-004` | packet readiness contract scan | `O(F)` required artifact checks | `O(F)` diagnostics | `contract_schema.rs`, readiness validator |
| `EP-005` | IO dispatch + header/shape budget checks | dispatch `O(1)`; schema checks `O(rank)` | `O(rank)` metadata structures | `fnp-io/src/lib.rs` |
| `EP-006` | deterministic RNG + bounded sampling | `O(1)` per draw expected; `fill_u64(len)` is `O(len)` | `O(1)` per draw; `O(len)` for fills | `fnp-random/src/lib.rs` |
| `EP-007` | linalg contract validators + small helper kernels | mostly `O(rank)`/`O(1)` (2x2 solve is constant) | mostly `O(1)` plus output-shape vectors | `fnp-linalg/src/lib.rs` |
| `EP-008` | transfer selector + overlap/index checks | selector/overlap `O(1)`; index validation `O(len)` | `O(1)` plus caller index storage | `fnp-iter/src/lib.rs` |

### DOC-PASS-05.2 Memory-growth characterization and boundedness controls

| Domain | Growth driver | Current boundedness controls | Residual risk |
|---|---|---|---|
| Ufunc execution | broadcasted output element count and dtype width | shape legality + element-count admission | large broadcast expansions can still pressure memory |
| Conformance/gate reports | fixture corpus size and run count | harness config and packet contract checks | longitudinal report/log accumulation can grow quickly |
| RNG fill workflows | requested output length | caller-controlled `len` and packet fixture contracts | large fill requests can dominate memory |
| Linalg contract workflows | batch-shape metadata and output tuples | `MAX_BATCH_SHAPE_CHECKS`, policy bounds | future full kernels will change memory profile materially |
| IO parser/workflow | shape rank, archive member count, payload footprint | `MAX_HEADER_BYTES`, `MAX_ARCHIVE_MEMBERS`, byte budget constants | decoded payload memory envelopes need cross-gate enforcement |
| Transfer/index checks | mask/fancy index lengths | fail-fast validation paths | large caller-provided index vectors still expensive |

### DOC-PASS-05.3 Hotspot family inventory and optimization-governance readiness

| Hotspot family | Current evidence | Governance gap | Owner |
|---|---|---|---|
| Ufunc loops | baseline artifact exists (`artifacts/baselines/ufunc_benchmark_baseline.json`) | continue per-lever profile/isomorphism deltas | `bd-23m.16` |
| Conformance orchestration | differential/gate reports exist | standardized profiler artifact convention missing | packet H/I + foundation |
| Runtime/workflow gates | gate summaries + logs exist | longitudinal cost tracking not yet formalized | `bd-23m.8` |
| Packet-local iter/random/linalg/io paths | unit/property evidence exists | dedicated perf baselines/profiles mostly missing | `bd-23m.14`/`18`/`19`/`20` |
| Durability sidecar pipeline | sidecar/scrub/decode artifacts generated | throughput/storage trend baselines missing | packet I + durability owners |

### DOC-PASS-05.4 Verification/logging implications for optimization work

| Domain | Unit/Property | Differential | E2E | Logging | Status |
|---|---|---|---|---|---|
| Ufunc optimization | covered | covered (scoped corpus) | partial | covered | partial outside scoped op set |
| Runtime/security optimization | covered | covered (policy-wire) | covered | covered | strong |
| Iter/random/linalg/io optimization | covered | missing packet-local lanes | missing packet-local lanes | schema-level only | missing/partial |
| Durability pipeline optimization | covered by contract suites | N/A | gate path exists | artifact metadata logged | partial trend visibility |

### DOC-PASS-05.5 Performance/memory contradiction register

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `PERF-C001` | EV-gated optimization policy is explicit, but non-ufunc packet paths lack standardized baseline/profile artifacts. | high | `bd-23m.8`, packet H beads | Require path-specific baseline/profile attachments in packet optimization reports. |
| `PERF-C002` | RNG rejection sampling (`bounded_u64`) has expected constant behavior but no explicit retry cap in adversarial worst cases. | medium | `bd-23m.18` | Add bounded retry policy or explicit guardrail rationale + adversarial witness tests. |
| `PERF-C003` | Gate-cost regressions are not tracked longitudinally as first-class artifacts. | medium | `bd-23m.8` | Add periodic gate-cost trend artifacts with alert thresholds. |
| `PERF-C004` | Crate-local memory bounds exist, but a cross-packet aggregate memory envelope is undocumented. | medium | `bd-23m.24.6` follow-on + packet owners | Publish aggregate memory budget matrix with enforcement hooks in gate scripts. |

## 1. Legacy Oracle

- Root: /dp/franken_numpy/legacy_numpy_code/numpy
- Upstream: numpy/numpy

## 2. Subsystem Map

- numpy/_core/src/multiarray: ndarray construction, shape/stride logic, assignment, nditer, text parsing.
- numpy/_core/src/umath: ufunc machinery and reduction kernels.
- numpy/_core/include/numpy: public ndarray/dtype APIs and ABI contracts.
- numpy/_core/src/_simd: CPU feature detection and SIMD dispatch checks.
- numpy/random and random/src: BitGenerator implementations and distribution paths.
- numpy/lib, numpy/linalg, numpy/fft, numpy/matrixlib, numpy/ma: higher-level Python semantics.
- numpy/tests and package-specific test folders: regression and parity baseline.

## 3. Semantic Hotspots (Must Preserve)

1. shape.c/shape.h dimensional and broadcast legality.
2. descriptor/dtypemeta and casting tables.
3. lowlevel_strided_loops and dtype_transfer assignment semantics.
4. nditer behavior across memory order/stride/layout combinations.
5. stride_tricks behavior for views and broadcasted representations.
6. ufunc override and dispatch selection semantics.
7. dtype promotion matrix determinism.

## 4. Compatibility-Critical Behaviors

- Array API and array-function overrides.
- Public PyArrayObject layout expectations for downstream interop.
- Runtime SIMD capability detection affecting chosen kernels.
- ufunc reduction vs elementwise override precedence.

## 5. Security and Stability Risk Areas

- textreading parser paths and malformed input handling.
- stringdtype conversion routines and buffer bounds.
- stride arithmetic UB risk in low-level loops.
- random generator state determinism and serialization paths.
- external data and dlpack interoperability boundaries.

## 6. Parity-Debt Sequencing Ledger (Replaces Historical V1 Scope-Cut Framing)

This program does not accept reduced-scope V1 completion. Remaining gaps are tracked as explicit parity debt with owners and closure gates.

| Domain | Current implementation state | Remaining parity debt | Owner packet |
|---|---|---|---|
| Shape/stride + ufunc + runtime policy | first-wave core implemented and gate-backed | broaden fixture/oracle coverage and close packet-local F/G/H/I lanes | `bd-23m.12`, `bd-23m.16`, `bd-23m.17` |
| Transfer / NDIter semantics | packet-local transfer/index contracts implemented | integrate full NDIter traversal behavior and gate-backed differential/replay evidence | `bd-23m.14`, `bd-23m.17` |
| RNG | deterministic stream/state contracts implemented | oracle differential + workflow replay + final evidence pack | `bd-23m.18` |
| Linalg | shape/solver policy contracts implemented | differential/replay/optimization proof chain to packet closure | `bd-23m.19` |
| IO | NPY/NPZ boundary validation contracts implemented | round-trip/adversarial differential and full gate-integrated evidence chain | `bd-23m.20` |
| Non-core ecosystem breadth | not release-cut criteria | remains explicit parity debt backlog until full drop-in behavior matrix reaches zero | master program `bd-23m` |

## 7. High-Value Conformance Fixture Families

- _core/tests for dtype, array interface, overlap, simd, string dtype.
- random/tests for deterministic generator streams.
- lib/linalg/fft/polynomial/matrixlib/ma tests for scoped behavioral parity.
- tests/test_public_api and testing/tests for API surface stability.

## 8. Extraction Notes for Rust Spec

- Start with shape/stride/dtype model before any heavy optimization.
- Keep promotion matrix and casting behavior explicit and versioned.
- Use parity-by-op-family reports as release gates.

## 9. Packet `FNP-P2C-006` Legacy Anchor + Behavior Ledger (A-stage)

Packet focus: stride-tricks and broadcasting API.

Packet-B output status: `artifacts/phase2c/FNP-P2C-006/contract_table.md` now defines strict/hardened invariant rows and failure reason-code vocabulary for this subsystem.

### 9.1 Legacy anchor -> Rust boundary map

| Legacy anchors | Observable behavior family | Planned Rust boundary |
|---|---|---|
| `numpy/lib/_stride_tricks_impl.py` (`as_strided`, `_broadcast_to`, `broadcast_to`) | stride-view construction, read-only/writeable semantics, shape validation | `crates/fnp-ndarray` public API + layout core (`broadcast_shape`, `broadcast_shapes`, `NdLayout`) |
| `numpy/lib/_stride_tricks_impl.py` (`_broadcast_shape`, `broadcast_arrays`) | N-ary broadcast merge, high-arity behavior, output view semantics | `crates/fnp-ndarray` + packet-specific conformance fixtures (`bd-23m.17.5`, `bd-23m.17.6`) |
| `numpy/_core/src/multiarray/nditer_constr.c` (`npyiter_fill_axisdata`, `broadcast_error`, `operand_different_than_broadcast`) | zero-stride propagation, no-broadcast rejection, mismatch diagnostics | `crates/fnp-iter` iterator semantics layer (planned under `bd-23m.17.4`) |
| `numpy/_core/src/multiarray/nditer_api.c` (`NpyIter_GetShape`, `NpyIter_CreateCompatibleStrides`) | iterator-shape exposure and compatible-stride derivation | `crates/fnp-iter` traversal/introspection contracts + `fnp-ndarray` layout integration |

### 9.2 Verification hooks recorded by packet-A ledger

| Verification lane | Current status | Next owner bead |
|---|---|---|
| Unit/property | Anchor ledger complete, executable packet-specific tests not yet implemented | `bd-23m.17.5` |
| Differential/metamorphic/adversarial | Anchor ledger complete, packet-specific differential corpus not yet implemented | `bd-23m.17.6` |
| E2E replay/forensics | Anchor ledger complete, packet-specific workflow scenario not yet implemented | `bd-23m.17.7` |
| Structured logging | Contract fields known; packet-local enforcement hooks still pending implementation | `bd-23m.17.5`, `bd-23m.17.7` |
