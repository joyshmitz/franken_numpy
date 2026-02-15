# FNP-P2C-002 Rust Implementation Plan

packet_id: `FNP-P2C-002`  
subsystem: `dtype descriptors and promotion`

## 1. Crate and Module Boundary Skeleton

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-dtype` | `dtype_catalog` (existing in `src/lib.rs`) | canonical scoped dtype taxonomy and parsing | `DType`, `DType::parse`, `DType::name`, `DType::item_size` |
| `crates/fnp-dtype` | `promotion_table` (existing core + packet-D expansion) | deterministic promotion decisions and policy invariants | `promote(lhs, rhs) -> DType` |
| `crates/fnp-dtype` | `cast_policy` (existing core + packet-D expansion) | lossless cast admissibility and rejection taxonomy | `can_cast_lossless(src, dst) -> bool` |
| `crates/fnp-dtype` | `descriptor_norm` (packet-D planned boundary) | descriptor/type-object normalization and alias canonicalization | `normalize_descriptor(...) -> Result<DType, DTypeError>` (planned) |
| `crates/fnp-conformance` | `dtype_promotion_suite` (existing) | fixture-driven dtype promotion validation | `run_dtype_promotion_suite` |
| `crates/fnp-conformance` | `dtype_packet_differential` (packet-F planned boundary) | differential/metamorphic/adversarial dtype and cast verification | packet-F runner entrypoint (planned) |
| `crates/fnp-runtime` | policy audit integration (existing) | strict/hardened decision logging and fail-closed boundaries | runtime policy decision/audit interfaces |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Stabilize packet-D module boundary plan around current `fnp-dtype` surface without introducing unsafe paths.
2. Add descriptor normalization scaffolding (alias/canonicalization) in `fnp-dtype` with explicit fail-closed error taxonomy.
3. Expand cast-policy matrix representation incrementally with deterministic tests and invariant checks.
4. Wire packet-F conformance runner to oracle-driven dtype/cast scenarios and adversarial edges.
5. Add packet-G e2e mixed-dtype replay path with structured logs and artifact linking.
6. Promote packet-I parity/risk/durability artifacts once E/F/G/H evidence is complete.

## 3. Public Surface Contract Notes

- Current stable boundary:
  - `DType`
  - `promote`
  - `can_cast_lossless`
- Packet-D planned additions are constrained to deterministic descriptor normalization and explicit error-class handling.
- No compatibility shims; direct contract-aligned evolution only.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | `crates/fnp-dtype/src/lib.rs` tests + packet-E expansion | packet-E structured logs + property artifacts |
| Differential/metamorphic/adversarial | packet-F conformance hooks + dtype fixtures | packet-F differential/parity reports |
| E2E/replay | packet-G mixed-dtype workflow scripts | replay logs under `artifacts/logs/` |
| Runtime policy audit | `fnp-runtime` decision and override logs | security gate + policy evidence ledger |

## 5. Structured Logging Emission Points

- Fixture execution for dtype promotion/cast decisions (`fixture_id`, `seed`, `mode`).
- Descriptor normalization/cast rejection paths (`reason_code`, `artifact_refs`).
- E2E pipeline checkpoints (`env_fingerprint`, artifact linkage).

## 6. Artifact Boundary Plan

- Packet artifacts under `artifacts/phase2c/FNP-P2C-002/`.
- Program controls under `artifacts/contracts/`.
- Replay/security logs under `artifacts/logs/`.
- Durability artifacts under `artifacts/raptorq/` and packet-I outputs.

## 7. Compile-Safe Skeleton Validation

- Planning-stage validation rules:
  - No API-breaking code changes in this bead.
  - Existing dtype/promotion/cast boundary remains compilable.
  - Packet validator keeps `missing_fields=0` for packet artifacts as they are added.
- Validation command (offloaded via `rch`):  
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-002`

## 8. Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## 9. Rollback Handle

If implementation-plan boundaries prove invalid, revert `artifacts/phase2c/FNP-P2C-002/implementation_plan.md` and restore the previous packet-D planning baseline.
