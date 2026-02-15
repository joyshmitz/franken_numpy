# FNP-P2C-004 Contract Table

packet_id: `FNP-P2C-004`  
subsystem: `nditer traversal/index semantics`

## strict_mode_policy

Strict mode must preserve legacy-observable iterator construction, traversal, index/multi-index controls, and flat-iterator indexing/assignment outcomes for the scoped packet surface.

## hardened_mode_policy

Hardened mode must preserve the same public iterator/indexing contract while adding bounded validation and deterministic audit semantics; unknown or incompatible semantics remain fail-closed.

## excluded_scope

- Full parity for every advanced legacy buffering/reduction/writeback nuance is deferred to packet `D`/`E`/`F` closure.
- Exact warning/error text parity for all deprecated iterator edge paths is deferred; class/family parity remains the immediate contract.
- APIs outside packet `FNP-P2C-004` are excluded from this table.

## performance_sentinels

- iterator construction overhead under high-operand/high-rank `op_axes` configurations.
- traversal/index retrieval overhead for `multi_index`, `c_index`, and `f_index` modes.
- flatiter fancy-index/assignment throughput and bounds-check cost.

## Machine-Checkable Contract Rows

| contract_id | preconditions | shape_stride_contract | dtype_cast_contract | memory_alias_contract | strict_mode_policy | hardened_mode_policy | postconditions | error_contract | failure_reason_code | unit_property_ids | differential_ids | e2e_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P2C004-R01` | iterator constructor receives operands + optional flags/op_flags/op_axes/itershape | iterator ndim/shape/broadcast planning is validated before traversal | dtype/casting checks follow constructor casting policy | constructor does not silently relax alias constraints | same accept/reject classes for flag/op_axes combinations as legacy | same classes; malformed/unknown metadata fails closed with deterministic audit | initialized iterator state is deterministic for fixed constructor input | reject invalid flag/op_axes/itershape combinations | `nditer_constructor_invalid_configuration` | `UP-004-constructor-validation` | `DF-004-constructor-matrix` | `E2E-004-iterator-constructor-path` |
| `P2C004-R02` | iterator created with `multi_index` tracking | permutation-aware index recovery must match traversal order | dtype unaffected by index tracking | no alias mutation in index exposure path | deterministic `multi_index` stream parity across C/F/reversed layouts | same stream with fail-closed handling for invalid mode transitions | emitted multi-index tuple sequence is deterministic and bounds-valid | raise stable error if multi-index requested in invalid state | `nditer_multi_index_state_invalid` | `UP-004-multi-index-order` | `DF-004-multi-index-oracle` | `E2E-004-iterator-index-replay` |
| `P2C004-R03` | iterator created with `c_index` or `f_index` mode | index-to-iteration mapping follows selected order semantics | dtype unaffected | no alias mutation in index exposure path | deterministic index stream parity for fixed input/layout | same parity with stable bounds/errors | index sequence remains coherent with traversal | reject out-of-bounds index seek requests | `nditer_index_out_of_bounds` | `UP-004-cf-index-order` | `DF-004-index-seek-oracle` | `E2E-004-iterator-index-replay` |
| `P2C004-R04` | index seek request via `multi_index`/`index`/`iterindex` setters | seek updates iterator position according to iterator shape/order contract | dtype unchanged by seek | no alias-policy changes from seek operation | in-bounds seek succeeds; out-of-bounds/invalid seek fails with stable class | same behavior; deterministic reason-code logging in hardened mode | iterator resumes from requested position with coherent index state | reject seek when mode preconditions are not met | `nditer_seek_mode_mismatch` | `UP-004-seek-coherence` | `DF-004-index-seek-oracle` | `E2E-004-iterator-index-replay` |
| `P2C004-R05` | iterator supports range manipulation (`iterrange`) | restricted range must remain within iterator iteration domain | dtype unchanged | no alias-policy changes | range reset semantics match legacy (`istart < iend` active else finished) | same semantics with fail-closed handling for malformed range values | iterator start/finished state aligns with range contract | reject invalid range tuple/state | `nditer_iterrange_invalid` | `UP-004-iterrange-state` | `DF-004-index-seek-oracle` | `E2E-004-iterator-index-replay` |
| `P2C004-R06` | transition requests for `remove_axis`/`remove_multi_index` | axis removal requires valid tracked mode and non-buffered constraints | dtype unchanged | no silent alias relaxation during mode transition | enforce documented preconditions and reset behavior | same transition contract; invalid transitions fail closed | transition result is deterministic and iterator reset semantics are preserved | reject when preconditions (mode/buffer/index) are violated | `nditer_transition_precondition_failed` | `UP-004-mode-transitions` | `DF-004-transition-error-taxonomy` | `E2E-004-iterator-mode-shift` |
| `P2C004-R07` | request to enable `external_loop` mode | external-loop compatibility checks against index/multi-index/range rules | dtype unchanged | no alias-policy changes | same compatibility/rejection behavior as legacy | same behavior with deterministic diagnostics | external-loop iterator state is consistent with traversal contract | reject incompatible flag combinations | `nditer_external_loop_incompatible_flags` | `UP-004-external-loop-compat` | `DF-004-transition-error-taxonomy` | `E2E-004-iterator-mode-shift` |
| `P2C004-R08` | operand flagged `no_broadcast` in multi-operand iterator | broadcast planner must not expand protected operand axes | dtype unchanged | no alias-policy changes | no-broadcast violations raise stable failure class | same failure class and reason-code audit | successful plans preserve no-broadcast invariants | reject non-broadcastable operand plans | `nditer_no_broadcast_violation` | `UP-004-no-broadcast-law` | `DF-004-no-broadcast-adversarial` | `E2E-004-iterator-broadcast-path` |
| `P2C004-R09` | iterator uses `copy_if_overlap` and optional `overlap_assume_elementwise` | shape/traversal law unchanged while overlap policy is applied | dtype/casting semantics preserved under copy pathway | overlap copy must prevent unsafe read/write alias behavior unless explicit safe assumption flag is provided | overlap-sensitive operations follow legacy copy/no-copy outcomes | same outward outcomes with bounded hardened auditing for risk decisions | resulting operands honor overlap safety contract | reject/adjust unsafe overlap configurations per policy | `nditer_overlap_policy_triggered` | `UP-004-copy-if-overlap` | `DF-004-overlap-policy-oracle` | `E2E-004-overlap-safety-replay` |
| `P2C004-R10` | flatiter receives integer/slice/fancy/boolean indexing or assignment | 1-D flattened indexing semantics enforced with deterministic bounds handling | assignment casts follow array dtype transfer semantics | writes require underlying array writeability checks | valid index/assignment forms succeed with legacy-compatible outcomes | same surface behavior; malformed/unsupported indices fail closed | flatiter result/assignment effects are deterministic for fixed input | reject unsupported index forms and out-of-range accesses with stable classes | `flatiter_indexing_contract_violation` | `UP-004-flatiter-indexing` | `DF-004-flatiter-indexing-oracle` | `E2E-004-flatiter-journey` |
| `P2C004-R11` | `ndindex`/`ndenumerate` invoked for valid shapes | generated coordinate stream follows shape cartesian product law | dtype unaffected except value retrieval in `ndenumerate` | no alias policy changes | index/value stream parity with legacy and ndindex↔ndenumerate consistency | same stream/failure behavior with class stability | stream is deterministic; empty/zero-dim cases preserve documented behavior | reject negative/non-integer dimensions with stable failure classes | `ndindex_shape_validation_failed` | `UP-004-ndindex-ndenumerate` | `DF-004-index-stream-oracle` | `E2E-004-index-stream-replay` |

## Logging and Failure Semantics

All packet validations must emit structured fields:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Reason-code vocabulary for this packet:

- `nditer_constructor_invalid_configuration`
- `nditer_multi_index_state_invalid`
- `nditer_index_out_of_bounds`
- `nditer_seek_mode_mismatch`
- `nditer_iterrange_invalid`
- `nditer_transition_precondition_failed`
- `nditer_external_loop_incompatible_flags`
- `nditer_no_broadcast_violation`
- `nditer_overlap_policy_triggered`
- `flatiter_indexing_contract_violation`
- `ndindex_shape_validation_failed`

## Budgeted Mode and Expected-Loss Notes

- Budgeted mode: hardened execution enforces explicit limits on iterator construction complexity, seek/reset retries, and overlap-remediation paths, with deterministic fail-closed exhaustion behavior.
- Expected-loss model: iterator policy mediation (especially overlap and mode-transition controls) must log state/action/loss rationale.
- Calibration trigger: if strict/hardened drift for iterator failure classes exceeds packet thresholds, automatically fallback to conservative deterministic behavior and emit audit reason codes.

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Rollback Handle

If iterator contract drift is detected, revert `artifacts/phase2c/FNP-P2C-004/contract_table.md` and restore the last green packet boundary contract baseline.
