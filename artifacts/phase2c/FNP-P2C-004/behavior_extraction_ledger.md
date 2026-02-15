# FNP-P2C-004 Behavior Extraction Ledger

Packet: `FNP-P2C-004-A`  
Subsystem: `NDIter traversal/index semantics`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C004-C01` | `nditer` constructor normalizes operands/flags/op_axes/itershape and rejects invalid combinations | deterministic constructor acceptance/rejection for same inputs | same acceptance/rejection class; malformed metadata/flags fail-closed with stable reason code | `nditer_pywrap.c:707`, `nditer_constr.c:111`, `test_nditer.py:768`, `test_nditer.py:843` |
| `P2C004-C02` | Multi-index traversal order reflects iterator permutation and source layout (C/F/reversed views) | `multi_index` sequences match legacy across memory/order permutations | same sequence guarantees; unsupported mode combinations fail-closed | `nditer_api.c:1001`, `test_nditer.py:231`, `test_nditer.py:258` |
| `P2C004-C03` | `c_index`/`f_index` provide stable flat indices under the same traversal path | deterministic `c_index` and `f_index` streams for fixed input/layout | same stream values with identical bounds/error semantics | `nditer_api.c:530`, `test_nditer.py:434`, `test_nditer.py:511` |
| `P2C004-C04` | Setting `multi_index` repositions iterator with bounds validation | in-bounds seek succeeds and resets state; out-of-bounds raises index error class | same seek semantics with deterministic diagnostics | `nditer_api.c:440`, `nditer_pywrap.c:1613`, `nditer_pywrap.c:1685`, `test_nditer.py:197` |
| `P2C004-C05` | Setting `index`/`iterindex`/`iterrange` repositions iteration cursor consistently | cursor control follows requested mode and range restrictions | same behavior; invalid range/index fails-closed with class stability | `nditer_api.c:530`, `nditer_api.c:618`, `nditer_pywrap.c:1706`, `nditer_pywrap.c:1799`, `nditer_pywrap.c:1859`, `test_nditer.py:1849`, `test_nditer.py:1883` |
| `P2C004-C06` | `remove_axis`/`remove_multi_index` enforce preconditions and reset state | transition calls are allowed only when documented constraints hold | same constraints; rejected transitions preserve fail-closed behavior | `nditer_api.c:30`, `nditer_api.c:160`, `nditer_pywrap.c:1368`, `nditer_pywrap.c:1398`, `test_nditer.py:1808`, `test_nditer.py:1820` |
| `P2C004-C07` | `enable_external_loop` is incompatible with index/multi-index modes (unless allowed by buffering/range rules) | invalid flag combinations produce stable `ValueError`-class behavior | same behavior and deterministic reason-code audit | `nditer_api.c:187`, `nditer_pywrap.c:1426`, `test_nditer.py:3338` |
| `P2C004-C08` | Per-operand `no_broadcast` prevents expansion of flagged operands | no-broadcast operands fail when broadcast would be required | same failure class with bounded diagnostics | `nditer_pywrap.c:330`, `test_nditer.py:2603` |
| `P2C004-C09` | `copy_if_overlap` and `overlap_assume_elementwise` control overlap-copy behavior | read/write overlap triggers copy unless safe elementwise assumptions apply | same outward result; hardened mode logs overlap-risk decisions deterministically | `test_nditer.py:1325` |
| `P2C004-C10` | `flatiter` indexing/assignment supports integer/slice/fancy/boolean cases with stable error classes | valid index/assign forms succeed; invalid forms raise documented index errors | same behavioral surface; malformed forms fail-closed | `iterators.c:491`, `iterators.c:746`, `test_indexing.py:1450`, `test_indexing.py:1602`, `test_multiarray.py:10031` |
| `P2C004-C11` | `ndenumerate` and `ndindex` coordinate streams align with ndarray shape law and each other | index tuple stream is deterministic; invalid shapes (e.g., negative dimensions) fail predictably | same observable stream/failure behavior with strict error-class stability | `_index_tricks_impl.py:591`, `_index_tricks_impl.py:641`, `test_index_tricks.py:366`, `test_index_tricks.py:552`, `test_index_tricks.py:629`, `test_index_tricks.py:678` |

## 2. Compatibility Invariants

1. Iterator state-machine invariant: constructor flags/modes determine legal transitions and illegal transitions fail with stable error classes.
2. Traversal determinism invariant: for fixed operands/layout/op_axes/flags, iteration value/index/multi-index stream is deterministic.
3. Index coherence invariant: `multi_index`, `index`, and `iterindex` refer to the same logical iterator state and seeking one updates the others consistently.
4. Broadcast policy invariant: `no_broadcast` operands are never silently expanded.
5. Overlap-safety invariant: `copy_if_overlap` never permits unsafe read/write overlap without the documented opt-in assumptions.
6. Flat-surface invariant: `flatiter`, `ndindex`, and `ndenumerate` preserve documented indexing/order/error semantics.

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C004-U01` | `fnp-iter` crate is currently a stub; no dedicated iterator state machine exists yet. | high | `bd-23m.15.4` | Land iterator boundary skeleton with constructor, cursor, and mode-transition APIs plus baseline tests. |
| `P2C004-U02` | Full parity for `op_axes`/`itershape` interactions (especially complex reductions and mixed-broadcast remaps) is not encoded in Rust contracts. | high | `bd-23m.15.2` | Contract table captures complete preconditions/postconditions/error taxonomy for op_axes and shaped iteration. |
| `P2C004-U03` | Hardened policy for overlap-sensitive iterator operations (`copy_if_overlap`, writeback-like flows) lacks packet-scoped threat controls. | high | `bd-23m.15.3` | Threat model explicitly defines fail-closed/full-validate boundaries and required audit reason codes. |
| `P2C004-U04` | Differential/adversarial corpus does not yet cover iterator-seeking and flatiter indexing edge matrix. | high | `bd-23m.15.6` | Add oracle-backed iterator fixture suites for seek bounds, op_axes errors, and flatiter indexing/assignment families. |
| `P2C004-U05` | Workflow replay artifacts do not currently include packet-specific nditer/flatiter journey traces. | medium | `bd-23m.15.7` | Add e2e scenarios with step-level logs and links to iterator unit/differential fixture IDs. |
| `P2C004-U06` | Exact parity for ndindex/ndenumerate non-integer-dimension/type-error nuances is not yet pinned in Rust tests. | medium | `bd-23m.15.5` | Unit/property suite asserts shape-validation and stream-equivalence behavior against legacy expectations. |

## 4. Planned Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | Introduce iterator-state tests for constructor flags, seek transitions, index-mode coherence, and no-broadcast checks | `crates/fnp-iter/src/lib.rs` + packet-E test modules (`bd-23m.15.5`) |
| Differential/metamorphic/adversarial | Build fixture corpus for nditer/flatiter/ndindex/ndenumerate behavior classes and compare against legacy oracle | `crates/fnp-conformance/fixtures/` + packet-F harness additions (`bd-23m.15.6`) |
| E2E | Add workflow scenarios that exercise iterator construction -> traversal -> seek/reset -> flat assignment paths in strict/hardened modes | `scripts/e2e/run_workflow_scenario_gate.sh` + `artifacts/logs/` (`bd-23m.15.7`) |
| Structured logging | Enforce `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code` on iterator test/e2e outputs | `artifacts/contracts/test_logging_contract_v1.json`, conformance gate outputs |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: any hardened-mode iterator policy intervention must carry explicit state/action/loss rationale and replay metadata.
- Optimization gate: iterator performance changes require baseline/profile + single-lever change + behavior-isomorphism proof.
- EV gate: optimization levers ship only when `EV >= 2.0`; otherwise remain explicit deferred debt.
- RaptorQ scope: packet-I closure artifacts must include sidecar/scrub/decode-proof linkage for durable evidence bundles.

## 6. Rollback Handle

If packet-local iterator behavior drifts, rollback by restoring `artifacts/phase2c/FNP-P2C-004/*` to the prior green baseline and re-running the packet’s unit/differential/e2e gates before reapplying changes.
