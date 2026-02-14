# ROUND3_OPPORTUNITY_MATRIX

## Scope

Round 3 applies one optimization lever to `fnp-ufunc::UFuncArray::reduce_sum(Some(axis), ...)`.
The previous implementation performed per-element `unravel_index + ravel_index` mapping. The new implementation uses a contiguous axis-reduction kernel (`outer/axis/inner` traversal) with no public API changes.

## Baseline + Profile Evidence

- Baseline command:
  - `hyperfine --warmup 3 --runs 10 '/data/tmp/cargo-target/release/generate_benchmark_baseline'`
- Pre-change artifacts:
  - `artifacts/optimization/hyperfine_generate_benchmark_baseline_round3_before.json`
  - `artifacts/baselines/ufunc_benchmark_baseline_round3_before.json`
  - `artifacts/optimization/strace_generate_benchmark_baseline_round3_before.txt`
- Post-change artifacts:
  - `artifacts/optimization/hyperfine_generate_benchmark_baseline_round3_after.json`
  - `artifacts/baselines/ufunc_benchmark_baseline_round3_after.json`
  - `artifacts/optimization/strace_generate_benchmark_baseline_round3_after.txt`
- Profiler fallback:
  - `perf` unavailable in this environment (`perf_event_paranoid=4`), so syscall-level profile fallback was used (`strace -c`).

## Opportunity Matrix

| Hotspot | Impact (1-5) | Confidence (1-5) | Effort (1-5) | Score (Impact*Confidence/Effort) | Decision |
|---|---:|---:|---:|---:|---|
| `fnp-ufunc::reduce_sum` axis path reindex overhead (`unravel/ravel` per element) | 5 | 5 | 2 | 12.5 | Implemented (Round 3) |
| `fnp-ufunc::elementwise_binary` broadcast add tail jitter | 2 | 3 | 2 | 3.0 | Monitor only (no lever this round) |
| Runtime policy ledger serialization overhead | 2 | 3 | 3 | 2.0 | Deferred |

EV gate used for implemented lever:

- `EV = (Impact * Confidence * Reuse) / (Effort * AdoptionFriction)`
- `EV = (5 * 5 * 4) / (2 * 2) = 25.0` (ship)

## One-Lever Change

- Replaced generic axis reduction index mapping with a contiguous reduction kernel:
  - `outer = product(shape[..axis])`
  - `inner = product(shape[axis+1..])`
  - for each output slot, accumulate along axis with `offset += inner`
- Preserved shape semantics for `keepdims=true/false`.
- Preserved zero-length axis behavior (output remains zero-initialized).
- Added focused tests:
  - non-last-axis reduction ordering (`axis=0`, 3D array)
  - empty-axis reduction behavior

## Measured Delta

### Hyperfine command latency

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| mean wall time (ms) | 22.924 | 10.070 | -12.853 ms (-56.07%) |
| stddev (ms) | 3.347 | 0.500 | -2.846 ms (-85.05%) |

### Workload percentile deltas (`generate_benchmark_baseline` output)

| Workload | p50 before | p50 after | p50 delta | p95 before | p95 after | p95 delta | p99 before | p99 after | p99 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reduce_sum_axis1_keepdims_false_256x256` | 0.460362 | 0.044763 | -90.28% | 0.478396 | 0.047589 | -90.05% | 0.566190 | 0.057057 | -89.92% |
| `reduce_sum_all_keepdims_false_256x256` | 0.049312 | 0.049332 | +0.04% | 0.054803 | 0.050114 | -8.56% | 0.121026 | 0.063639 | -47.42% |
| `ufunc_add_broadcast_256x256_by_256` | 0.190256 | 0.185988 | -2.24% | 0.368650 | 0.404527 | +9.73% | 0.501168 | 0.606795 | +21.08% |

Observation:
- The targeted reduction hotspot improved substantially.
- Add-workload tail increased in this short run, but absolute movement remains sub-millisecond; regression guard is tracked in fallback triggers below.

## Budgeted Mode + Fallback

- Time budget: deterministic `O(nelems)` traversal with fixed trip counts (`outer * axis_len * inner`).
- Memory budget: `O(out_count)` output buffer + constant extra state.
- Exhaustion behavior: no recursion, no unbounded search/retry loops.
- Fallback trigger:
  - Any isomorphism mismatch (`cargo test` or checksum mismatch) => immediate rollback.
  - Sustained add-workload p99 regression >25% across 3 consecutive benchmark captures => rollback and re-open hotspot analysis.
- Rollback command:
  - `git revert <round3_commit_sha>`

## Graveyard Mapping

- `alien_cs_graveyard.md`:
  - §0.1 Mandatory Optimization Loop
  - §0.2 Opportunity Matrix Gate
  - §0.3 Isomorphism Proof Block
  - §0.15 Tail-Latency Decomposition Requirement
  - §8.2 Vectorized Execution + Morsel-Driven Parallelism (kernel-style iteration strategy)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md`:
  - §0.2 Mandatory Optimization Loop
  - §0.3 Opportunity Matrix Gate
  - §0.12 Evidence Ledger Schema + CI Gates
  - Project-Level Decision Contracts (loss/calibration/fallback)

