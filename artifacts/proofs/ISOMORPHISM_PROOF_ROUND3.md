# ISOMORPHISM_PROOF_ROUND3

## Change: Contiguous Axis Reduction Kernel In `fnp-ufunc::reduce_sum(Some(axis), ...)`

- Ordering preserved: yes; output remains C-order over the reduced output shape.
- Tie-breaking unchanged: N/A (deterministic numeric accumulation; no comparator tie-breaks).
- Floating-point: accumulation order per reduced output element is preserved (`axis` index ascends identically to prior flat traversal bucket order).
- RNG seeds: N/A.
- Golden outputs: `sha256sum -c artifacts/proofs/golden_checksums_round3.txt` passed.

## Behavioral Evidence

- `cargo test -p fnp-ufunc -- --nocapture` passed (includes new axis-order + empty-axis tests).
- `cargo test -p fnp-conformance benchmark::tests::baseline_generator_writes_json -- --nocapture` passed.
- `sha256sum -c artifacts/proofs/golden_checksums_round3.txt` passed.

## Performance Evidence

- `artifacts/optimization/hyperfine_generate_benchmark_baseline_round3_before.json`
- `artifacts/optimization/hyperfine_generate_benchmark_baseline_round3_after.json`
- `artifacts/baselines/ufunc_benchmark_baseline_round3_before.json`
- `artifacts/baselines/ufunc_benchmark_baseline_round3_after.json`
- `artifacts/optimization/strace_generate_benchmark_baseline_round3_before.txt`
- `artifacts/optimization/strace_generate_benchmark_baseline_round3_after.txt`
- Hyperfine mean command latency improved from `22.924 ms` to `10.070 ms` (`-56.07%`).
- Targeted workload `reduce_sum_axis1_keepdims_false_256x256` p50 improved from `0.460362 ms` to `0.044763 ms` (`-90.28%`).

## Regression Surface

- No API or type-shape contract changes in `UFuncArray`.
- Strict/hardened runtime policy behavior unchanged.
- Add-workload p95/p99 moved upward in this short sample run; guard policy is to rollback if sustained p99 regression exceeds 25% across 3 consecutive captures.

