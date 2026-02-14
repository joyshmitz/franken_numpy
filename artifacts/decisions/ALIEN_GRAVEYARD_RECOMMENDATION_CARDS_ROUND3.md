# ALIEN_GRAVEYARD_RECOMMENDATION_CARDS_ROUND3

## Card 1

Change:
- Replace `reduce_sum` axis-path `unravel/ravel` mapping with contiguous kernel traversal.

Hotspot evidence:
- Pre-change benchmark artifact showed `reduce_sum_axis1_keepdims_false_256x256` p50 at `0.460362 ms`.
- Command-level baseline (`generate_benchmark_baseline`) mean was `22.924 ms`.

Mapped graveyard sections:
- `alien_cs_graveyard.md` §0.1, §0.2, §0.3, §0.15, §8.2
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.2, §0.3, §0.12

EV score (Impact * Confidence * Reuse / Effort * Friction):
- `(5 * 5 * 4) / (2 * 2) = 25.0`

Priority tier (S/A/B/C):
- A

Adoption wedge (boundary/compatibility/rollout):
- Internal kernel implementation change in `crates/fnp-ufunc/src/lib.rs` only.
- No API surface change; no dtype/broadcast contract change.

Budgeted mode (default budget + on-exhaustion behavior):
- Runtime bound: deterministic `O(nelems)` loops with no recursion.
- Memory bound: output vector + constant loop state.
- On budget breach symptom (sustained p99 regression gate), rollback to previous implementation.

Expected-loss model (states/actions/loss):
- States `S = {isomorphic+faster, isomorphic+neutral, non_isomorphic, tail_regressed}`
- Actions `A = {ship, rollback}`
- Loss matrix (representative):
  - `L(ship, non_isomorphic)=1000`
  - `L(ship, tail_regressed)=120`
  - `L(ship, isomorphic+neutral)=20`
  - `L(ship, isomorphic+faster)=1`
  - `L(rollback, isomorphic+faster)=40`
  - `L(rollback, non_isomorphic)=1`
- Decision rule: ship only when empirical evidence keeps expected loss below rollback path.

Calibration + fallback trigger:
- Trigger rollback if:
  - Any isomorphism evidence fails (`tests` or checksum mismatch), or
  - `ufunc_add_broadcast_256x256_by_256` p99 regresses by >25% over 3 consecutive captures.

Isomorphism proof plan:
- Preserve output ordering semantics.
- Verify via unit tests + golden checksum manifest + conformance benchmark generator.

p50/p95/p99 before/after target:
- Targeted workload (`reduce_sum_axis1_keepdims_false_256x256`) achieved:
  - p50: `0.460362 -> 0.044763` (`-90.28%`)
  - p95: `0.478396 -> 0.047589` (`-90.05%`)
  - p99: `0.566190 -> 0.057057` (`-89.92%`)

Primary failure risk + countermeasure:
- Risk: axis-order accumulation mismatch for non-last-axis reductions.
- Countermeasure: explicit 3D `axis=0` ordering test + empty-axis test + checksum verification.

Repro artifact pack (env/manifest/repro.lock/legal/provenance):
- `artifacts/optimization/hyperfine_generate_benchmark_baseline_round3_before.json`
- `artifacts/optimization/hyperfine_generate_benchmark_baseline_round3_after.json`
- `artifacts/baselines/ufunc_benchmark_baseline_round3_before.json`
- `artifacts/baselines/ufunc_benchmark_baseline_round3_after.json`
- `artifacts/proofs/golden_checksums_round3.txt`
- `artifacts/proofs/ISOMORPHISM_PROOF_ROUND3.md`

Primary paper status (hypothesis/read/reproduced + checklist state):
- Internal optimization pattern reproduced (no new external-paper claim).

Interference test status (required when composing controllers):
- N/A (no additional adaptive controller composition in this change).

Demo linkage (`demo_id` + `claim_id`, if production-facing):
- `demo_id: ufunc-reduce-sum-round3`
- `claim_id: fnp-ufunc-reduce-contiguous-kernel-v1`

Rollback:
- Revert `crates/fnp-ufunc/src/lib.rs` change via `git revert <round3_commit_sha>`.

Baseline comparator (what are we beating?):
- Prior per-element `unravel/ravel` axis reduction path.

## Card 2

Change:
- Propagate round-3 evidence expectations to active/open execution beads through explicit artifact references and sequencing guidance.

Hotspot evidence:
- `bv --robot-next` and `bv --robot-insights` identify `bd-23m.5`, `bd-23m.4`, `bd-23m.1`, and packet-chain A/B/C tasks as highest leverage blockers.

Mapped graveyard sections:
- `alien_cs_graveyard.md` §0.2, §0.3, §0.19, §0.20
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` Project-Level Decision Contracts, Evidence Ledger Schema + CI Gates

EV score (Impact * Confidence * Reuse / Effort * Friction):
- `(4 * 4 * 5) / (2 * 2) = 20.0`

Priority tier (S/A/B/C):
- A

Adoption wedge (boundary/compatibility/rollout):
- `br comments` updates on open bead chain, no scope reduction and no feature deletion.

Budgeted mode (default budget + on-exhaustion behavior):
- Budget: comments/metadata-only updates, no forced dependency rewrites.
- On uncertainty: log recommendation in bead comments and defer graph rewiring.

Expected-loss model (states/actions/loss):
- States `{bead_actionable, bead_under_specified}`
- Actions `{execute_now, refine_then_execute}`
- Loss favors refining under-specified critical-path beads before implementation.

Calibration + fallback trigger:
- If `bv --robot-alerts` reports critical alerts, halt implementation updates and repair graph first.

Isomorphism proof plan:
- Preserve bead intent; augment only with measurable artifact references and guardrails.

p50/p95/p99 before/after target:
- Planning-space task (N/A runtime metric); success measured by critical-path clarity and artifact linkage completeness.

Primary failure risk + countermeasure:
- Risk: over-constraining dependencies from low-signal auto-suggestions.
- Countermeasure: accept only high-confidence, context-validated sequencing updates.

Repro artifact pack:
- `bv --robot-next`
- `bv --robot-insights`
- `bv --robot-alerts`

Rollback:
- Remove added comments or revert `.beads/issues.jsonl` updates.

Baseline comparator:
- Open beads without round-3 artifact linkage context.

