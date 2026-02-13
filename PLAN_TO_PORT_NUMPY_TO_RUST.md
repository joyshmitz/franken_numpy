# PLAN_TO_PORT_NUMPY_TO_RUST

## 1. Porting Methodology (Mandatory)

This project follows the spec-first `porting-to-rust` workflow:

1. Extract legacy behavior into executable spec artifacts.
2. Implement from spec artifacts, never line-by-line translation.
3. Prove parity with differential conformance harnesses.
4. Gate optimizations with behavior-isomorphism artifacts.

## 2. Canonical References

Primary oracle and spec set:

- Legacy oracle: `/dp/franken_numpy/legacy_numpy_code/numpy`
- FrankenNumPy spec: `COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1.md`
- FrankenSQLite exemplar (copied): `COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md`
- FrankenSQLite original path: `/dp/frankensqlite/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md`

The FrankenSQLite spec is a required style/quality baseline for evidence artifacts, mode split, durability contracts, and gate topology.

## 3. V1 Included Scope

- ndarray shape/stride/layout legality (SCE foundation)
- reshape/broadcast legality and deterministic output shape
- core dtype promotion/cast policy for scoped dtypes
- strict/hardened runtime mode split with fail-closed behavior
- fixture-driven conformance harness for shape/stride/promotion/policy
- initial npy/npz IO and RNG contracts (first-wave extraction + implementation)

## 4. Explicit V1 Exclusions

- full long-tail NumPy public API surface
- packaging/doc/build ecosystems from legacy tree
- f2py, C-extension ABI compatibility, Python runtime embedding
- advanced linalg/fft breadth beyond scoped parity packets
- non-critical modules until shape/stride/promotion/ufunc core is parity-green

## 5. Phase Plan and Exit Gates

### Phase 1: Bootstrap + Planning

Deliverables:
- four canonical docs present and coherent
- explicit exclusions locked

Exit gate:
- scope/exclusion sign-off complete

### Phase 2: Deep Extraction

Deliverables:
- packetized extraction (`FNP-P2C-*`) with anchor maps and contracts
- fixture manifests per packet

Exit gate:
- all packets marked `READY_FOR_IMPL`

### Phase 3: Architecture Synthesis

Deliverables:
- crate boundaries and compatibility mode matrix
- runtime decision/evidence schema

Exit gate:
- architecture doc maps every extraction packet to crates

### Phase 4: Implementation

Deliverables:
- one vertical slice at a time (SCE, dtype, dispatch, io, random)
- strict/hardened policy enforcement in runtime

Exit gate:
- scoped conformance suites green for each implemented family

### Phase 5: Conformance + QA

Deliverables:
- differential parity report vs legacy oracle
- benchmark deltas + isomorphism proofs
- RaptorQ sidecar + scrub/decode proofs for durable artifacts

Exit gate:
- G1-G6 gates pass from comprehensive spec section 18

## 6. Method-Stack Artifact Requirements (Per Meaningful Change)

1. `alien-artifact-coding`: decision ledger entry + explicit loss/decision rule.
2. `extreme-software-optimization`: baseline/profile matrix + one-lever proof.
3. `RaptorQ-everywhere`: sidecar manifest + scrub/decode evidence (or explicit defer note).
4. `frankenlibc/frankenfs doctrine`: strict/hardened behavior matrix + fail-closed checks.

## 7. Immediate Priorities (Post-This-Turn)

1. Implement `FNP-P2C-001` and `FNP-P2C-002` beyond smoke-level coverage.
2. Build differential fixture capture from legacy oracle tests.
3. Add first benchmark baseline for broadcast/reduction sentinel workloads.
4. Wire RaptorQ sidecar generation pipeline for conformance bundles.

## 8. Mandatory Post-Change Commands

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --workspace
```

If conformance/bench crates exist:

```bash
cargo test -p fnp-conformance -- --nocapture
cargo bench
```
