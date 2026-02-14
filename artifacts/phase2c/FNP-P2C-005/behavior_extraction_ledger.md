# FNP-P2C-005 Behavior Extraction Ledger

Packet: `FNP-P2C-005-A`  
Subsystem: `Ufunc dispatch + gufunc signature`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C005-C01` | `sig` and `signature` are mutually exclusive | fail with stable type error on dual specification | same fail-closed behavior | `ufunc_object.c:3782`, `test_ufunc.py:51` |
| `P2C005-C02` | signature grammar errors must produce deterministic failure class | parse failure with stable diagnostic family | parse failure with bounded diagnostic addenda only | `ufunc_object.c:307`, `test_ufunc.py:426`, `test_ufunc.py:553` |
| `P2C005-C03` | override dispatch respects `__array_ufunc__` precedence | use override path matching legacy order | same order; reject malformed override payloads fail-closed | `override.c:206`, `test_umath.py:3447`, `test_umath.py:3973` |
| `P2C005-C04` | gufunc exception propagation preserves error visibility | propagate loop exceptions (no silent swallow) | same, plus audit reason code on policy-mediated fallback | `test_umath.py:5094` |
| `P2C005-C05` | dispatch loop/type selection is deterministic for fixed inputs/signature | deterministic loop selection and result dtype | same deterministic selection; unknown incompatibilities fail-closed | `dispatching.h:23`, `ufunc_object.c:2339`, `ufunc_object.c:4688` |
| `P2C005-C06` | reduction wrapper path honors axis/keepdims contract | stable reduction semantics and axis bounds checks | same semantics; add bounded guardrails for hostile axis payloads | `reduction.c:178`, `ufunc_object.c:2561` |

## 2. Compatibility Invariants

1. Signature normalization invariant: canonical input signature representation is unique for equivalent user inputs.
2. Dispatch determinism invariant: same operand dtypes + signature + method => same selected kernel path.
3. Override precedence invariant: no inversion in override vs built-in dispatch ordering.
4. Error-shape invariant: signature parse failures and conflict failures remain class-stable.

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C005-U01` | Exact diagnostic string stability for signature parse errors may drift across legacy micro-revisions. | medium | `bd-23m.16.6` | Differential harness stores normalized error-class taxonomy and verifies containment rules instead of exact full-string equality. |
| `P2C005-U02` | Full custom-loop registration semantics for user dtypes are not yet represented in Rust boundary. | high | `bd-23m.16.4` | Add loop registry model and conformance fixtures for registration + dispatch resolution. |
| `P2C005-U03` | Override dispatch interactions with mixed subclass hierarchies need explicit precedence matrix. | high | `bd-23m.16.5` | Add unit/property suite with subclass permutation matrix + structured logs (`reason_code=override_precedence`). |
| `P2C005-U04` | gufunc signature edge cases with optional dimensions (`?`) need explicit shrink strategy for minimal counterexamples. | medium | `bd-23m.16.5` | Add property-based signature corpus and deterministic shrink replay fields (`seed`, `artifact_refs`). |

## 4. Planned Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | Signature parse corpus + override precedence property tests + deterministic shrink replay | `crates/fnp-conformance/fixtures/ufunc_signature_property_cases.json` (planned), structured JSONL logs |
| Differential/metamorphic/adversarial | Extend ufunc differential suite with signature/override/gufunc adversarial fixtures and relation checks | `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/fixtures/ufunc_adversarial_cases.json` |
| E2E | Add ufunc dispatch scenario journey with strict/hardened replay and failure-forensics links | `scripts/e2e/run_ufunc_dispatch_journey.sh` (planned) |
| Structured logging | Ensure all packet tests emit `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code` | `artifacts/contracts/test_logging_contract_v1.json`, `scripts/e2e/run_test_contract_gate.sh` |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: dispatch/override policy decisions must log state, action, and expected-loss rationale when policy mediation occurs.
- Optimization gate: no dispatch optimization accepted without baseline/profile + single-lever + isomorphism proof artifact.
- EV gate: dispatch optimization levers promoted only when `EV >= 2.0`; otherwise tracked as deferred research.
- RaptorQ scope: durable packet evidence bundle for `FNP-P2C-005` must include sidecar/scrub/decode-proof links at packet-I closure.

## 6. Rollback Handle

If packet-local changes cause compatibility drift, rollback to previous packet boundary baseline by reverting `artifacts/phase2c/FNP-P2C-005/*` and restoring prior `fnp-ufunc` dispatch behavior snapshot from the last green differential report.
