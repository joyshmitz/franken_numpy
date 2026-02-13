# RAPTORQ_STATUS

Status: first implementation delivered.

Implemented artifacts:
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json`
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json`
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json`
- `artifacts/raptorq/benchmark_bundle_v1.sidecar.json`
- `artifacts/raptorq/benchmark_bundle_v1.scrub_report.json`
- `artifacts/raptorq/benchmark_bundle_v1.decode_proof.json`

Current scope limitations:
1. Bundle schemas are project-local and not yet versioned as external stable contracts.
2. Recovery drill currently drops one symbol per bundle; multi-symbol stress profiles are pending.
3. Artifact signing/attestation is not implemented yet.

Next expansion:
- add multi-loss recovery matrices,
- add cryptographic attestation chain,
- integrate sidecar generation into CI gate pipeline.
