#![forbid(unsafe_code)]

use fnp_conformance::raptorq_artifacts::generate_bundle_sidecar_and_reports;

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_raptorq_sidecars failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture_root =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fnp-conformance/fixtures");

    let conformance_files = vec![
        fixture_root.join("ufunc_input_cases.json"),
        fixture_root.join("oracle_outputs/ufunc_oracle_output.json"),
        fixture_root.join("oracle_outputs/ufunc_differential_report.json"),
    ];

    generate_bundle_sidecar_and_reports(
        "conformance_bundle_v1",
        &repo_root,
        &conformance_files,
        &repo_root.join("artifacts/raptorq/conformance_bundle_v1.sidecar.json"),
        &repo_root.join("artifacts/raptorq/conformance_bundle_v1.scrub_report.json"),
        &repo_root.join("artifacts/raptorq/conformance_bundle_v1.decode_proof.json"),
        1001,
    )?;

    let benchmark_files = vec![repo_root.join("artifacts/baselines/ufunc_benchmark_baseline.json")];

    generate_bundle_sidecar_and_reports(
        "benchmark_bundle_v1",
        &repo_root,
        &benchmark_files,
        &repo_root.join("artifacts/raptorq/benchmark_bundle_v1.sidecar.json"),
        &repo_root.join("artifacts/raptorq/benchmark_bundle_v1.scrub_report.json"),
        &repo_root.join("artifacts/raptorq/benchmark_bundle_v1.decode_proof.json"),
        1002,
    )?;

    println!("generated RaptorQ sidecars and reports for conformance + benchmark bundles");
    Ok(())
}
