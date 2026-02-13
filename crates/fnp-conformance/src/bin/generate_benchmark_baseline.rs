#![forbid(unsafe_code)]

use fnp_conformance::benchmark::generate_benchmark_baseline;

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_benchmark_baseline failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = repo_root.join("artifacts/baselines/ufunc_benchmark_baseline.json");

    let baseline = generate_benchmark_baseline(&repo_root, &output_path)?;
    println!(
        "generated benchmark baseline with {} workloads",
        baseline.workloads.len()
    );
    println!("wrote {}", output_path.display());
    Ok(())
}
