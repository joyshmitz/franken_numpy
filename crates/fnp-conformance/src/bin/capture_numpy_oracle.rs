#![forbid(unsafe_code)]

use fnp_conformance::HarnessConfig;
use fnp_conformance::ufunc_differential::capture_numpy_oracle;

fn main() {
    if let Err(err) = run() {
        eprintln!("capture_numpy_oracle failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cfg = HarnessConfig::default_paths();
    let input_path = cfg.fixture_root.join("ufunc_input_cases.json");
    let output_path = cfg
        .fixture_root
        .join("oracle_outputs/ufunc_oracle_output.json");

    let capture = capture_numpy_oracle(&input_path, &output_path, &cfg.oracle_root)?;
    println!(
        "captured {} oracle cases using {}",
        capture.cases.len(),
        capture.oracle_source
    );
    println!("wrote {}", output_path.display());
    Ok(())
}
