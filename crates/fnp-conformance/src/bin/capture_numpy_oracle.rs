#![forbid(unsafe_code)]

use fnp_conformance::HarnessConfig;
use fnp_conformance::ufunc_differential::capture_numpy_oracle;
use std::env;

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
    if require_real_numpy_oracle() && capture.oracle_source == "pure_python_fallback" {
        return Err(
            "FNP_REQUIRE_REAL_NUMPY_ORACLE=1 but oracle capture used pure_python_fallback; set FNP_ORACLE_PYTHON to a NumPy-backed interpreter".to_string(),
        );
    }
    println!(
        "captured {} oracle cases using {}",
        capture.cases.len(),
        capture.oracle_source
    );
    println!("wrote {}", output_path.display());
    Ok(())
}

fn require_real_numpy_oracle() -> bool {
    match env::var("FNP_REQUIRE_REAL_NUMPY_ORACLE") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}
