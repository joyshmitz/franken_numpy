#![forbid(unsafe_code)]

use fnp_conformance::HarnessConfig;
use fnp_conformance::ufunc_differential::{compare_against_oracle, write_differential_report};

fn main() {
    if let Err(err) = run() {
        eprintln!("run_ufunc_differential failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cfg = HarnessConfig::default_paths();
    let input_path = cfg.fixture_root.join("ufunc_input_cases.json");
    let oracle_path = cfg
        .fixture_root
        .join("oracle_outputs/ufunc_oracle_output.json");
    let report_path = cfg
        .fixture_root
        .join("oracle_outputs/ufunc_differential_report.json");

    let report = compare_against_oracle(&input_path, &oracle_path, 1e-9, 1e-9)?;
    write_differential_report(&report_path, &report)?;

    println!(
        "ufunc differential: total={} passed={} failed={}",
        report.total_cases, report.passed_cases, report.failed_cases
    );
    println!("wrote {}", report_path.display());
    Ok(())
}
