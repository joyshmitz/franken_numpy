#![forbid(unsafe_code)]

use fnp_conformance::workflow_scenarios::{
    run_user_workflow_scenario_suite, set_workflow_scenario_log_path,
};
use fnp_conformance::{HarnessConfig, SuiteReport};
use serde::Serialize;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
struct SuiteSummary {
    suite: String,
    case_count: usize,
    pass_count: usize,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct GateSummary {
    status: &'static str,
    workflow_log: String,
    suites: Vec<SuiteSummary>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_workflow_scenario_gate failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut log_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--log-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--log-path requires a value".to_string())?;
                log_path = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin run_workflow_scenario_gate -- [--log-path <path>]"
                );
                return Ok(());
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    let ts_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis());
    let log_path = log_path.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/logs")
            .join(format!("workflow_scenario_e2e_{ts_millis}.jsonl"))
    });
    set_workflow_scenario_log_path(Some(log_path.clone()));

    let cfg = HarnessConfig::default_paths();
    let suite = run_user_workflow_scenario_suite(&cfg)?;

    let status = if suite.all_passed() { "pass" } else { "fail" };
    let summary = GateSummary {
        status,
        workflow_log: log_path.display().to_string(),
        suites: vec![summarize_suite(suite)],
    };

    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("failed serializing summary: {err}"))?;
    println!("{summary_json}");

    if status == "fail" {
        std::process::exit(2);
    }
    Ok(())
}

fn summarize_suite(report: SuiteReport) -> SuiteSummary {
    SuiteSummary {
        suite: report.suite.to_string(),
        case_count: report.case_count,
        pass_count: report.pass_count,
        failures: report.failures,
    }
}
