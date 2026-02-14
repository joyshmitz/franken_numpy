#![forbid(unsafe_code)]

use fnp_conformance::{
    HarnessConfig, SuiteReport, run_runtime_policy_adversarial_suite, run_runtime_policy_suite,
    set_runtime_policy_log_path, test_contracts,
};
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const REQUIRED_LOG_FIELDS: &[&str] = &[
    "fixture_id",
    "seed",
    "mode",
    "env_fingerprint",
    "artifact_refs",
    "reason_code",
];

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
    runtime_policy_log: String,
    suites: Vec<SuiteSummary>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_test_contract_gate failed: {err}");
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
                    "Usage: cargo run -p fnp-conformance --bin run_test_contract_gate -- [--log-path <path>]"
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
            .join(format!("test_contract_e2e_{ts_millis}.jsonl"))
    });
    set_runtime_policy_log_path(Some(log_path.clone()));

    let cfg = HarnessConfig::default_paths();
    let suites = vec![
        test_contracts::run_test_contract_suite(&cfg)?,
        run_runtime_policy_suite(&cfg)?,
        run_runtime_policy_adversarial_suite(&cfg)?,
        validate_runtime_policy_log(&log_path)?,
    ];

    let status = if suites.iter().all(SuiteReport::all_passed) {
        "pass"
    } else {
        "fail"
    };

    let summary = GateSummary {
        status,
        runtime_policy_log: log_path.display().to_string(),
        suites: suites.into_iter().map(summarize_suite).collect(),
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

fn validate_runtime_policy_log(path: &Path) -> Result<SuiteReport, String> {
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed reading runtime policy log {}: {err}",
            path.display()
        )
    })?;

    let mut report = SuiteReport {
        suite: "runtime_policy_log_contract",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };

    let mut entry_count = 0usize;
    for (line_no, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        entry_count += 1;

        let value: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(err) => {
                report.case_count += 1;
                report.failures.push(format!(
                    "line {} invalid json in {}: {err}",
                    line_no + 1,
                    path.display()
                ));
                continue;
            }
        };

        let Some(obj) = value.as_object() else {
            report.case_count += 1;
            report.failures.push(format!(
                "line {} must be a JSON object in {}",
                line_no + 1,
                path.display()
            ));
            continue;
        };

        for field in REQUIRED_LOG_FIELDS {
            report.case_count += 1;
            let pass = match *field {
                "artifact_refs" => obj
                    .get(*field)
                    .and_then(Value::as_array)
                    .is_some_and(|arr| {
                        !arr.is_empty()
                            && arr
                                .iter()
                                .all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
                    }),
                "seed" => obj.get(*field).is_some_and(Value::is_u64),
                _ => obj
                    .get(*field)
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty()),
            };

            if pass {
                report.pass_count += 1;
            } else {
                report.failures.push(format!(
                    "line {} missing/invalid required field {}",
                    line_no + 1,
                    field
                ));
            }
        }
    }

    if entry_count == 0 {
        report.case_count += 1;
        report
            .failures
            .push("runtime policy log must contain at least one entry".to_string());
    } else {
        report.pass_count += 1;
        report.case_count += 1;
    }

    Ok(report)
}
