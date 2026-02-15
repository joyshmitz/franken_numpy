#![forbid(unsafe_code)]

use fnp_conformance::workflow_scenarios::{
    run_user_workflow_scenario_suite, set_workflow_scenario_log_path,
    set_workflow_scenario_log_required,
};
use fnp_conformance::{HarnessConfig, SuiteReport};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
struct SuiteSummary {
    suite: String,
    case_count: usize,
    pass_count: usize,
    failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AttemptSummary {
    attempt: usize,
    status: String,
    workflow_log: String,
    suite: SuiteSummary,
}

#[derive(Debug, Serialize)]
struct ReliabilityDiagnostic {
    subsystem: String,
    reason_code: String,
    message: String,
    evidence_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReliabilitySummary {
    retries: usize,
    attempts_run: usize,
    flaky_failures: usize,
    flake_budget: usize,
    coverage_floor: f64,
    coverage_ratio: f64,
    diagnostics: Vec<ReliabilityDiagnostic>,
}

#[derive(Debug, Serialize)]
struct GateSummary {
    status: &'static str,
    workflow_log: String,
    attempts: Vec<AttemptSummary>,
    suites: Vec<SuiteSummary>,
    reliability: ReliabilitySummary,
    report_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WorkflowScenarioFixtureCase {
    id: String,
}

#[derive(Debug)]
struct GateOptions {
    log_path: PathBuf,
    retries: usize,
    flake_budget: usize,
    coverage_floor: f64,
    report_path: Option<PathBuf>,
}

struct WorkflowLogRequirementGuard;

impl Drop for WorkflowLogRequirementGuard {
    fn drop(&mut self) {
        set_workflow_scenario_log_required(false);
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_workflow_scenario_gate failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;

    set_workflow_scenario_log_required(true);
    let _log_requirement_guard = WorkflowLogRequirementGuard;

    let cfg = HarnessConfig::default_paths();
    let mut attempts = Vec::new();

    for attempt in 0..=options.retries {
        let attempt_log_path = derive_attempt_log_path(&options.log_path, attempt);
        set_workflow_scenario_log_path(Some(attempt_log_path.clone()));

        let suite = run_user_workflow_scenario_suite(&cfg)?;
        let mut suite_summary = summarize_suite(suite);
        if let Err(err) = validate_workflow_log_coverage(&attempt_log_path, &cfg.fixture_root) {
            suite_summary
                .failures
                .push(format!("workflow_log_validation: {err}"));
        }

        let attempt_status = if suite_summary.pass_count == suite_summary.case_count
            && suite_summary.failures.is_empty()
        {
            "pass".to_string()
        } else {
            "fail".to_string()
        };

        attempts.push(AttemptSummary {
            attempt,
            status: attempt_status.clone(),
            workflow_log: attempt_log_path.display().to_string(),
            suite: suite_summary,
        });

        if attempt_status == "pass" {
            break;
        }
    }

    let pass_attempt_index = attempts.iter().position(|attempt| attempt.status == "pass");
    let final_attempt = attempts
        .last()
        .ok_or_else(|| "workflow gate produced no attempts".to_string())?;
    let workflow_log = final_attempt.workflow_log.clone();
    let final_suite = final_attempt.suite.clone();
    let coverage_ratio = coverage_ratio(&final_suite);
    let flaky_failures = pass_attempt_index.unwrap_or(0);

    let mut diagnostics = Vec::new();
    if pass_attempt_index.is_none() {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "workflow_scenarios".to_string(),
            reason_code: "deterministic_failure".to_string(),
            message: "workflow scenario suite did not pass within retry budget".to_string(),
            evidence_refs: vec![workflow_log.clone()],
        });
    }
    if flaky_failures > options.flake_budget {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "workflow_scenarios".to_string(),
            reason_code: "flake_budget_exceeded".to_string(),
            message: format!(
                "flaky failures {} exceeded flake budget {}",
                flaky_failures, options.flake_budget
            ),
            evidence_refs: attempts
                .iter()
                .map(|attempt| attempt.workflow_log.clone())
                .collect(),
        });
    }
    if coverage_ratio + f64::EPSILON < options.coverage_floor {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "workflow_scenarios".to_string(),
            reason_code: "coverage_floor_breach".to_string(),
            message: format!(
                "coverage ratio {:.6} is below floor {:.6}",
                coverage_ratio, options.coverage_floor
            ),
            evidence_refs: vec![workflow_log.clone()],
        });
    }

    let status = if diagnostics.is_empty() {
        "pass"
    } else {
        "fail"
    };
    let attempts_run = attempts.len();
    let summary = GateSummary {
        status,
        workflow_log,
        attempts,
        suites: vec![final_suite],
        reliability: ReliabilitySummary {
            retries: options.retries,
            attempts_run,
            flaky_failures,
            flake_budget: options.flake_budget,
            coverage_floor: options.coverage_floor,
            coverage_ratio,
            diagnostics,
        },
        report_path: options
            .report_path
            .as_ref()
            .map(|path| path.display().to_string()),
    };

    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("failed serializing summary: {err}"))?;

    if let Some(report_path) = options.report_path {
        if let Some(parent) = report_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed creating report directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        fs::write(&report_path, summary_json.as_bytes())
            .map_err(|err| format!("failed writing report {}: {err}", report_path.display()))?;
    }

    println!("{summary_json}");

    if status == "fail" {
        std::process::exit(2);
    }
    Ok(())
}

fn parse_args() -> Result<GateOptions, String> {
    let mut log_path: Option<PathBuf> = None;
    let mut report_path: Option<PathBuf> = None;
    let mut retries = 0usize;
    let mut flake_budget = 0usize;
    let mut coverage_floor = 1.0f64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--log-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--log-path requires a value".to_string())?;
                log_path = Some(PathBuf::from(value));
            }
            "--report-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--report-path requires a value".to_string())?;
                report_path = Some(PathBuf::from(value));
            }
            "--retries" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--retries requires a value".to_string())?;
                retries = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --retries value '{value}': {err}"))?;
            }
            "--flake-budget" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--flake-budget requires a value".to_string())?;
                flake_budget = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --flake-budget value '{value}': {err}"))?;
            }
            "--coverage-floor" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--coverage-floor requires a value".to_string())?;
                coverage_floor = value
                    .parse::<f64>()
                    .map_err(|err| format!("invalid --coverage-floor value '{value}': {err}"))?;
                if !(0.0..=1.0).contains(&coverage_floor) {
                    return Err(format!(
                        "--coverage-floor must be between 0.0 and 1.0, got {coverage_floor}"
                    ));
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin run_workflow_scenario_gate -- [--log-path <path>] [--report-path <path>] [--retries <n>] [--flake-budget <n>] [--coverage-floor <ratio>]"
                );
                std::process::exit(0);
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

    Ok(GateOptions {
        log_path,
        retries,
        flake_budget,
        coverage_floor,
        report_path,
    })
}

fn summarize_suite(report: SuiteReport) -> SuiteSummary {
    SuiteSummary {
        suite: report.suite.to_string(),
        case_count: report.case_count,
        pass_count: report.pass_count,
        failures: report.failures,
    }
}

fn validate_workflow_log_coverage(log_path: &Path, fixture_root: &Path) -> Result<(), String> {
    let expected_ids =
        load_expected_scenario_ids(&fixture_root.join("workflow_scenario_corpus.json"))?;
    if expected_ids.is_empty() {
        return Err("workflow scenario corpus did not define any scenario ids".to_string());
    }

    let raw = fs::read_to_string(log_path)
        .map_err(|err| format!("failed reading workflow log {}: {err}", log_path.display()))?;
    if raw.lines().all(|line| line.trim().is_empty()) {
        return Err(format!("workflow log is empty: {}", log_path.display()));
    }

    let mut actual_ids = BTreeSet::new();
    for (idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).map_err(|err| {
            format!(
                "workflow log parse error {} line {}: {err}",
                log_path.display(),
                idx + 1
            )
        })?;
        let scenario_id = value
            .get("scenario_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!(
                    "workflow log missing scenario_id {} line {}",
                    log_path.display(),
                    idx + 1
                )
            })?;
        if scenario_id.trim().is_empty() {
            return Err(format!(
                "workflow log has empty scenario_id {} line {}",
                log_path.display(),
                idx + 1
            ));
        }
        actual_ids.insert(scenario_id.to_string());
    }

    if actual_ids.is_empty() {
        return Err(format!(
            "workflow log has no scenario entries: {}",
            log_path.display()
        ));
    }

    let missing: Vec<String> = expected_ids.difference(&actual_ids).cloned().collect();
    if !missing.is_empty() {
        return Err(format!(
            "workflow log missing scenario coverage for {} scenario(s): {}",
            missing.len(),
            missing.join(", ")
        ));
    }

    Ok(())
}

fn load_expected_scenario_ids(path: &Path) -> Result<BTreeSet<String>, String> {
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed reading workflow scenario corpus {}: {err}",
            path.display()
        )
    })?;
    let cases: Vec<WorkflowScenarioFixtureCase> = serde_json::from_str(&raw).map_err(|err| {
        format!(
            "invalid workflow scenario corpus json {}: {err}",
            path.display()
        )
    })?;

    Ok(cases.into_iter().map(|case| case.id).collect())
}

fn derive_attempt_log_path(base: &Path, attempt: usize) -> PathBuf {
    if attempt == 0 {
        return base.to_path_buf();
    }

    let stem = base
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("workflow_scenario_e2e");
    let ext = base.extension().and_then(|ext| ext.to_str());
    let file_name = if let Some(ext) = ext {
        format!("{stem}.attempt{attempt}.{ext}")
    } else {
        format!("{stem}.attempt{attempt}")
    };

    if let Some(parent) = base.parent() {
        parent.join(file_name)
    } else {
        PathBuf::from(file_name)
    }
}

fn coverage_ratio(summary: &SuiteSummary) -> f64 {
    if summary.case_count == 0 {
        0.0
    } else {
        summary.pass_count as f64 / summary.case_count as f64
    }
}
