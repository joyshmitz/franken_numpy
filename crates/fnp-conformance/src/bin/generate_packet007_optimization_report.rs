#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PACKET_ID: &str = "FNP-P2C-007";
const SUBTASK_ID: &str = "FNP-P2C-007-H";

#[derive(Debug, Clone, Serialize)]
struct PercentileSummary {
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ProfileSummary {
    implementation: String,
    runs: usize,
    lookups_per_run: usize,
    percentiles: PercentileSummary,
    throughput_lookups_per_sec_p50: f64,
    throughput_lookups_per_sec_p95: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DeltaSummary {
    p50_delta_percent: f64,
    p95_delta_percent: f64,
    p99_delta_percent: f64,
    throughput_gain_percent_p50: f64,
    throughput_gain_percent_p95: f64,
}

#[derive(Debug, Clone, Serialize)]
struct LeverSummary {
    id: String,
    description: String,
    rationale: String,
    risk_note: String,
    rollback_command: String,
}

#[derive(Debug, Clone, Serialize)]
struct EVSummary {
    impact: f64,
    confidence: f64,
    reuse: f64,
    effort: f64,
    adoption_friction: f64,
    score: f64,
    promoted: bool,
}

#[derive(Debug, Clone, Serialize)]
struct IsomorphismCheck {
    case_id: String,
    status: String,
    details: String,
}

#[derive(Debug, Serialize)]
struct Packet007OptimizationReport {
    schema_version: u8,
    packet_id: String,
    subtask_id: String,
    generated_at_unix_ms: u128,
    environment_fingerprint: String,
    reproducibility_command: String,
    lever: LeverSummary,
    baseline_profile: ProfileSummary,
    rebaseline_profile: ProfileSummary,
    delta: DeltaSummary,
    ev: EVSummary,
    isomorphism_checks: Vec<IsomorphismCheck>,
}

#[derive(Debug, Deserialize)]
struct RngDifferentialFixtureCase {
    id: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_packet007_optimization_report failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = parse_output_path(&repo_root)?;

    let fixture_path =
        repo_root.join("crates/fnp-conformance/fixtures/rng_differential_cases.json");
    let case_ids = load_rng_case_ids(&fixture_path)?;
    if case_ids.is_empty() {
        return Err(format!(
            "rng differential fixture corpus is empty: {}",
            fixture_path.display()
        ));
    }
    let scaled_case_ids = expand_case_ids(&case_ids, 128);

    let selected_case_ids = select_case_ids(&scaled_case_ids, 1024);
    let query_workload = build_query_workload(&selected_case_ids, 256);
    let lookups_per_run = query_workload.len();

    let case_index_map = build_case_index_map(&scaled_case_ids);

    let baseline_profile =
        profile_implementation("baseline_linear_id_scan", 80, lookups_per_run, || {
            run_linear_workload(&scaled_case_ids, &query_workload)
        });

    let rebaseline_profile =
        profile_implementation("rebaseline_hash_map_lookup", 80, lookups_per_run, || {
            run_map_workload(&case_index_map, &query_workload)
        });

    let delta = DeltaSummary {
        p50_delta_percent: percent_delta(
            baseline_profile.percentiles.p50_ms,
            rebaseline_profile.percentiles.p50_ms,
        ),
        p95_delta_percent: percent_delta(
            baseline_profile.percentiles.p95_ms,
            rebaseline_profile.percentiles.p95_ms,
        ),
        p99_delta_percent: percent_delta(
            baseline_profile.percentiles.p99_ms,
            rebaseline_profile.percentiles.p99_ms,
        ),
        throughput_gain_percent_p50: percent_delta(
            baseline_profile.throughput_lookups_per_sec_p50,
            rebaseline_profile.throughput_lookups_per_sec_p50,
        ),
        throughput_gain_percent_p95: percent_delta(
            baseline_profile.throughput_lookups_per_sec_p95,
            rebaseline_profile.throughput_lookups_per_sec_p95,
        ),
    };

    let isomorphism_checks = run_isomorphism_checks(
        &scaled_case_ids,
        &case_index_map,
        &query_workload,
        &selected_case_ids,
    );
    let all_checks_pass = isomorphism_checks
        .iter()
        .all(|check| check.status == "pass");

    let impact = if delta.throughput_gain_percent_p95 >= 10.0 {
        4.0
    } else if delta.throughput_gain_percent_p95 > 0.0 {
        3.0
    } else {
        1.0
    };
    let confidence = if all_checks_pass { 4.0 } else { 1.0 };
    let reuse = 4.0;
    let effort = 2.0;
    let adoption_friction = 1.0;
    let score = (impact * confidence * reuse) / (effort * adoption_friction);
    let promoted = score >= 2.0
        && all_checks_pass
        && delta.throughput_gain_percent_p95 > 0.0
        && delta.p95_delta_percent < 0.0;

    let ev = EVSummary {
        impact,
        confidence,
        reuse,
        effort,
        adoption_friction,
        score,
        promoted,
    };

    let report = Packet007OptimizationReport {
        schema_version: 1,
        packet_id: PACKET_ID.to_string(),
        subtask_id: SUBTASK_ID.to_string(),
        generated_at_unix_ms: now_unix_ms(),
        environment_fingerprint: environment_fingerprint(),
        reproducibility_command:
            "cargo run -p fnp-conformance --bin generate_packet007_optimization_report"
                .to_string(),
        lever: LeverSummary {
            id: "P2C007-H-LEVER-001".to_string(),
            description:
                "Use precomputed RNG fixture ID maps for workflow-step dispatch".to_string(),
            rationale:
                "Packet-007 workflow replay repeatedly resolves fixture IDs for strict/hardened scenario steps; replacing linear scans with precomputed HashMap lookups removes O(n) per-step lookup overhead while preserving deterministic fixture resolution semantics."
                    .to_string(),
            risk_note:
                "Optimization only changes fixture ID lookup strategy in workflow dispatch; fixture payloads, execution order, and reason-code outcomes are unchanged."
                    .to_string(),
            rollback_command:
                "git restore --source <last-green-commit> -- crates/fnp-conformance/src/workflow_scenarios.rs"
                    .to_string(),
        },
        baseline_profile,
        rebaseline_profile,
        delta,
        ev,
        isomorphism_checks,
    };

    let raw = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed serializing optimization report: {err}"))?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    fs::write(&output_path, raw)
        .map_err(|err| format!("failed writing {}: {err}", output_path.display()))?;

    println!("wrote {}", output_path.display());
    Ok(())
}

fn parse_output_path(repo_root: &Path) -> Result<PathBuf, String> {
    let mut output_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output-path requires a value".to_string())?;
                output_path = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin generate_packet007_optimization_report -- [--output-path <path>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(output_path.unwrap_or_else(|| {
        repo_root.join("artifacts/phase2c/FNP-P2C-007/optimization_profile_report.json")
    }))
}

fn load_rng_case_ids(path: &Path) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<RngDifferentialFixtureCase> = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid JSON {}: {err}", path.display()))?;
    Ok(cases.into_iter().map(|case| case.id).collect())
}

fn select_case_ids(case_ids: &[String], max_count: usize) -> Vec<String> {
    let count = max_count.min(case_ids.len());
    case_ids.iter().take(count).cloned().collect()
}

fn expand_case_ids(case_ids: &[String], multiplier: usize) -> Vec<String> {
    let mut expanded = Vec::with_capacity(case_ids.len().saturating_mul(multiplier));
    for repeat in 0..multiplier {
        for case_id in case_ids {
            if repeat == 0 {
                expanded.push(case_id.clone());
            } else {
                expanded.push(format!("{case_id}::scaled::{repeat}"));
            }
        }
    }
    expanded
}

fn build_query_workload(case_ids: &[String], repetitions: usize) -> Vec<String> {
    let mut queries =
        Vec::with_capacity(repetitions.saturating_mul(case_ids.len()).saturating_mul(2));

    for rep in 0..repetitions {
        for case_id in case_ids {
            queries.push(case_id.clone());
            if rep % 8 == 0 {
                queries.push(format!("{case_id}::missing"));
            }
        }
    }

    queries
}

fn build_case_index_map(case_ids: &[String]) -> HashMap<String, usize> {
    case_ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.clone(), idx))
        .collect()
}

fn profile_implementation<F>(
    implementation: &str,
    runs: usize,
    lookups_per_run: usize,
    mut run_fn: F,
) -> ProfileSummary
where
    F: FnMut() -> usize,
{
    let mut samples_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let checksum = run_fn();
        std::hint::black_box(checksum);
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let percentiles = summarize_samples(&samples_ms);
    let throughput_lookups_per_sec_p50 =
        compute_throughput(lookups_per_run as f64, percentiles.p50_ms);
    let throughput_lookups_per_sec_p95 =
        compute_throughput(lookups_per_run as f64, percentiles.p95_ms);

    ProfileSummary {
        implementation: implementation.to_string(),
        runs,
        lookups_per_run,
        percentiles,
        throughput_lookups_per_sec_p50,
        throughput_lookups_per_sec_p95,
    }
}

fn run_linear_workload(case_ids: &[String], queries: &[String]) -> usize {
    let mut checksum = 0usize;
    for query in queries {
        if let Some(idx) = linear_lookup(case_ids, query) {
            checksum = checksum.wrapping_add(idx.wrapping_mul(31));
        } else {
            checksum = checksum.wrapping_add(query.len());
        }
    }
    checksum
}

fn run_map_workload(case_map: &HashMap<String, usize>, queries: &[String]) -> usize {
    let mut checksum = 0usize;
    for query in queries {
        if let Some(idx) = case_map.get(query) {
            checksum = checksum.wrapping_add(idx.wrapping_mul(31));
        } else {
            checksum = checksum.wrapping_add(query.len());
        }
    }
    checksum
}

fn linear_lookup(case_ids: &[String], query: &str) -> Option<usize> {
    case_ids.iter().position(|id| id == query)
}

fn run_isomorphism_checks(
    case_ids: &[String],
    case_map: &HashMap<String, usize>,
    query_workload: &[String],
    selected_case_ids: &[String],
) -> Vec<IsomorphismCheck> {
    let mut checks = Vec::new();

    for case_id in selected_case_ids.iter().take(5) {
        let linear = linear_lookup(case_ids, case_id);
        let mapped = case_map.get(case_id).copied();
        let status = if linear == mapped { "pass" } else { "fail" };
        checks.push(IsomorphismCheck {
            case_id: format!("lookup_hit::{case_id}"),
            status: status.to_string(),
            details: format!("linear={linear:?} mapped={mapped:?}"),
        });
    }

    let missing_id = "rng::nonexistent::fixture";
    let linear_missing = linear_lookup(case_ids, missing_id);
    let mapped_missing = case_map.get(missing_id).copied();
    checks.push(IsomorphismCheck {
        case_id: "lookup_miss".to_string(),
        status: if linear_missing == mapped_missing {
            "pass".to_string()
        } else {
            "fail".to_string()
        },
        details: format!("linear={linear_missing:?} mapped={mapped_missing:?}"),
    });

    let linear_checksum = run_linear_workload(case_ids, query_workload);
    let mapped_checksum = run_map_workload(case_map, query_workload);
    checks.push(IsomorphismCheck {
        case_id: "full_workload_checksum".to_string(),
        status: if linear_checksum == mapped_checksum {
            "pass".to_string()
        } else {
            "fail".to_string()
        },
        details: format!("linear_checksum={linear_checksum} mapped_checksum={mapped_checksum}"),
    });

    checks
}

fn summarize_samples(samples_ms: &[f64]) -> PercentileSummary {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    PercentileSummary {
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        p99_ms: percentile(&sorted, 0.99),
        min_ms: *sorted.first().unwrap_or(&0.0),
        max_ms: *sorted.last().unwrap_or(&0.0),
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let max_index = sorted.len().saturating_sub(1);
    let idx = ((max_index as f64) * p).round() as usize;
    sorted[idx.min(max_index)]
}

fn compute_throughput(lookups_per_run: f64, millis: f64) -> f64 {
    if millis <= f64::EPSILON {
        return 0.0;
    }
    lookups_per_run / (millis / 1000.0)
}

fn percent_delta(baseline: f64, rebaseline: f64) -> f64 {
    if baseline.abs() <= f64::EPSILON {
        return 0.0;
    }
    ((rebaseline - baseline) / baseline) * 100.0
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn environment_fingerprint() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|line| line.trim().to_string())
        .unwrap_or_else(|| "rustc unknown".to_string());

    let cpus = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);

    format!("os={os} arch={arch} cpus={cpus} rustc={rustc}")
}
