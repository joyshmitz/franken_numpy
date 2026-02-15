#![forbid(unsafe_code)]

use fnp_dtype::DType;
use fnp_ufunc::{BinaryOp, UFuncArray};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const REPRO_COMMAND: &str = "cargo run -p fnp-conformance --bin generate_benchmark_baseline";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileSummary {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkWorkload {
    pub name: String,
    pub runs: usize,
    pub samples_ms: Vec<f64>,
    pub percentiles: PercentileSummary,
    #[serde(default)]
    pub telemetry: WorkloadTelemetry,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkloadTelemetry {
    pub elements_per_run: usize,
    pub bytes_processed_per_run: usize,
    pub throughput_elements_per_sec_p50: f64,
    pub throughput_elements_per_sec_p95: f64,
    pub bandwidth_mib_per_sec_p50: f64,
    pub bandwidth_mib_per_sec_p95: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReproMetadata {
    pub command: String,
    pub repo_root: String,
    pub output_path: String,
    pub rustc_version: String,
    pub cargo_profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkBaseline {
    pub schema_version: u8,
    pub generated_at_unix_ms: u128,
    pub git_commit: String,
    pub workloads: Vec<BenchmarkWorkload>,
    #[serde(default)]
    pub environment_fingerprint: String,
    #[serde(default)]
    pub reproducibility: ReproMetadata,
    #[serde(default)]
    pub evidence_log_refs: Vec<String>,
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn git_commit_short(repo_root: &Path) -> String {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(repo_root)
        .output();

    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn rustc_version() -> String {
    let output = Command::new("rustc").arg("--version").output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn compute_per_second(units_per_run: f64, sample_ms: f64) -> f64 {
    if sample_ms <= 0.0 {
        return 0.0;
    }
    units_per_run * 1000.0 / sample_ms
}

fn relative_path_or_display(repo_root: &Path, path: &Path) -> String {
    match path.strip_prefix(repo_root) {
        Ok(relative) => relative.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

fn newest_log_with_prefix(log_dir: &Path, prefix: &str) -> Option<String> {
    let mut candidates = Vec::new();
    let Ok(entries) = fs::read_dir(log_dir) else {
        return None;
    };

    for entry in entries.filter_map(Result::ok) {
        let name_os = entry.file_name();
        let name = name_os.to_string_lossy();
        if name.starts_with(prefix) {
            candidates.push(name.to_string());
        }
    }

    candidates.sort();
    candidates.pop()
}

fn discover_evidence_log_refs(repo_root: &Path) -> Vec<String> {
    let log_dir = repo_root.join("artifacts/logs");
    let mut refs = Vec::new();
    for prefix in [
        "runtime_policy_e2e_",
        "test_contract_e2e_",
        "workflow_scenario_e2e_",
    ] {
        if let Some(file_name) = newest_log_with_prefix(&log_dir, prefix) {
            refs.push(format!("artifacts/logs/{file_name}"));
        }
    }

    if refs.is_empty() {
        refs.push("artifacts/contracts/test_logging_contract_v1.json".to_string());
        refs.push("scripts/e2e/run_test_contract_gate.sh".to_string());
        refs.push("scripts/e2e/run_workflow_scenario_gate.sh".to_string());
    }

    refs
}

fn percentile_index(len: usize, percentile_num: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let last = len - 1;
    (last * percentile_num + 50) / 100
}

fn summarize_samples(samples: &[f64]) -> PercentileSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let min_ms = sorted.first().copied().unwrap_or(0.0);
    let max_ms = sorted.last().copied().unwrap_or(0.0);
    let p50_ms = sorted
        .get(percentile_index(sorted.len(), 50))
        .copied()
        .unwrap_or(0.0);
    let p95_ms = sorted
        .get(percentile_index(sorted.len(), 95))
        .copied()
        .unwrap_or(0.0);
    let p99_ms = sorted
        .get(percentile_index(sorted.len(), 99))
        .copied()
        .unwrap_or(0.0);

    PercentileSummary {
        p50_ms,
        p95_ms,
        p99_ms,
        min_ms,
        max_ms,
    }
}

fn time_workload<F>(
    name: &str,
    runs: usize,
    elements_per_run: usize,
    bytes_processed_per_run: usize,
    mut run_fn: F,
) -> Result<BenchmarkWorkload, String>
where
    F: FnMut() -> Result<(), String>,
{
    let mut samples_ms = Vec::with_capacity(runs);

    for _ in 0..runs {
        let start = Instant::now();
        run_fn()?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        samples_ms.push(elapsed_ms);
    }

    let percentiles = summarize_samples(&samples_ms);
    let bytes_per_run_mib = bytes_processed_per_run as f64 / (1024.0 * 1024.0);
    let telemetry = WorkloadTelemetry {
        elements_per_run,
        bytes_processed_per_run,
        throughput_elements_per_sec_p50: compute_per_second(
            elements_per_run as f64,
            percentiles.p50_ms,
        ),
        throughput_elements_per_sec_p95: compute_per_second(
            elements_per_run as f64,
            percentiles.p95_ms,
        ),
        bandwidth_mib_per_sec_p50: compute_per_second(bytes_per_run_mib, percentiles.p50_ms),
        bandwidth_mib_per_sec_p95: compute_per_second(bytes_per_run_mib, percentiles.p95_ms),
    };

    Ok(BenchmarkWorkload {
        name: name.to_string(),
        runs,
        samples_ms,
        percentiles,
        telemetry,
    })
}

pub fn generate_benchmark_baseline(
    repo_root: &Path,
    output_path: &Path,
) -> Result<BenchmarkBaseline, String> {
    let item_size = DType::F64.item_size();
    let lhs_add = UFuncArray::new(
        vec![256, 256],
        (0..(256 * 256)).map(|i| f64::from(i as u32)).collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark lhs_add init failed: {err}"))?;
    let rhs_add = UFuncArray::new(
        vec![256],
        (0..256).map(|i| f64::from((i * 2) as u32)).collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark rhs_add init failed: {err}"))?;

    let reduce_in = UFuncArray::new(
        vec![256, 256],
        (0..(256 * 256))
            .map(|i| f64::from((i % 97) as u32))
            .collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark reduce init failed: {err}"))?;

    let add_elements_per_run = 256 * 256;
    let add_bytes_per_run = add_elements_per_run * item_size * 3;
    let add_workload = time_workload(
        "ufunc_add_broadcast_256x256_by_256",
        20,
        add_elements_per_run,
        add_bytes_per_run,
        || {
            let out = lhs_add
                .elementwise_binary(&rhs_add, BinaryOp::Add)
                .map_err(|err| format!("broadcast add failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let reduce_axis1_elements_per_run = 256;
    let reduce_axis1_bytes_per_run = (256 * 256 + reduce_axis1_elements_per_run) * item_size;
    let reduce_axis1 = time_workload(
        "reduce_sum_axis1_keepdims_false_256x256",
        20,
        reduce_axis1_elements_per_run,
        reduce_axis1_bytes_per_run,
        || {
            let out = reduce_in
                .reduce_sum(Some(1), false)
                .map_err(|err| format!("axis reduction failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let reduce_all_elements_per_run = 1;
    let reduce_all_bytes_per_run = (256 * 256 + reduce_all_elements_per_run) * item_size;
    let reduce_all = time_workload(
        "reduce_sum_all_keepdims_false_256x256",
        20,
        reduce_all_elements_per_run,
        reduce_all_bytes_per_run,
        || {
            let out = reduce_in
                .reduce_sum(None, false)
                .map_err(|err| format!("global reduction failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let rustc = rustc_version();
    let environment_fingerprint = format!(
        "os={} arch={} cpus={} rustc={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(1, usize::from),
        rustc
    );
    let reproducibility = ReproMetadata {
        command: REPRO_COMMAND.to_string(),
        repo_root: repo_root.display().to_string(),
        output_path: relative_path_or_display(repo_root, output_path),
        rustc_version: rustc,
        cargo_profile: "release".to_string(),
    };

    let baseline = BenchmarkBaseline {
        schema_version: 1,
        generated_at_unix_ms: now_unix_ms(),
        git_commit: git_commit_short(repo_root),
        workloads: vec![add_workload, reduce_axis1, reduce_all],
        environment_fingerprint,
        reproducibility,
        evidence_log_refs: discover_evidence_log_refs(repo_root),
    };

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let raw = serde_json::to_string_pretty(&baseline)
        .map_err(|err| format!("failed serializing baseline: {err}"))?;
    fs::write(output_path, raw)
        .map_err(|err| format!("failed writing {}: {err}", output_path.display()))?;

    Ok(baseline)
}

#[cfg(test)]
mod tests {
    use super::{BenchmarkBaseline, generate_benchmark_baseline};
    use std::fs;

    fn temp_file(name: &str) -> std::path::PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("fnp_{name}_{ts}.json"))
    }

    #[test]
    fn baseline_generator_writes_json() {
        let output_path = temp_file("baseline");
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");

        let baseline = generate_benchmark_baseline(&repo_root, &output_path)
            .expect("baseline generation should succeed");
        assert_eq!(baseline.schema_version, 1);
        assert!(!baseline.workloads.is_empty());
        assert!(!baseline.environment_fingerprint.trim().is_empty());
        assert!(
            baseline
                .reproducibility
                .command
                .contains("generate_benchmark_baseline")
        );
        assert!(!baseline.evidence_log_refs.is_empty());
        for workload in &baseline.workloads {
            assert!(workload.telemetry.bytes_processed_per_run > 0);
            assert!(workload.telemetry.bandwidth_mib_per_sec_p50 > 0.0);
            assert!(workload.telemetry.throughput_elements_per_sec_p50 > 0.0);
        }

        let raw = fs::read_to_string(&output_path).expect("baseline file readable");
        let parsed: BenchmarkBaseline = serde_json::from_str(&raw).expect("baseline json parse");
        assert_eq!(parsed.workloads.len(), baseline.workloads.len());
        assert_eq!(
            parsed.reproducibility.command,
            baseline.reproducibility.command
        );

        let _ = fs::remove_file(output_path);
    }
}
