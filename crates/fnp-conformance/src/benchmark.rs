#![forbid(unsafe_code)]

use fnp_dtype::DType;
use fnp_ufunc::{BinaryOp, UFuncArray};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkBaseline {
    pub schema_version: u8,
    pub generated_at_unix_ms: u128,
    pub git_commit: String,
    pub workloads: Vec<BenchmarkWorkload>,
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

fn time_workload<F>(name: &str, runs: usize, mut run_fn: F) -> Result<BenchmarkWorkload, String>
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

    Ok(BenchmarkWorkload {
        name: name.to_string(),
        runs,
        samples_ms,
        percentiles,
    })
}

pub fn generate_benchmark_baseline(
    repo_root: &Path,
    output_path: &Path,
) -> Result<BenchmarkBaseline, String> {
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

    let add_workload = time_workload("ufunc_add_broadcast_256x256_by_256", 20, || {
        let out = lhs_add
            .elementwise_binary(&rhs_add, BinaryOp::Add)
            .map_err(|err| format!("broadcast add failed: {err}"))?;
        std::hint::black_box(out.values()[0]);
        Ok(())
    })?;

    let reduce_axis1 = time_workload("reduce_sum_axis1_keepdims_false_256x256", 20, || {
        let out = reduce_in
            .reduce_sum(Some(1), false)
            .map_err(|err| format!("axis reduction failed: {err}"))?;
        std::hint::black_box(out.values()[0]);
        Ok(())
    })?;

    let reduce_all = time_workload("reduce_sum_all_keepdims_false_256x256", 20, || {
        let out = reduce_in
            .reduce_sum(None, false)
            .map_err(|err| format!("global reduction failed: {err}"))?;
        std::hint::black_box(out.values()[0]);
        Ok(())
    })?;

    let baseline = BenchmarkBaseline {
        schema_version: 1,
        generated_at_unix_ms: now_unix_ms(),
        git_commit: git_commit_short(repo_root),
        workloads: vec![add_workload, reduce_axis1, reduce_all],
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

        let raw = fs::read_to_string(&output_path).expect("baseline file readable");
        let parsed: BenchmarkBaseline = serde_json::from_str(&raw).expect("baseline json parse");
        assert_eq!(parsed.workloads.len(), baseline.workloads.len());

        let _ = fs::remove_file(output_path);
    }
}
