use std::path::Path;

use fnp_conformance::{HarnessConfig, run_all_core_suites, run_smoke};

#[test]
fn smoke_report_is_stable() {
    let cfg = HarnessConfig::default_paths();
    let report = run_smoke(&cfg);
    assert_eq!(report.suite, "smoke");
    assert!(report.fixture_count >= 1);
    assert!(report.oracle_present);

    let fixture_path = cfg.fixture_root.join("smoke_case.json");
    assert!(Path::new(&fixture_path).exists());
}

#[test]
fn core_conformance_suites_pass() {
    let cfg = HarnessConfig::default_paths();
    let suites = run_all_core_suites(&cfg).expect("core suites should execute");

    for suite in suites {
        assert!(
            suite.all_passed(),
            "suite {} failed with {:?}",
            suite.suite,
            suite.failures
        );
    }
}
