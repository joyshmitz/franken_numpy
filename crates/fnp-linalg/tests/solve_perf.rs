use fnp_linalg::cholesky_solve_multi;
use std::time::Instant;

#[test]
fn bench_cholesky_solve_multi_naive() {
    let n = 100;
    let m = 1000;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        l[i * n + i] = 1.0; // Identity matrix as Cholesky factor
    }
    let b = vec![1.0; n * m];

    let start = Instant::now();
    let _res = cholesky_solve_multi(&l, &b, n, m).unwrap();
    let duration = start.elapsed();

    println!("cholesky_solve_multi (n={n}, m={m}) took: {:?}", duration);
}
