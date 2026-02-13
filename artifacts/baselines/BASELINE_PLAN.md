# BASELINE_PLAN

Planned baseline commands for first performance round:

```bash
hyperfine --warmup 3 --runs 10 'cargo test -p fnp-conformance -- --nocapture'
hyperfine --warmup 3 --runs 10 'cargo test -p fnp-ndarray broadcast_shape_matches_numpy_style -- --exact'
```

Planned profile commands:

```bash
cargo flamegraph -p fnp-ndarray --test '*'
strace -c cargo test -p fnp-conformance -- --nocapture
```
