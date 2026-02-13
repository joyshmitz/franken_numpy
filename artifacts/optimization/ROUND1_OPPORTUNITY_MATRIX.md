# ROUND1_OPPORTUNITY_MATRIX

## Profile-First Rule

Round 1 implemented core semantics and benchmark baseline capture. Optimization levers remain pre-scored and gated.

## Matrix

| Hotspot | Impact (1-5) | Confidence (1-5) | Effort (1-5) | Score | Decision |
|---|---:|---:|---:|---:|---|
| Ufunc broadcast index mapping (`fnp-ufunc::broadcasted_source_index`) | 5 | 4 | 2 | 10.0 | Eligible for one-lever optimization after larger differential corpus |
| Ufunc reduction index/ravel path (`fnp-ufunc::reduce_sum`) | 4 | 4 | 2 | 8.0 | Eligible after profile confirms hotspot |
| Broadcast shape merge loop (`fnp-ndarray::broadcast_shape`) | 4 | 4 | 2 | 8.0 | Eligible after parity expansion |
| Contiguous stride synthesis (`fnp-ndarray::contiguous_strides`) | 3 | 5 | 1 | 15.0 | Eligible after baseline capture |
| Dtype promotion dispatch (`fnp-dtype::promote`) | 2 | 5 | 1 | 10.0 | Keep table form; optimize later only if profiled hotspot |
| Runtime decision logging (`fnp-runtime::decide_and_record`) | 2 | 3 | 2 | 3.0 | Defer; not currently throughput-critical |

## Current Baseline Artifact

- `artifacts/baselines/ufunc_benchmark_baseline.json`

## One-Lever Policy

Only one optimization lever may be merged per commit after baseline/profile/proof artifacts exist.
