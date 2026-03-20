use fnp_ufunc::UFuncArray;
use fnp_dtype::DType;

#[test]
fn test_isclose_symmetry_and_special_values() {
    let a = UFuncArray::scalar(1e10, DType::F64);
    let b = UFuncArray::scalar(1e10 + 1.0, DType::F64);
    
    // Asymmetric formula: |a - b| <= atol + rtol * |b|
    // With rtol=1e-9, atol=0:
    // |1e10 - (1e10 + 1)| = 1.0
    // atol + rtol * |1e10 + 1| = 0 + 1e-9 * 10000000001 = 10.000000001
    // 1.0 <= 10.000000001 -> TRUE
    
    // BUT if we swap them:
    // |(1e10 + 1) - 1e10| = 1.0
    // atol + rtol * |1e10| = 0 + 1e-9 * 1e10 = 10.0
    // 1.0 <= 10.0 -> TRUE (still true here, let's pick values where it flips)
    
    let rtol = 1e-11;
    // |1e10 - (1e10 + 1)| = 1.0
    // 1e-11 * 1e10 = 0.1
    // 1.0 <= 0.1 -> FALSE
    
    assert_eq!(a.isclose(&b, rtol, 0.0).unwrap().values()[0], 0.0, "Should be false");
    
    // NaN handling: NumPy isclose(NaN, NaN) is FALSE by default
    let nan = UFuncArray::scalar(f64::NAN, DType::F64);
    assert_eq!(nan.isclose(&nan, 1e-5, 1e-8).unwrap().values()[0], 0.0, "isclose(NaN, NaN) should be false");

    // Inf handling: NumPy isclose(Inf, Inf) is TRUE
    let inf = UFuncArray::scalar(f64::INFINITY, DType::F64);
    assert_eq!(inf.isclose(&inf, 1e-5, 1e-8).unwrap().values()[0], 1.0, "isclose(Inf, Inf) should be true");
}

#[test]
fn test_isclose_broadcasting() {
    let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
    let b = UFuncArray::scalar(1.0, DType::F64);
    // This should work via broadcasting but currently fails with "size mismatch"
    let res = a.isclose(&b, 1e-5, 1e-8);
    assert!(res.is_ok(), "isclose should support broadcasting: {:?}", res.err());
}
