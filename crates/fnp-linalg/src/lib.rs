#![forbid(unsafe_code)]

use core::fmt;

pub const LINALG_PACKET_ID: &str = "FNP-P2C-008";
pub const MAX_TOLERANCE_SEARCH_DEPTH: usize = 128;
pub const MAX_BACKEND_REVALIDATION_ATTEMPTS: usize = 64;
pub const MAX_BATCH_SHAPE_CHECKS: usize = 2_000_000;

pub const LINALG_PACKET_REASON_CODES: [&str; 10] = [
    "linalg_shape_contract_violation",
    "linalg_solver_singularity",
    "linalg_cholesky_contract_violation",
    "linalg_qr_mode_invalid",
    "linalg_svd_nonconvergence",
    "linalg_spectral_convergence_failed",
    "linalg_lstsq_tuple_contract_violation",
    "linalg_norm_det_rank_policy_violation",
    "linalg_backend_bridge_invalid",
    "linalg_policy_unknown_metadata",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinAlgRuntimeMode {
    Strict,
    Hardened,
}

impl LinAlgRuntimeMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hardened => "hardened",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrMode {
    Reduced,
    Complete,
    R,
    Raw,
}

impl QrMode {
    pub fn from_mode_token(mode: &str) -> Result<Self, LinAlgError> {
        match mode {
            "reduced" => Ok(Self::Reduced),
            "complete" => Ok(Self::Complete),
            "r" => Ok(Self::R),
            "raw" => Ok(Self::Raw),
            _ => Err(LinAlgError::QrModeInvalid),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorNormOrder {
    One,
    Two,
    Inf,
    NegInf,
}

impl VectorNormOrder {
    pub fn from_token(token: &str) -> Result<Self, LinAlgError> {
        match token.trim().to_ascii_lowercase().as_str() {
            "1" => Ok(Self::One),
            "2" => Ok(Self::Two),
            "inf" | "+inf" => Ok(Self::Inf),
            "-inf" => Ok(Self::NegInf),
            _ => Err(LinAlgError::NormDetRankPolicyViolation(
                "unsupported vector norm order token",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNormOrder {
    Fro,
    One,
    Inf,
    Two,
    NegTwo,
    Nuclear,
}

impl MatrixNormOrder {
    pub fn from_token(token: &str) -> Result<Self, LinAlgError> {
        match token.trim().to_ascii_lowercase().as_str() {
            "fro" | "f" => Ok(Self::Fro),
            "1" => Ok(Self::One),
            "inf" | "+inf" => Ok(Self::Inf),
            "2" => Ok(Self::Two),
            "-2" => Ok(Self::NegTwo),
            "nuc" => Ok(Self::Nuclear),
            _ => Err(LinAlgError::NormDetRankPolicyViolation(
                "unsupported matrix norm order token",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinAlgError {
    ShapeContractViolation(&'static str),
    SolverSingularity,
    CholeskyContractViolation(&'static str),
    QrModeInvalid,
    SvdNonConvergence,
    SpectralConvergenceFailed,
    LstsqTupleContractViolation(&'static str),
    NormDetRankPolicyViolation(&'static str),
    BackendBridgeInvalid(&'static str),
    PolicyUnknownMetadata(&'static str),
}

impl LinAlgError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::ShapeContractViolation(_) => "linalg_shape_contract_violation",
            Self::SolverSingularity => "linalg_solver_singularity",
            Self::CholeskyContractViolation(_) => "linalg_cholesky_contract_violation",
            Self::QrModeInvalid => "linalg_qr_mode_invalid",
            Self::SvdNonConvergence => "linalg_svd_nonconvergence",
            Self::SpectralConvergenceFailed => "linalg_spectral_convergence_failed",
            Self::LstsqTupleContractViolation(_) => "linalg_lstsq_tuple_contract_violation",
            Self::NormDetRankPolicyViolation(_) => "linalg_norm_det_rank_policy_violation",
            Self::BackendBridgeInvalid(_) => "linalg_backend_bridge_invalid",
            Self::PolicyUnknownMetadata(_) => "linalg_policy_unknown_metadata",
        }
    }
}

impl fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeContractViolation(msg) => write!(f, "{msg}"),
            Self::SolverSingularity => write!(f, "solve/inv rejected singular matrix"),
            Self::CholeskyContractViolation(msg) => write!(f, "{msg}"),
            Self::QrModeInvalid => write!(f, "qr mode is not one of reduced|complete|r|raw"),
            Self::SvdNonConvergence => write!(f, "svd did not converge"),
            Self::SpectralConvergenceFailed => write!(f, "spectral decomposition did not converge"),
            Self::LstsqTupleContractViolation(msg) => write!(f, "{msg}"),
            Self::NormDetRankPolicyViolation(msg) => write!(f, "{msg}"),
            Self::BackendBridgeInvalid(msg) => write!(f, "{msg}"),
            Self::PolicyUnknownMetadata(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LinAlgError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QrOutputShapes {
    pub q_shape: Option<Vec<usize>>,
    pub r_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvdOutputShapes {
    pub u_shape: Vec<usize>,
    pub s_shape: Vec<usize>,
    pub vh_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LstsqOutputShapes {
    pub x_shape: Vec<usize>,
    pub residuals_shape: Vec<usize>,
    pub rank_upper_bound: usize,
    pub singular_values_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qr2x2Result {
    pub q: Option<[[f64; 2]; 2]>,
    pub r: [[f64; 2]; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Svd2x2Result {
    pub u: [[f64; 2]; 2],
    pub singular_values: [f64; 2],
    pub vt: [[f64; 2]; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lstsq2x2Result {
    pub solution: [f64; 2],
    pub residual_sum_squares: f64,
    pub rank: usize,
    pub singular_values: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinAlgLogRecord {
    pub ts_utc: String,
    pub suite_id: String,
    pub test_id: String,
    pub packet_id: String,
    pub fixture_id: String,
    pub mode: LinAlgRuntimeMode,
    pub seed: u64,
    pub input_digest: String,
    pub output_digest: String,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub duration_ms: u64,
    pub outcome: String,
    pub reason_code: String,
}

impl LinAlgLogRecord {
    #[must_use]
    pub fn is_replay_complete(&self) -> bool {
        if self.ts_utc.trim().is_empty()
            || self.suite_id.trim().is_empty()
            || self.test_id.trim().is_empty()
            || self.packet_id.trim().is_empty()
            || self.fixture_id.trim().is_empty()
            || self.input_digest.trim().is_empty()
            || self.output_digest.trim().is_empty()
            || self.env_fingerprint.trim().is_empty()
            || self.reason_code.trim().is_empty()
        {
            return false;
        }

        if self.packet_id != LINALG_PACKET_ID {
            return false;
        }

        if self.outcome != "pass" && self.outcome != "fail" {
            return false;
        }

        if self.artifact_refs.is_empty()
            || self
                .artifact_refs
                .iter()
                .any(|artifact| artifact.trim().is_empty())
        {
            return false;
        }

        LINALG_PACKET_REASON_CODES
            .iter()
            .any(|code| *code == self.reason_code)
    }
}

pub fn validate_matrix_shape(shape: &[usize]) -> Result<(usize, usize), LinAlgError> {
    if shape.len() < 2 {
        return Err(LinAlgError::ShapeContractViolation(
            "linalg input must be at least 2D",
        ));
    }

    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    if rows == 0 || cols == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix rows/cols must be non-zero",
        ));
    }

    let batch_rank = shape.len() - 2;
    let batch_lanes = match batch_rank {
        0 => 1usize,
        1 => shape[0],
        2 => shape[0]
            .checked_mul(shape[1])
            .ok_or(LinAlgError::ShapeContractViolation(
                "batch lane multiplication overflowed",
            ))?,
        _ => shape[..batch_rank]
            .iter()
            .copied()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or(LinAlgError::ShapeContractViolation(
                "batch lane multiplication overflowed",
            ))?,
    };

    if batch_lanes > MAX_BATCH_SHAPE_CHECKS {
        return Err(LinAlgError::ShapeContractViolation(
            "batch lanes exceeded bounded validation budget",
        ));
    }

    Ok((rows, cols))
}

pub fn validate_square_matrix(shape: &[usize]) -> Result<usize, LinAlgError> {
    let (rows, cols) = validate_matrix_shape(shape)?;
    if rows != cols {
        return Err(LinAlgError::ShapeContractViolation(
            "square matrix required for solve/inv/cholesky",
        ));
    }
    Ok(rows)
}

pub fn solve_2x2(lhs: [[f64; 2]; 2], rhs: [f64; 2]) -> Result<[f64; 2], LinAlgError> {
    let det = lhs[0][0] * lhs[1][1] - lhs[0][1] * lhs[1][0];
    if det.abs() <= f64::EPSILON {
        return Err(LinAlgError::SolverSingularity);
    }

    let inv_det = 1.0 / det;
    let x0 = (rhs[0] * lhs[1][1] - lhs[0][1] * rhs[1]) * inv_det;
    let x1 = (lhs[0][0] * rhs[1] - rhs[0] * lhs[1][0]) * inv_det;
    Ok([x0, x1])
}

fn validate_finite_matrix_2x2(matrix: [[f64; 2]; 2]) -> Result<(), LinAlgError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for norm/det/rank/pinv operations",
        ));
    }
    Ok(())
}

pub fn det_2x2(matrix: [[f64; 2]; 2]) -> Result<f64, LinAlgError> {
    validate_finite_matrix_2x2(matrix)?;
    Ok(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
}

pub fn slogdet_2x2(matrix: [[f64; 2]; 2]) -> Result<(f64, f64), LinAlgError> {
    let det = det_2x2(matrix)?;
    if det > 0.0 {
        Ok((1.0, det.ln()))
    } else if det < 0.0 {
        Ok((-1.0, (-det).ln()))
    } else {
        Ok((0.0, f64::NEG_INFINITY))
    }
}

pub fn inv_2x2(matrix: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2], LinAlgError> {
    let det = det_2x2(matrix)?;
    if det.abs() <= f64::EPSILON {
        return Err(LinAlgError::SolverSingularity);
    }

    let inv_det = 1.0 / det;
    Ok([
        [matrix[1][1] * inv_det, -matrix[0][1] * inv_det],
        [-matrix[1][0] * inv_det, matrix[0][0] * inv_det],
    ])
}

fn singular_values_2x2(matrix: [[f64; 2]; 2]) -> Result<[f64; 2], LinAlgError> {
    validate_finite_matrix_2x2(matrix)?;
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];

    let trace = a.mul_add(a, b.mul_add(b, c.mul_add(c, d * d)));
    let det = a * d - b * c;
    let mut disc = trace.mul_add(trace, -4.0 * det * det);
    if disc < 0.0 && disc.abs() <= f64::EPSILON * 32.0 {
        disc = 0.0;
    }
    if disc < 0.0 || !disc.is_finite() {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "singular-value discriminant became invalid",
        ));
    }

    let sqrt_disc = disc.sqrt();
    let lambda_max = ((trace + sqrt_disc) * 0.5).max(0.0);
    let lambda_min = ((trace - sqrt_disc) * 0.5).max(0.0);
    Ok([lambda_max.sqrt(), lambda_min.sqrt()])
}

fn validate_finite_spectral_matrix_2x2(matrix: [[f64; 2]; 2]) -> Result<(), LinAlgError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    Ok(())
}

fn real_eigenvalues_2x2(matrix: [[f64; 2]; 2]) -> Result<[f64; 2], LinAlgError> {
    let trace = matrix[0][0] + matrix[1][1];
    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let mut disc = trace.mul_add(trace, -4.0 * det);
    if disc < 0.0 && disc.abs() <= f64::EPSILON * 32.0 {
        disc = 0.0;
    }
    if disc < 0.0 || !disc.is_finite() {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    let sqrt_disc = disc.sqrt();
    Ok([(trace + sqrt_disc) * 0.5, (trace - sqrt_disc) * 0.5])
}

pub fn matrix_rank_2x2(matrix: [[f64; 2]; 2], rcond: f64) -> Result<usize, LinAlgError> {
    validate_tolerance_policy(rcond, 0)?;
    let singular_values = singular_values_2x2(matrix)?;
    let sigma_max = singular_values[0];
    if sigma_max <= f64::EPSILON {
        return Ok(0);
    }

    let threshold = sigma_max * rcond;
    Ok(singular_values
        .iter()
        .filter(|&&sigma| sigma > threshold)
        .count())
}

pub fn pinv_2x2(matrix: [[f64; 2]; 2], rcond: f64) -> Result<[[f64; 2]; 2], LinAlgError> {
    validate_tolerance_policy(rcond, 0)?;
    validate_finite_matrix_2x2(matrix)?;

    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];

    // Right-singular vectors are eigenvectors of A^T A.
    let m00 = a.mul_add(a, c * c);
    let m01 = a.mul_add(b, c * d);
    let m11 = b.mul_add(b, d * d);
    let trace = m00 + m11;
    let det = m00 * m11 - m01 * m01;
    let mut disc = trace.mul_add(trace, -4.0 * det);
    if disc < 0.0 && disc.abs() <= f64::EPSILON * 32.0 {
        disc = 0.0;
    }
    if disc < 0.0 || !disc.is_finite() {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "pinv eigendecomposition discriminant became invalid",
        ));
    }

    let sqrt_disc = disc.sqrt();
    let lambda_1 = ((trace + sqrt_disc) * 0.5).max(0.0);
    let lambda_2 = ((trace - sqrt_disc) * 0.5).max(0.0);
    let sigma_1 = lambda_1.sqrt();
    let sigma_2 = lambda_2.sqrt();

    let mut v1 = if m01.abs() > f64::EPSILON {
        [m01, lambda_1 - m00]
    } else if m00 >= m11 {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let v1_norm = (v1[0].mul_add(v1[0], v1[1] * v1[1])).sqrt();
    if !v1_norm.is_finite() || v1_norm <= f64::EPSILON {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "pinv eigenvector normalization failed",
        ));
    }
    v1[0] /= v1_norm;
    v1[1] /= v1_norm;
    let v2 = [-v1[1], v1[0]];
    let sigmas = [sigma_1, sigma_2];
    let vectors = [v1, v2];

    let sigma_max = sigma_1;
    if sigma_max <= f64::EPSILON {
        return Ok([[0.0, 0.0], [0.0, 0.0]]);
    }
    let cutoff = sigma_max * rcond;

    let mut pinv = [[0.0_f64; 2]; 2];
    for idx in 0..2 {
        let sigma = sigmas[idx];
        if sigma <= cutoff {
            continue;
        }

        let v = vectors[idx];
        let av = [a.mul_add(v[0], b * v[1]), c.mul_add(v[0], d * v[1])];
        let u = [av[0] / sigma, av[1] / sigma];
        let inv_sigma = 1.0 / sigma;

        pinv[0][0] += inv_sigma * v[0] * u[0];
        pinv[0][1] += inv_sigma * v[0] * u[1];
        pinv[1][0] += inv_sigma * v[1] * u[0];
        pinv[1][1] += inv_sigma * v[1] * u[1];
    }

    Ok(pinv)
}

pub fn vector_norm(values: &[f64], ord: Option<VectorNormOrder>) -> Result<f64, LinAlgError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "vector norm requires finite entries",
        ));
    }

    let order = ord.unwrap_or(VectorNormOrder::Two);
    if values.is_empty() {
        if matches!(order, VectorNormOrder::NegInf) {
            return Err(LinAlgError::NormDetRankPolicyViolation(
                "negative infinity vector norm is undefined for empty inputs",
            ));
        }
        return Ok(0.0);
    }

    let abs_values = values.iter().map(|value| value.abs());
    let result = match order {
        VectorNormOrder::One => abs_values.sum(),
        VectorNormOrder::Two => abs_values.map(|value| value * value).sum::<f64>().sqrt(),
        VectorNormOrder::Inf => abs_values.fold(0.0, f64::max),
        VectorNormOrder::NegInf => abs_values.fold(f64::INFINITY, f64::min),
    };
    Ok(result)
}

pub fn matrix_norm_2x2(
    matrix: [[f64; 2]; 2],
    ord: Option<MatrixNormOrder>,
) -> Result<f64, LinAlgError> {
    validate_finite_matrix_2x2(matrix)?;

    let order = ord.unwrap_or(MatrixNormOrder::Fro);
    let result = match order {
        MatrixNormOrder::Fro => {
            let mut sum_sq = 0.0;
            for row in matrix {
                for value in row {
                    sum_sq += value * value;
                }
            }
            sum_sq.sqrt()
        }
        MatrixNormOrder::One => {
            let col0 = matrix[0][0].abs() + matrix[1][0].abs();
            let col1 = matrix[0][1].abs() + matrix[1][1].abs();
            col0.max(col1)
        }
        MatrixNormOrder::Inf => {
            let row0 = matrix[0][0].abs() + matrix[0][1].abs();
            let row1 = matrix[1][0].abs() + matrix[1][1].abs();
            row0.max(row1)
        }
        MatrixNormOrder::Two => singular_values_2x2(matrix)?[0],
        MatrixNormOrder::NegTwo => singular_values_2x2(matrix)?[1],
        MatrixNormOrder::Nuclear => {
            let singular_values = singular_values_2x2(matrix)?;
            singular_values[0] + singular_values[1]
        }
    };
    Ok(result)
}

pub fn eigvals_2x2(matrix: [[f64; 2]; 2], converged: bool) -> Result<[f64; 2], LinAlgError> {
    if !converged {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    validate_finite_spectral_matrix_2x2(matrix)?;
    real_eigenvalues_2x2(matrix)
}

pub fn eigh_2x2(
    matrix: [[f64; 2]; 2],
    uplo: &str,
    converged: bool,
) -> Result<([f64; 2], [[f64; 2]; 2]), LinAlgError> {
    validate_spectral_branch(uplo, converged)?;
    validate_finite_spectral_matrix_2x2(matrix)?;

    let symmetric = match uplo {
        "L" => [[matrix[0][0], matrix[1][0]], [matrix[1][0], matrix[1][1]]],
        "U" => [[matrix[0][0], matrix[0][1]], [matrix[0][1], matrix[1][1]]],
        _ => return Err(LinAlgError::SpectralConvergenceFailed),
    };

    let mut eigenvalues = real_eigenvalues_2x2(symmetric)?;
    if eigenvalues[0] > eigenvalues[1] {
        eigenvalues.swap(0, 1);
    }

    let mut eigenvectors = [[0.0_f64; 2]; 2];
    let a = symmetric[0][0];
    let b = symmetric[0][1];
    let d = symmetric[1][1];

    let lambda0 = eigenvalues[0];
    let mut v0 = if b.abs() > f64::EPSILON {
        [b, lambda0 - a]
    } else if (a - lambda0).abs() <= (d - lambda0).abs() {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let v0_norm = v0[0].hypot(v0[1]);
    if !v0_norm.is_finite() || v0_norm <= f64::EPSILON {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    v0[0] /= v0_norm;
    v0[1] /= v0_norm;

    let v1 = [-v0[1], v0[0]];
    eigenvectors[0][0] = v0[0];
    eigenvectors[1][0] = v0[1];
    eigenvectors[0][1] = v1[0];
    eigenvectors[1][1] = v1[1];

    Ok((eigenvalues, eigenvectors))
}

fn validate_cholesky_uplo(uplo: &str) -> Result<(), LinAlgError> {
    if uplo == "L" || uplo == "U" {
        Ok(())
    } else {
        Err(LinAlgError::CholeskyContractViolation(
            "cholesky uplo must be L or U",
        ))
    }
}

pub fn cholesky_2x2(matrix: [[f64; 2]; 2], uplo: &str) -> Result<[[f64; 2]; 2], LinAlgError> {
    validate_cholesky_uplo(uplo)?;

    let (a, b, d) = match uplo {
        "L" => (matrix[0][0], matrix[1][0], matrix[1][1]),
        "U" => (matrix[0][0], matrix[0][1], matrix[1][1]),
        _ => unreachable!(),
    };

    if !a.is_finite() || !b.is_finite() || !d.is_finite() {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky requires finite selected-triangle entries",
        ));
    }
    if a <= 0.0 {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky requires positive leading principal minor",
        ));
    }

    let l11 = a.sqrt();
    let l21 = b / l11;
    let schur = d - l21 * l21;
    if !schur.is_finite() || schur <= 0.0 {
        return Err(LinAlgError::CholeskyContractViolation(
            "matrix is not positive definite on selected triangle",
        ));
    }
    let l22 = schur.sqrt();

    let result = match uplo {
        "L" => [[l11, 0.0], [l21, l22]],
        "U" => [[l11, l21], [0.0, l22]],
        _ => unreachable!(),
    };
    Ok(result)
}

pub fn validate_cholesky_diagonal(diagonal: &[f64]) -> Result<(), LinAlgError> {
    if diagonal.is_empty() {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky diagonal cannot be empty",
        ));
    }
    if diagonal
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky requires strictly positive finite diagonal",
        ));
    }
    Ok(())
}

pub fn qr_output_shapes(shape: &[usize], mode: QrMode) -> Result<QrOutputShapes, LinAlgError> {
    let (m, n) = validate_matrix_shape(shape)?;
    let k = m.min(n);

    let output = match mode {
        QrMode::Reduced => QrOutputShapes {
            q_shape: Some(vec![m, k]),
            r_shape: vec![k, n],
        },
        QrMode::Complete => QrOutputShapes {
            q_shape: Some(vec![m, m]),
            r_shape: vec![m, n],
        },
        QrMode::R => QrOutputShapes {
            q_shape: None,
            r_shape: vec![k, n],
        },
        QrMode::Raw => QrOutputShapes {
            q_shape: Some(vec![n, m]),
            r_shape: vec![k],
        },
    };
    Ok(output)
}

pub fn qr_2x2(matrix: [[f64; 2]; 2], mode: QrMode) -> Result<Qr2x2Result, LinAlgError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LinAlgError::QrModeInvalid);
    }
    if mode == QrMode::Raw {
        return Err(LinAlgError::QrModeInvalid);
    }

    let c1 = [matrix[0][0], matrix[1][0]];
    let c2 = [matrix[0][1], matrix[1][1]];

    let mut q1 = [1.0_f64, 0.0_f64];
    let r11 = c1[0].hypot(c1[1]);
    if r11 > f64::EPSILON {
        q1 = [c1[0] / r11, c1[1] / r11];
    }

    let r12 = q1[0].mul_add(c2[0], q1[1] * c2[1]);
    let u2 = [c2[0] - r12 * q1[0], c2[1] - r12 * q1[1]];
    let mut q2 = [-q1[1], q1[0]];
    let mut r22 = u2[0].hypot(u2[1]);
    if r22 > f64::EPSILON {
        q2 = [u2[0] / r22, u2[1] / r22];
    } else {
        r22 = q2[0].mul_add(c2[0], q2[1] * c2[1]);
    }

    if r22 < 0.0 {
        q2[0] = -q2[0];
        q2[1] = -q2[1];
        r22 = -r22;
    }

    let q = [[q1[0], q2[0]], [q1[1], q2[1]]];
    let r = [[r11, r12], [0.0, r22]];

    let q_out = match mode {
        QrMode::Reduced | QrMode::Complete => Some(q),
        QrMode::R => None,
        QrMode::Raw => unreachable!(),
    };
    Ok(Qr2x2Result { q: q_out, r })
}

pub fn svd_2x2(matrix: [[f64; 2]; 2], converged: bool) -> Result<Svd2x2Result, LinAlgError> {
    if !converged {
        return Err(LinAlgError::SvdNonConvergence);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LinAlgError::SvdNonConvergence);
    }

    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];

    // Right-singular vectors come from eigendecomposition of A^T A.
    let m00 = a.mul_add(a, c * c);
    let m01 = a.mul_add(b, c * d);
    let m11 = b.mul_add(b, d * d);
    let trace = m00 + m11;
    let det = m00 * m11 - m01 * m01;
    let mut disc = trace.mul_add(trace, -4.0 * det);
    if disc < 0.0 && disc.abs() <= f64::EPSILON * 32.0 {
        disc = 0.0;
    }
    if disc < 0.0 || !disc.is_finite() {
        return Err(LinAlgError::SvdNonConvergence);
    }

    let sqrt_disc = disc.sqrt();
    let lambda_1 = ((trace + sqrt_disc) * 0.5).max(0.0);
    let lambda_2 = ((trace - sqrt_disc) * 0.5).max(0.0);
    let sigma_1 = lambda_1.sqrt();
    let sigma_2 = lambda_2.sqrt();
    if !sigma_1.is_finite() || !sigma_2.is_finite() {
        return Err(LinAlgError::SvdNonConvergence);
    }

    let mut v1 = if m01.abs() > f64::EPSILON {
        [m01, lambda_1 - m00]
    } else if m00 >= m11 {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let v1_norm = v1[0].hypot(v1[1]);
    if !v1_norm.is_finite() || v1_norm <= f64::EPSILON {
        return Err(LinAlgError::SvdNonConvergence);
    }
    v1[0] /= v1_norm;
    v1[1] /= v1_norm;
    let v2 = [-v1[1], v1[0]];
    let vectors = [v1, v2];

    let singular_values = [sigma_1, sigma_2];
    let mut u_cols = [[0.0_f64; 2]; 2];
    if sigma_1 <= f64::EPSILON {
        u_cols[0] = [1.0, 0.0];
        u_cols[1] = [0.0, 1.0];
    } else {
        for idx in 0..2 {
            let sigma = singular_values[idx];
            let v = vectors[idx];

            let mut u = if sigma > f64::EPSILON {
                [
                    a.mul_add(v[0], b * v[1]) / sigma,
                    c.mul_add(v[0], d * v[1]) / sigma,
                ]
            } else {
                [-u_cols[0][1], u_cols[0][0]]
            };

            if idx == 1 && sigma > f64::EPSILON {
                let proj = u_cols[0][0].mul_add(u[0], u_cols[0][1] * u[1]);
                u[0] -= proj * u_cols[0][0];
                u[1] -= proj * u_cols[0][1];
            }

            let norm = u[0].hypot(u[1]);
            if !norm.is_finite() || norm <= f64::EPSILON {
                if idx == 0 {
                    return Err(LinAlgError::SvdNonConvergence);
                }
                u = [-u_cols[0][1], u_cols[0][0]];
            } else {
                u[0] /= norm;
                u[1] /= norm;
            }
            u_cols[idx] = u;
        }
    }

    let u = [[u_cols[0][0], u_cols[1][0]], [u_cols[0][1], u_cols[1][1]]];
    let vt = [
        [vectors[0][0], vectors[0][1]],
        [vectors[1][0], vectors[1][1]],
    ];

    Ok(Svd2x2Result {
        u,
        singular_values,
        vt,
    })
}

pub fn svd_output_shapes(
    shape: &[usize],
    full_matrices: bool,
    converged: bool,
) -> Result<SvdOutputShapes, LinAlgError> {
    if !converged {
        return Err(LinAlgError::SvdNonConvergence);
    }

    let (m, n) = validate_matrix_shape(shape)?;
    let k = m.min(n);

    let (u_shape, vh_shape) = if full_matrices {
        (vec![m, m], vec![n, n])
    } else {
        (vec![m, k], vec![k, n])
    };

    Ok(SvdOutputShapes {
        u_shape,
        s_shape: vec![k],
        vh_shape,
    })
}

pub fn validate_spectral_branch(uplo: &str, converged: bool) -> Result<(), LinAlgError> {
    if uplo != "L" && uplo != "U" {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    if !converged {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    Ok(())
}

pub fn lstsq_output_shapes(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Result<LstsqOutputShapes, LinAlgError> {
    if lhs_shape.len() != 2 {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "lhs must be 2D for lstsq",
        ));
    }
    let m = lhs_shape[0];
    let n = lhs_shape[1];
    if m == 0 || n == 0 {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "lhs dimensions must be non-zero",
        ));
    }

    let rhs_cols = match rhs_shape {
        [rows] => {
            if *rows != m {
                return Err(LinAlgError::LstsqTupleContractViolation(
                    "rhs rows must equal lhs rows",
                ));
            }
            1usize
        }
        [rows, cols] => {
            if *rows != m || *cols == 0 {
                return Err(LinAlgError::LstsqTupleContractViolation(
                    "rhs shape must be (m,) or (m,k>0)",
                ));
            }
            *cols
        }
        _ => {
            return Err(LinAlgError::LstsqTupleContractViolation(
                "rhs must be 1D or 2D for lstsq",
            ));
        }
    };

    let x_shape = if rhs_shape.len() == 1 {
        vec![n]
    } else {
        vec![n, rhs_cols]
    };
    let residuals_shape = if m > n { vec![rhs_cols] } else { Vec::new() };

    Ok(LstsqOutputShapes {
        x_shape,
        residuals_shape,
        rank_upper_bound: m.min(n),
        singular_values_shape: vec![m.min(n)],
    })
}

pub fn lstsq_2x2(
    lhs: [[f64; 2]; 2],
    rhs: [f64; 2],
    rcond: f64,
) -> Result<Lstsq2x2Result, LinAlgError> {
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "rcond must be finite and >= 0 for lstsq_2x2",
        ));
    }
    if lhs.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "lhs entries must be finite for lstsq_2x2",
        ));
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "rhs entries must be finite for lstsq_2x2",
        ));
    }

    let pinv = pinv_2x2(lhs, rcond)
        .map_err(|_| LinAlgError::LstsqTupleContractViolation("lstsq_2x2 pinv route failed"))?;
    let solution = [
        pinv[0][0].mul_add(rhs[0], pinv[0][1] * rhs[1]),
        pinv[1][0].mul_add(rhs[0], pinv[1][1] * rhs[1]),
    ];

    let residual = [
        lhs[0][0].mul_add(solution[0], lhs[0][1] * solution[1]) - rhs[0],
        lhs[1][0].mul_add(solution[0], lhs[1][1] * solution[1]) - rhs[1],
    ];
    let residual_sum_squares = residual[0].mul_add(residual[0], residual[1] * residual[1]);

    let rank = matrix_rank_2x2(lhs, rcond).map_err(|_| {
        LinAlgError::LstsqTupleContractViolation("lstsq_2x2 rank evaluation failed")
    })?;
    let singular_values = singular_values_2x2(lhs).map_err(|_| {
        LinAlgError::LstsqTupleContractViolation("lstsq_2x2 singular-value evaluation failed")
    })?;

    Ok(Lstsq2x2Result {
        solution,
        residual_sum_squares,
        rank,
        singular_values,
    })
}

pub fn validate_tolerance_policy(rcond: f64, search_depth: usize) -> Result<(), LinAlgError> {
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "rcond must be finite and >= 0",
        ));
    }
    if search_depth > MAX_TOLERANCE_SEARCH_DEPTH {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "search depth exceeded tolerance budget",
        ));
    }
    Ok(())
}

pub fn validate_backend_bridge(
    backend_supported: bool,
    revalidation_attempts: usize,
) -> Result<(), LinAlgError> {
    if !backend_supported {
        return Err(LinAlgError::BackendBridgeInvalid(
            "backend bridge is unsupported",
        ));
    }
    if revalidation_attempts > MAX_BACKEND_REVALIDATION_ATTEMPTS {
        return Err(LinAlgError::BackendBridgeInvalid(
            "backend bridge revalidation budget exceeded",
        ));
    }
    Ok(())
}

pub fn validate_policy_metadata(mode: &str, class: &str) -> Result<(), LinAlgError> {
    let known_mode = mode == "strict" || mode == "hardened";
    let known_class = class == "known_compatible_low_risk"
        || class == "known_compatible_high_risk"
        || class == "known_incompatible_semantics"
        || class == "unknown_semantics";

    if !known_mode || !known_class {
        return Err(LinAlgError::PolicyUnknownMetadata(
            "unknown mode/class metadata rejected fail-closed",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        LINALG_PACKET_ID, LINALG_PACKET_REASON_CODES, LinAlgError, LinAlgLogRecord,
        LinAlgRuntimeMode, MAX_BACKEND_REVALIDATION_ATTEMPTS, MAX_BATCH_SHAPE_CHECKS,
        MAX_TOLERANCE_SEARCH_DEPTH, MatrixNormOrder, QrMode, VectorNormOrder, cholesky_2x2,
        det_2x2, eigh_2x2, eigvals_2x2, inv_2x2, lstsq_2x2, lstsq_output_shapes, matrix_norm_2x2,
        matrix_rank_2x2, pinv_2x2, qr_2x2, qr_output_shapes, slogdet_2x2, solve_2x2, svd_2x2,
        svd_output_shapes, validate_backend_bridge, validate_cholesky_diagonal,
        validate_matrix_shape, validate_policy_metadata, validate_spectral_branch,
        validate_square_matrix, validate_tolerance_policy, vector_norm,
    };

    fn packet008_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-008/contract_table.md".to_string(),
            "artifacts/phase2c/FNP-P2C-008/unit_property_evidence.json".to_string(),
        ]
    }

    fn approx_equal(lhs: f64, rhs: f64, tol: f64) -> bool {
        (lhs - rhs).abs() <= tol
    }

    fn legacy_validate_matrix_shape(shape: &[usize]) -> Result<(usize, usize), LinAlgError> {
        if shape.len() < 2 {
            return Err(LinAlgError::ShapeContractViolation(
                "linalg input must be at least 2D",
            ));
        }

        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        if rows == 0 || cols == 0 {
            return Err(LinAlgError::ShapeContractViolation(
                "matrix rows/cols must be non-zero",
            ));
        }

        let batch_lanes = shape[..shape.len() - 2]
            .iter()
            .copied()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or(LinAlgError::ShapeContractViolation(
                "batch lane multiplication overflowed",
            ))?;

        if batch_lanes > MAX_BATCH_SHAPE_CHECKS {
            return Err(LinAlgError::ShapeContractViolation(
                "batch lanes exceeded bounded validation budget",
            ));
        }

        Ok((rows, cols))
    }

    #[test]
    fn reason_code_registry_matches_packet_contract() {
        assert_eq!(
            LINALG_PACKET_REASON_CODES,
            [
                "linalg_shape_contract_violation",
                "linalg_solver_singularity",
                "linalg_cholesky_contract_violation",
                "linalg_qr_mode_invalid",
                "linalg_svd_nonconvergence",
                "linalg_spectral_convergence_failed",
                "linalg_lstsq_tuple_contract_violation",
                "linalg_norm_det_rank_policy_violation",
                "linalg_backend_bridge_invalid",
                "linalg_policy_unknown_metadata",
            ]
        );
    }

    #[test]
    fn matrix_shape_accepts_batched_square_shapes() {
        assert_eq!(
            validate_matrix_shape(&[8, 16, 3, 3]).expect("batched matrix"),
            (3, 3)
        );
        assert_eq!(
            validate_square_matrix(&[4, 4]).expect("square matrix"),
            4usize
        );
    }

    #[test]
    fn matrix_shape_rejects_invalid_rank_or_budget() {
        let err = validate_matrix_shape(&[4]).expect_err("rank<2 should fail");
        assert_eq!(err.reason_code(), "linalg_shape_contract_violation");

        let huge_batch = [MAX_TOLERANCE_SEARCH_DEPTH * 20_000usize, 2usize, 2usize];
        let err = validate_matrix_shape(&huge_batch).expect_err("batch budget should fail");
        assert_eq!(err.reason_code(), "linalg_shape_contract_violation");
    }

    #[test]
    fn matrix_shape_fast_paths_are_isomorphic_with_legacy_path() {
        let fixtures: [&[usize]; 13] = [
            &[2, 2],
            &[4, 4],
            &[1, 2, 2],
            &[8, 2, 2],
            &[7, 9, 3, 3],
            &[3, 5, 7, 11, 2, 2],
            &[4],
            &[0, 2],
            &[2, 0],
            &[MAX_BATCH_SHAPE_CHECKS + 1, 2, 2],
            &[usize::MAX, usize::MAX, 2, 2],
            &[usize::MAX, 1, 2, 2],
            &[1, usize::MAX, 2, 2],
        ];

        for shape in fixtures {
            let baseline = legacy_validate_matrix_shape(shape);
            let optimized = validate_matrix_shape(shape);
            assert_eq!(optimized, baseline, "shape={shape:?}");
        }
    }

    #[test]
    fn square_validation_rejects_non_square() {
        let err = validate_square_matrix(&[2, 3]).expect_err("non-square should fail");
        assert_eq!(err.reason_code(), "linalg_shape_contract_violation");
    }

    #[test]
    fn solve_2x2_reconstructs_rhs_across_seed_grid() {
        for seed in 1_u32..=256_u32 {
            let alpha = f64::from((seed % 19) + 2);
            let beta = f64::from((seed % 11) + 3);
            let matrix = [[alpha + 5.0, 1.0], [1.0, beta + 4.0]];
            let expected = [f64::from(seed) / 7.0, f64::from(seed) / 11.0];
            let rhs = [
                matrix[0][0] * expected[0] + matrix[0][1] * expected[1],
                matrix[1][0] * expected[0] + matrix[1][1] * expected[1],
            ];

            let solved = solve_2x2(matrix, rhs).expect("non-singular solve");
            assert!(
                approx_equal(solved[0], expected[0], 1e-10),
                "fixture_id=UP-008-solve-inv-contract seed={seed} lhs={} rhs={}",
                solved[0],
                expected[0]
            );
            assert!(
                approx_equal(solved[1], expected[1], 1e-10),
                "fixture_id=UP-008-solve-inv-contract seed={seed} lhs={} rhs={}",
                solved[1],
                expected[1]
            );
        }
    }

    #[test]
    fn solve_2x2_rejects_singular_system() {
        let err = solve_2x2([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
            .expect_err("singular matrix should fail");
        assert_eq!(err.reason_code(), "linalg_solver_singularity");
    }

    #[test]
    fn det_and_slogdet_are_deterministic() {
        let matrix = [[4.0, 7.0], [2.0, 6.0]];
        let det = det_2x2(matrix).expect("det");
        assert!(approx_equal(det, 10.0, 1e-12));

        let (sign, log_abs_det) = slogdet_2x2(matrix).expect("slogdet");
        assert!(approx_equal(sign, 1.0, 1e-12));
        assert!(approx_equal(log_abs_det, 10.0_f64.ln(), 1e-12));

        let singular = [[1.0, 2.0], [2.0, 4.0]];
        let (sign, log_abs_det) = slogdet_2x2(singular).expect("singular slogdet");
        assert_eq!(sign, 0.0);
        assert_eq!(log_abs_det, f64::NEG_INFINITY);
    }

    #[test]
    fn inv_2x2_matches_identity_reconstruction() {
        let matrix = [[4.0, 7.0], [2.0, 6.0]];
        let inv = inv_2x2(matrix).expect("inverse");
        let m00 = matrix[0][0].mul_add(inv[0][0], matrix[0][1] * inv[1][0]);
        let m01 = matrix[0][0].mul_add(inv[0][1], matrix[0][1] * inv[1][1]);
        let m10 = matrix[1][0].mul_add(inv[0][0], matrix[1][1] * inv[1][0]);
        let m11 = matrix[1][0].mul_add(inv[0][1], matrix[1][1] * inv[1][1]);
        assert!(approx_equal(m00, 1.0, 1e-12));
        assert!(approx_equal(m01, 0.0, 1e-12));
        assert!(approx_equal(m10, 0.0, 1e-12));
        assert!(approx_equal(m11, 1.0, 1e-12));

        let err = inv_2x2([[1.0, 2.0], [2.0, 4.0]]).expect_err("singular inverse");
        assert_eq!(err.reason_code(), "linalg_solver_singularity");
    }

    #[test]
    fn matrix_rank_2x2_detects_rank_profiles() {
        let full_rank = matrix_rank_2x2([[3.0, 1.0], [2.0, 4.0]], 1e-12).expect("rank");
        assert_eq!(full_rank, 2);

        let rank_one = matrix_rank_2x2([[1.0, 2.0], [2.0, 4.0]], 1e-12).expect("rank");
        assert_eq!(rank_one, 1);

        let rank_zero = matrix_rank_2x2([[0.0, 0.0], [0.0, 0.0]], 1e-12).expect("rank");
        assert_eq!(rank_zero, 0);
    }

    #[test]
    fn pinv_2x2_full_rank_and_rank_deficient_paths() {
        let matrix = [[4.0, 7.0], [2.0, 6.0]];
        let pinv = pinv_2x2(matrix, 1e-12).expect("pinv");
        let inv = inv_2x2(matrix).expect("inv");
        for row in 0..2 {
            for col in 0..2 {
                assert!(approx_equal(pinv[row][col], inv[row][col], 1e-10));
            }
        }

        let rank_def = [[1.0, 2.0], [2.0, 4.0]];
        let pinv_rank_def = pinv_2x2(rank_def, 1e-12).expect("rank-def pinv");
        let expected = [[1.0 / 25.0, 2.0 / 25.0], [2.0 / 25.0, 4.0 / 25.0]];
        for row in 0..2 {
            for col in 0..2 {
                assert!(approx_equal(
                    pinv_rank_def[row][col],
                    expected[row][col],
                    1e-10
                ));
            }
        }
    }

    #[test]
    fn norm_det_rank_pinv_reject_non_finite_inputs() {
        let err = det_2x2([[f64::NAN, 1.0], [2.0, 3.0]]).expect_err("nan matrix");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");

        let err = pinv_2x2([[f64::INFINITY, 0.0], [0.0, 1.0]], 1e-12).expect_err("inf matrix");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn norm_order_token_parsers_are_fail_closed() {
        assert_eq!(
            VectorNormOrder::from_token("1").expect("vector one"),
            VectorNormOrder::One
        );
        assert_eq!(
            VectorNormOrder::from_token("-inf").expect("vector neginf"),
            VectorNormOrder::NegInf
        );
        assert_eq!(
            MatrixNormOrder::from_token("fro").expect("matrix fro"),
            MatrixNormOrder::Fro
        );
        assert_eq!(
            MatrixNormOrder::from_token("nuc").expect("matrix nuc"),
            MatrixNormOrder::Nuclear
        );

        let err = VectorNormOrder::from_token("hostile").expect_err("vector token should fail");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
        let err = MatrixNormOrder::from_token("hostile").expect_err("matrix token should fail");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn vector_norm_orders_match_first_wave_contracts() {
        let values = [3.0, -4.0];
        assert!(approx_equal(
            vector_norm(&values, None).expect("default l2"),
            5.0,
            1e-12
        ));
        assert!(approx_equal(
            vector_norm(&values, Some(VectorNormOrder::One)).expect("l1"),
            7.0,
            1e-12
        ));
        assert!(approx_equal(
            vector_norm(&values, Some(VectorNormOrder::Inf)).expect("inf"),
            4.0,
            1e-12
        ));
        assert!(approx_equal(
            vector_norm(&values, Some(VectorNormOrder::NegInf)).expect("-inf"),
            3.0,
            1e-12
        ));

        assert!(approx_equal(
            vector_norm(&[], None).expect("empty default"),
            0.0,
            1e-12
        ));
        let err = vector_norm(&[], Some(VectorNormOrder::NegInf)).expect_err("empty -inf");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn matrix_norm_orders_match_first_wave_contracts() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]];
        assert!(approx_equal(
            matrix_norm_2x2(matrix, None).expect("default fro"),
            5.477225575051661,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::One)).expect("one"),
            6.0,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::Inf)).expect("inf"),
            7.0,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::Two)).expect("two"),
            5.464985704219043,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::NegTwo)).expect("-two"),
            0.36596619062625746,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::Nuclear)).expect("nuclear"),
            5.8309518948453,
            1e-12
        ));
    }

    #[test]
    fn norm_paths_reject_non_finite_inputs() {
        let err = vector_norm(&[f64::NAN, 1.0], None).expect_err("vector nan");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");

        let err = matrix_norm_2x2([[f64::INFINITY, 0.0], [0.0, 1.0]], None).expect_err("inf");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn eigvals_2x2_preserves_trace_and_determinant_relations() {
        let matrix = [[4.0, 2.0], [1.0, 3.0]];
        let eigvals = eigvals_2x2(matrix, true).expect("eigvals");
        let trace = eigvals[0] + eigvals[1];
        let det = eigvals[0] * eigvals[1];
        assert!(approx_equal(trace, 7.0, 1e-12));
        assert!(approx_equal(det, 10.0, 1e-12));
    }

    #[test]
    fn eigh_2x2_returns_orthonormal_eigenvectors() {
        let matrix = [[2.0, 1.0], [1.0, 2.0]];
        let (eigvals, eigvecs) = eigh_2x2(matrix, "L", true).expect("eigh");
        assert!(approx_equal(eigvals[0], 1.0, 1e-12));
        assert!(approx_equal(eigvals[1], 3.0, 1e-12));

        let dot = eigvecs[0][0] * eigvecs[0][1] + eigvecs[1][0] * eigvecs[1][1];
        assert!(approx_equal(dot, 0.0, 1e-12));
        let n0 = eigvecs[0][0].hypot(eigvecs[1][0]);
        let n1 = eigvecs[0][1].hypot(eigvecs[1][1]);
        assert!(approx_equal(n0, 1.0, 1e-12));
        assert!(approx_equal(n1, 1.0, 1e-12));

        for col in 0..2 {
            let lambda = eigvals[col];
            let v = [eigvecs[0][col], eigvecs[1][col]];
            let av = [
                matrix[0][0].mul_add(v[0], matrix[0][1] * v[1]),
                matrix[1][0].mul_add(v[0], matrix[1][1] * v[1]),
            ];
            assert!(approx_equal(av[0], lambda * v[0], 1e-10));
            assert!(approx_equal(av[1], lambda * v[1], 1e-10));
        }
    }

    #[test]
    fn eigh_2x2_respects_uplo_branch_choice() {
        let matrix = [[2.0, 100.0], [1.0, 2.0]];
        let (eigvals_l, _) = eigh_2x2(matrix, "L", true).expect("L");
        let (eigvals_u, _) = eigh_2x2(matrix, "U", true).expect("U");

        assert!(approx_equal(eigvals_l[0], 1.0, 1e-12));
        assert!(approx_equal(eigvals_l[1], 3.0, 1e-12));
        assert!(approx_equal(eigvals_u[0], -98.0, 1e-10));
        assert!(approx_equal(eigvals_u[1], 102.0, 1e-10));
    }

    #[test]
    fn spectral_kernels_fail_closed_for_invalid_inputs() {
        let err = eigvals_2x2([[1.0, 2.0], [3.0, 4.0]], false).expect_err("non-converged");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");

        let err = eigvals_2x2([[f64::NAN, 0.0], [0.0, 1.0]], true).expect_err("nan matrix");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");

        let err = eigvals_2x2([[0.0, -1.0], [1.0, 0.0]], true).expect_err("complex spectrum");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");

        let err = eigh_2x2([[1.0, 0.0], [0.0, 1.0]], "X", true).expect_err("invalid uplo");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");
    }

    #[test]
    fn cholesky_2x2_lower_and_upper_reconstruct_matrix() {
        let matrix = [[4.0, 1.0], [1.0, 3.0]];

        let lower = cholesky_2x2(matrix, "L").expect("lower");
        let ll_t = [
            [
                lower[0][0].mul_add(lower[0][0], lower[0][1] * lower[0][1]),
                lower[0][0].mul_add(lower[1][0], lower[0][1] * lower[1][1]),
            ],
            [
                lower[1][0].mul_add(lower[0][0], lower[1][1] * lower[0][1]),
                lower[1][0].mul_add(lower[1][0], lower[1][1] * lower[1][1]),
            ],
        ];
        assert!(approx_equal(ll_t[0][0], 4.0, 1e-12));
        assert!(approx_equal(ll_t[0][1], 1.0, 1e-12));
        assert!(approx_equal(ll_t[1][0], 1.0, 1e-12));
        assert!(approx_equal(ll_t[1][1], 3.0, 1e-12));

        let upper = cholesky_2x2(matrix, "U").expect("upper");
        let u_tu = [
            [
                upper[0][0].mul_add(upper[0][0], upper[1][0] * upper[1][0]),
                upper[0][0].mul_add(upper[0][1], upper[1][0] * upper[1][1]),
            ],
            [
                upper[0][1].mul_add(upper[0][0], upper[1][1] * upper[1][0]),
                upper[0][1].mul_add(upper[0][1], upper[1][1] * upper[1][1]),
            ],
        ];
        assert!(approx_equal(u_tu[0][0], 4.0, 1e-12));
        assert!(approx_equal(u_tu[0][1], 1.0, 1e-12));
        assert!(approx_equal(u_tu[1][0], 1.0, 1e-12));
        assert!(approx_equal(u_tu[1][1], 3.0, 1e-12));
    }

    #[test]
    fn cholesky_2x2_uses_selected_triangle_only() {
        let lower_only = [[4.0, 999.0], [1.0, 3.0]];
        let lower = cholesky_2x2(lower_only, "L").expect("lower");
        assert!(approx_equal(lower[0][0], 2.0, 1e-12));
        assert!(approx_equal(lower[1][0], 0.5, 1e-12));

        let upper_only = [[4.0, 1.0], [999.0, 3.0]];
        let upper = cholesky_2x2(upper_only, "U").expect("upper");
        assert!(approx_equal(upper[0][0], 2.0, 1e-12));
        assert!(approx_equal(upper[0][1], 0.5, 1e-12));
    }

    #[test]
    fn cholesky_2x2_fail_closed_for_invalid_inputs() {
        let err = cholesky_2x2([[1.0, 2.0], [2.0, 1.0]], "L").expect_err("non-pd");
        assert_eq!(err.reason_code(), "linalg_cholesky_contract_violation");

        let err = cholesky_2x2([[4.0, 1.0], [1.0, 3.0]], "X").expect_err("bad uplo");
        assert_eq!(err.reason_code(), "linalg_cholesky_contract_violation");

        let err = cholesky_2x2([[f64::NAN, 0.0], [0.0, 1.0]], "U").expect_err("non-finite");
        assert_eq!(err.reason_code(), "linalg_cholesky_contract_violation");
    }

    #[test]
    fn cholesky_diagonal_contract_is_enforced() {
        validate_cholesky_diagonal(&[4.0, 3.0, 2.0]).expect("pd diagonal should pass");
        let err = validate_cholesky_diagonal(&[3.0, 0.0]).expect_err("zero diagonal should fail");
        assert_eq!(err.reason_code(), "linalg_cholesky_contract_violation");
    }

    #[test]
    fn qr_mode_shapes_are_deterministic() {
        let reduced = qr_output_shapes(&[5, 3], QrMode::Reduced).expect("reduced");
        assert_eq!(reduced.q_shape, Some(vec![5, 3]));
        assert_eq!(reduced.r_shape, vec![3, 3]);

        let complete = qr_output_shapes(&[5, 3], QrMode::Complete).expect("complete");
        assert_eq!(complete.q_shape, Some(vec![5, 5]));
        assert_eq!(complete.r_shape, vec![5, 3]);

        let just_r = qr_output_shapes(&[5, 3], QrMode::R).expect("r mode");
        assert_eq!(just_r.q_shape, None);
        assert_eq!(just_r.r_shape, vec![3, 3]);

        let err = QrMode::from_mode_token("hostile_mode").expect_err("mode should fail");
        assert_eq!(err.reason_code(), "linalg_qr_mode_invalid");
    }

    #[test]
    fn qr_2x2_reduced_reconstructs_input_and_is_orthonormal() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]];
        let out = qr_2x2(matrix, QrMode::Reduced).expect("qr reduced");
        let q = out.q.expect("reduced has q");
        let r = out.r;

        let qtq = [
            [
                q[0][0].mul_add(q[0][0], q[1][0] * q[1][0]),
                q[0][0].mul_add(q[0][1], q[1][0] * q[1][1]),
            ],
            [
                q[0][1].mul_add(q[0][0], q[1][1] * q[1][0]),
                q[0][1].mul_add(q[0][1], q[1][1] * q[1][1]),
            ],
        ];
        assert!(approx_equal(qtq[0][0], 1.0, 1e-12));
        assert!(approx_equal(qtq[0][1], 0.0, 1e-12));
        assert!(approx_equal(qtq[1][0], 0.0, 1e-12));
        assert!(approx_equal(qtq[1][1], 1.0, 1e-12));

        let qr = [
            [
                q[0][0].mul_add(r[0][0], q[0][1] * r[1][0]),
                q[0][0].mul_add(r[0][1], q[0][1] * r[1][1]),
            ],
            [
                q[1][0].mul_add(r[0][0], q[1][1] * r[1][0]),
                q[1][0].mul_add(r[0][1], q[1][1] * r[1][1]),
            ],
        ];
        assert!(approx_equal(qr[0][0], matrix[0][0], 1e-10));
        assert!(approx_equal(qr[0][1], matrix[0][1], 1e-10));
        assert!(approx_equal(qr[1][0], matrix[1][0], 1e-10));
        assert!(approx_equal(qr[1][1], matrix[1][1], 1e-10));
    }

    #[test]
    fn qr_2x2_r_mode_and_complete_mode_contracts() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]];
        let reduced = qr_2x2(matrix, QrMode::Reduced).expect("reduced");
        let complete = qr_2x2(matrix, QrMode::Complete).expect("complete");
        let r_only = qr_2x2(matrix, QrMode::R).expect("r");

        assert_eq!(complete.q, reduced.q);
        assert_eq!(complete.r, reduced.r);
        assert!(r_only.q.is_none());
        assert_eq!(r_only.r, reduced.r);
        assert!(approx_equal(r_only.r[1][0], 0.0, 1e-12));
    }

    #[test]
    fn qr_2x2_handles_rank_deficient_input() {
        let matrix = [[1.0, 2.0], [2.0, 4.0]];
        let out = qr_2x2(matrix, QrMode::Reduced).expect("rank-def qr");
        assert!(approx_equal(out.r[1][1], 0.0, 1e-10));

        let q = out.q.expect("q");
        let qr = [
            [
                q[0][0].mul_add(out.r[0][0], q[0][1] * out.r[1][0]),
                q[0][0].mul_add(out.r[0][1], q[0][1] * out.r[1][1]),
            ],
            [
                q[1][0].mul_add(out.r[0][0], q[1][1] * out.r[1][0]),
                q[1][0].mul_add(out.r[0][1], q[1][1] * out.r[1][1]),
            ],
        ];
        assert!(approx_equal(qr[0][0], matrix[0][0], 1e-10));
        assert!(approx_equal(qr[0][1], matrix[0][1], 1e-10));
        assert!(approx_equal(qr[1][0], matrix[1][0], 1e-10));
        assert!(approx_equal(qr[1][1], matrix[1][1], 1e-10));
    }

    #[test]
    fn qr_2x2_fail_closed_for_non_finite_and_raw_mode() {
        let err = qr_2x2([[f64::NAN, 0.0], [0.0, 1.0]], QrMode::Reduced).expect_err("nan");
        assert_eq!(err.reason_code(), "linalg_qr_mode_invalid");

        let err = qr_2x2([[1.0, 0.0], [0.0, 1.0]], QrMode::Raw).expect_err("raw");
        assert_eq!(err.reason_code(), "linalg_qr_mode_invalid");
    }

    #[test]
    fn svd_2x2_reconstructs_and_orders_singular_values() {
        let matrix = [[3.0, 1.0], [1.0, 3.0]];
        let out = svd_2x2(matrix, true).expect("svd");
        assert!(out.singular_values[0] >= out.singular_values[1]);
        assert!(out.singular_values[1] >= 0.0);

        let u = out.u;
        let vt = out.vt;
        let s = out.singular_values;

        let utu = [
            [
                u[0][0].mul_add(u[0][0], u[1][0] * u[1][0]),
                u[0][0].mul_add(u[0][1], u[1][0] * u[1][1]),
            ],
            [
                u[0][1].mul_add(u[0][0], u[1][1] * u[1][0]),
                u[0][1].mul_add(u[0][1], u[1][1] * u[1][1]),
            ],
        ];
        assert!(approx_equal(utu[0][0], 1.0, 1e-10));
        assert!(approx_equal(utu[0][1], 0.0, 1e-10));
        assert!(approx_equal(utu[1][0], 0.0, 1e-10));
        assert!(approx_equal(utu[1][1], 1.0, 1e-10));

        let vvt = [
            [
                vt[0][0].mul_add(vt[0][0], vt[0][1] * vt[0][1]),
                vt[0][0].mul_add(vt[1][0], vt[0][1] * vt[1][1]),
            ],
            [
                vt[1][0].mul_add(vt[0][0], vt[1][1] * vt[0][1]),
                vt[1][0].mul_add(vt[1][0], vt[1][1] * vt[1][1]),
            ],
        ];
        assert!(approx_equal(vvt[0][0], 1.0, 1e-10));
        assert!(approx_equal(vvt[0][1], 0.0, 1e-10));
        assert!(approx_equal(vvt[1][0], 0.0, 1e-10));
        assert!(approx_equal(vvt[1][1], 1.0, 1e-10));

        let us = [
            [u[0][0] * s[0], u[0][1] * s[1]],
            [u[1][0] * s[0], u[1][1] * s[1]],
        ];
        let recon = [
            [
                us[0][0].mul_add(vt[0][0], us[0][1] * vt[1][0]),
                us[0][0].mul_add(vt[0][1], us[0][1] * vt[1][1]),
            ],
            [
                us[1][0].mul_add(vt[0][0], us[1][1] * vt[1][0]),
                us[1][0].mul_add(vt[0][1], us[1][1] * vt[1][1]),
            ],
        ];
        assert!(approx_equal(recon[0][0], matrix[0][0], 1e-10));
        assert!(approx_equal(recon[0][1], matrix[0][1], 1e-10));
        assert!(approx_equal(recon[1][0], matrix[1][0], 1e-10));
        assert!(approx_equal(recon[1][1], matrix[1][1], 1e-10));
    }

    #[test]
    fn svd_2x2_handles_rank_deficient_and_fail_closed_paths() {
        let rank_def = [[1.0, 2.0], [2.0, 4.0]];
        let out = svd_2x2(rank_def, true).expect("rank-def svd");
        assert!(approx_equal(out.singular_values[1], 0.0, 1e-10));

        let err = svd_2x2([[f64::NAN, 0.0], [0.0, 1.0]], true).expect_err("nan");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");

        let err = svd_2x2([[1.0, 0.0], [0.0, 1.0]], false).expect_err("non-converged");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");
    }

    #[test]
    fn svd_shapes_match_full_and_reduced_contracts() {
        let reduced = svd_output_shapes(&[6, 4], false, true).expect("reduced svd");
        assert_eq!(reduced.u_shape, vec![6, 4]);
        assert_eq!(reduced.s_shape, vec![4]);
        assert_eq!(reduced.vh_shape, vec![4, 4]);

        let full = svd_output_shapes(&[6, 4], true, true).expect("full svd");
        assert_eq!(full.u_shape, vec![6, 6]);
        assert_eq!(full.s_shape, vec![4]);
        assert_eq!(full.vh_shape, vec![4, 4]);

        let err = svd_output_shapes(&[6, 4], false, false).expect_err("non-convergence");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");
    }

    #[test]
    fn spectral_branch_is_fail_closed() {
        validate_spectral_branch("L", true).expect("L branch should pass");
        validate_spectral_branch("U", true).expect("U branch should pass");
        let err = validate_spectral_branch("X", true).expect_err("unknown branch");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");
    }

    #[test]
    fn lstsq_tuple_shapes_cover_vector_and_matrix_rhs() {
        let vector_rhs = lstsq_output_shapes(&[5, 3], &[5]).expect("vector rhs");
        assert_eq!(vector_rhs.x_shape, vec![3]);
        assert_eq!(vector_rhs.residuals_shape, vec![1]);
        assert_eq!(vector_rhs.rank_upper_bound, 3);
        assert_eq!(vector_rhs.singular_values_shape, vec![3]);

        let matrix_rhs = lstsq_output_shapes(&[5, 3], &[5, 2]).expect("matrix rhs");
        assert_eq!(matrix_rhs.x_shape, vec![3, 2]);
        assert_eq!(matrix_rhs.residuals_shape, vec![2]);

        let err = lstsq_output_shapes(&[5, 3], &[4, 2]).expect_err("mismatch rows");
        assert_eq!(err.reason_code(), "linalg_lstsq_tuple_contract_violation");
    }

    #[test]
    fn lstsq_2x2_runtime_outputs_match_contract() {
        let lhs = [[3.0, 1.0], [1.0, 2.0]];
        let rhs = [5.0, 5.0];
        let out = lstsq_2x2(lhs, rhs, 1e-12).expect("lstsq runtime");
        assert!(approx_equal(out.solution[0], 1.0, 1e-12));
        assert!(approx_equal(out.solution[1], 2.0, 1e-12));
        assert!(approx_equal(out.residual_sum_squares, 0.0, 1e-12));
        assert_eq!(out.rank, 2);
        assert!(out.singular_values[0] >= out.singular_values[1]);

        let rank_def = [[1.0, 2.0], [2.0, 4.0]];
        let rank_def_rhs = [3.0, 6.0];
        let rank_def_out = lstsq_2x2(rank_def, rank_def_rhs, 1e-12).expect("rank-def");
        assert!(approx_equal(rank_def_out.solution[0], 0.6, 1e-10));
        assert!(approx_equal(rank_def_out.solution[1], 1.2, 1e-10));
        assert!(approx_equal(rank_def_out.residual_sum_squares, 0.0, 1e-10));
        assert_eq!(rank_def_out.rank, 1);
    }

    #[test]
    fn lstsq_2x2_reports_residual_for_inconsistent_rhs() {
        let lhs = [[1.0, 2.0], [2.0, 4.0]];
        let rhs = [1.0, 0.0];
        let out = lstsq_2x2(lhs, rhs, 1e-12).expect("inconsistent");
        assert!(out.residual_sum_squares > 0.1);
        assert_eq!(out.rank, 1);
    }

    #[test]
    fn lstsq_2x2_fail_closed_policy_checks() {
        let err = lstsq_2x2([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0], -1.0).expect_err("rcond");
        assert_eq!(err.reason_code(), "linalg_lstsq_tuple_contract_violation");

        let err =
            lstsq_2x2([[f64::INFINITY, 0.0], [0.0, 1.0]], [1.0, 2.0], 1e-12).expect_err("lhs");
        assert_eq!(err.reason_code(), "linalg_lstsq_tuple_contract_violation");

        let err = lstsq_2x2([[1.0, 0.0], [0.0, 1.0]], [f64::NAN, 2.0], 1e-12).expect_err("rhs");
        assert_eq!(err.reason_code(), "linalg_lstsq_tuple_contract_violation");
    }

    #[test]
    fn tolerance_policy_enforces_bounds() {
        for depth in [0usize, 1, 8, 32, 64, MAX_TOLERANCE_SEARCH_DEPTH] {
            validate_tolerance_policy(1e-6, depth).expect("depth within budget");
        }
        let err = validate_tolerance_policy(-1.0, 1).expect_err("negative rcond");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");

        let err = validate_tolerance_policy(1e-6, MAX_TOLERANCE_SEARCH_DEPTH + 1)
            .expect_err("search-depth overflow");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn backend_bridge_enforces_support_and_budget() {
        validate_backend_bridge(true, 0).expect("baseline backend pass");
        validate_backend_bridge(true, MAX_BACKEND_REVALIDATION_ATTEMPTS).expect("edge budget pass");

        let unsupported =
            validate_backend_bridge(false, 0).expect_err("unsupported backend must fail");
        assert_eq!(unsupported.reason_code(), "linalg_backend_bridge_invalid");

        let overflow = validate_backend_bridge(true, MAX_BACKEND_REVALIDATION_ATTEMPTS + 1)
            .expect_err("revalidation budget overflow");
        assert_eq!(overflow.reason_code(), "linalg_backend_bridge_invalid");
    }

    #[test]
    fn policy_metadata_is_fail_closed_for_unknowns() {
        validate_policy_metadata("strict", "known_compatible_low_risk").expect("known strict");
        validate_policy_metadata("hardened", "known_incompatible_semantics")
            .expect("known hardened");

        let err = validate_policy_metadata("weird", "known_compatible_low_risk")
            .expect_err("unknown mode should fail");
        assert_eq!(err.reason_code(), "linalg_policy_unknown_metadata");
    }

    #[test]
    fn packet008_log_record_is_replay_complete() {
        let record = LinAlgLogRecord {
            ts_utc: "2026-02-16T00:00:00Z".to_string(),
            suite_id: "fnp-linalg::tests".to_string(),
            test_id: "UP-008-solve-inv-contract".to_string(),
            packet_id: LINALG_PACKET_ID.to_string(),
            fixture_id: "UP-008-solve-inv-contract".to_string(),
            mode: LinAlgRuntimeMode::Strict,
            seed: 8008,
            input_digest: "sha256:input".to_string(),
            output_digest: "sha256:output".to_string(),
            env_fingerprint: "fnp-linalg-unit-tests".to_string(),
            artifact_refs: packet008_artifacts(),
            duration_ms: 1,
            outcome: "pass".to_string(),
            reason_code: "linalg_solver_singularity".to_string(),
        };
        assert!(record.is_replay_complete());
    }

    #[test]
    fn packet008_log_record_rejects_missing_fields() {
        let record = LinAlgLogRecord {
            ts_utc: String::new(),
            suite_id: String::new(),
            test_id: String::new(),
            packet_id: "wrong-packet".to_string(),
            fixture_id: String::new(),
            mode: LinAlgRuntimeMode::Hardened,
            seed: 9001,
            input_digest: String::new(),
            output_digest: String::new(),
            env_fingerprint: String::new(),
            artifact_refs: vec![String::new()],
            duration_ms: 0,
            outcome: "unknown".to_string(),
            reason_code: String::new(),
        };
        assert!(!record.is_replay_complete());
    }

    #[test]
    fn packet008_reason_codes_round_trip_into_logs() {
        for (idx, reason_code) in LINALG_PACKET_REASON_CODES.iter().enumerate() {
            let seed = u64::try_from(idx).expect("small index");
            let record = LinAlgLogRecord {
                ts_utc: "2026-02-16T00:00:00Z".to_string(),
                suite_id: "fnp-linalg::tests".to_string(),
                test_id: format!("UP-008-{idx}"),
                packet_id: LINALG_PACKET_ID.to_string(),
                fixture_id: format!("UP-008-{idx}"),
                mode: LinAlgRuntimeMode::Strict,
                seed: 10_000 + seed,
                input_digest: "sha256:input".to_string(),
                output_digest: "sha256:output".to_string(),
                env_fingerprint: "fnp-linalg-unit-tests".to_string(),
                artifact_refs: packet008_artifacts(),
                duration_ms: 1,
                outcome: "pass".to_string(),
                reason_code: (*reason_code).to_string(),
            };
            assert!(record.is_replay_complete());
            assert_eq!(record.reason_code, *reason_code);
        }
    }
}
