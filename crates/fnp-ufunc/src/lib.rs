#![forbid(unsafe_code)]

use fnp_dtype::{DType, promote, promote_for_mean_reduction, promote_for_sum_reduction};
use fnp_ndarray::{ShapeError, broadcast_shape, element_count};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UFuncRuntimeMode {
    Strict,
    Hardened,
}

impl UFuncRuntimeMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hardened => "hardened",
        }
    }
}

pub const UFUNC_PACKET_REASON_CODES: [&str; 16] = [
    "ufunc_shape_contract_violation",
    "ufunc_invalid_input_length",
    "ufunc_axis_out_of_bounds",
    "ufunc_division_by_zero_observed",
    "ufunc_broadcast_selector_determinism",
    "ufunc_reduce_keepdims_contract",
    "ufunc_reduce_axis_contract",
    "ufunc_scalar_broadcast_contract",
    "ufunc_dtype_promotion_contract",
    "ufunc_signature_conflict",
    "ufunc_signature_parse_failed",
    "ufunc_fixed_signature_invalid",
    "ufunc_override_precedence_violation",
    "gufunc_loop_exception_propagated",
    "ufunc_loop_registry_invalid",
    "ufunc_policy_unknown_metadata",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Power,
    Remainder,
    Minimum,
    Maximum,
    Arctan2,
    Fmod,
    Copysign,
    Fmax,
    Fmin,
}

impl BinaryOp {
    #[must_use]
    pub fn apply(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Add => lhs + rhs,
            Self::Sub => lhs - rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
            Self::Power => lhs.powf(rhs),
            Self::Remainder => {
                // NumPy remainder follows Python semantics: sign of result matches divisor
                if rhs == 0.0 {
                    f64::NAN
                } else {
                    lhs - (lhs / rhs).floor() * rhs
                }
            }
            Self::Minimum => {
                if lhs.is_nan() || rhs.is_nan() {
                    f64::NAN
                } else {
                    lhs.min(rhs)
                }
            }
            Self::Maximum => {
                if lhs.is_nan() || rhs.is_nan() {
                    f64::NAN
                } else {
                    lhs.max(rhs)
                }
            }
            Self::Arctan2 => lhs.atan2(rhs),
            Self::Fmod => {
                // C-style fmod: sign of result matches dividend
                if rhs == 0.0 {
                    f64::NAN
                } else {
                    lhs % rhs
                }
            }
            Self::Copysign => lhs.copysign(rhs),
            // fmax/fmin ignore NaN (return the non-NaN value)
            Self::Fmax => lhs.max(rhs),
            Self::Fmin => lhs.min(rhs),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Abs,
    Negative,
    Sign,
    Sqrt,
    Square,
    Exp,
    Log,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
    Round,
    Reciprocal,
    Sinh,
    Cosh,
    Tanh,
    Arcsin,
    Arccos,
    Arctan,
    Cbrt,
    Expm1,
    Log1p,
    Degrees,
    Radians,
    Rint,
    Trunc,
}

impl UnaryOp {
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Abs => x.abs(),
            Self::Negative => -x,
            Self::Sign => {
                if x.is_nan() {
                    f64::NAN
                } else if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Sqrt => x.sqrt(),
            Self::Square => x * x,
            Self::Exp => x.exp(),
            Self::Log => x.ln(),
            Self::Log2 => x.log2(),
            Self::Log10 => x.log10(),
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Tan => x.tan(),
            Self::Floor => x.floor(),
            Self::Ceil => x.ceil(),
            Self::Round => {
                // NumPy uses banker's rounding (round half to even)
                let rounded = x.round();
                if (x - rounded).abs() == 0.0 {
                    rounded
                } else {
                    let frac = (x - x.floor()).abs();
                    if (frac - 0.5).abs() < f64::EPSILON {
                        // Exactly 0.5 — round to even
                        let floor = x.floor();
                        let ceil = x.ceil();
                        if floor % 2.0 == 0.0 {
                            floor
                        } else {
                            ceil
                        }
                    } else {
                        rounded
                    }
                }
            }
            Self::Reciprocal => 1.0 / x,
            Self::Sinh => x.sinh(),
            Self::Cosh => x.cosh(),
            Self::Tanh => x.tanh(),
            Self::Arcsin => x.asin(),
            Self::Arccos => x.acos(),
            Self::Arctan => x.atan(),
            Self::Cbrt => x.cbrt(),
            Self::Expm1 => x.exp_m1(),
            Self::Log1p => x.ln_1p(),
            Self::Degrees => x.to_degrees(),
            Self::Radians => x.to_radians(),
            Self::Rint => {
                // Same as Round: banker's rounding (round half to even)
                let rounded = x.round();
                if (x - rounded).abs() == 0.0 {
                    rounded
                } else {
                    let frac = (x - x.floor()).abs();
                    if (frac - 0.5).abs() < f64::EPSILON {
                        let floor = x.floor();
                        let ceil = x.ceil();
                        if floor % 2.0 == 0.0 {
                            floor
                        } else {
                            ceil
                        }
                    } else {
                        rounded
                    }
                }
            }
            Self::Trunc => x.trunc(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GufuncSignature {
    pub inputs: Vec<Vec<String>>,
    pub outputs: Vec<Vec<String>>,
    canonical: String,
}

impl GufuncSignature {
    #[must_use]
    pub fn canonical(&self) -> &str {
        &self.canonical
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryDispatchPlan {
    pub out_shape: Vec<usize>,
    pub out_count: usize,
    pub out_dtype: DType,
}

pub fn normalize_signature_keywords(
    sig: Option<&str>,
    signature: Option<&str>,
) -> Result<Option<String>, UFuncError> {
    let sig_trimmed = sig.map(str::trim);
    let signature_trimmed = signature.map(str::trim);

    if sig.is_some() && sig_trimmed.is_some_and(str::is_empty) {
        return Err(UFuncError::FixedSignatureInvalid {
            detail: "sig keyword must not be empty".to_string(),
        });
    }

    if signature.is_some() && signature_trimmed.is_some_and(str::is_empty) {
        return Err(UFuncError::FixedSignatureInvalid {
            detail: "signature keyword must not be empty".to_string(),
        });
    }

    match (sig_trimmed, signature_trimmed) {
        (Some(sig_raw), Some(signature_raw)) => {
            if sig_raw != signature_raw {
                return Err(UFuncError::SignatureConflict {
                    sig: sig_raw.to_string(),
                    signature: signature_raw.to_string(),
                });
            }
            Ok(Some(sig_raw.to_string()))
        }
        (Some(sig_raw), None) => Ok(Some(sig_raw.to_string())),
        (None, Some(signature_raw)) => Ok(Some(signature_raw.to_string())),
        (None, None) => Ok(None),
    }
}

pub fn parse_gufunc_signature(
    sig: Option<&str>,
    signature: Option<&str>,
) -> Result<Option<GufuncSignature>, UFuncError> {
    let Some(normalized) = normalize_signature_keywords(sig, signature)? else {
        return Ok(None);
    };

    let (inputs_raw, outputs_raw) =
        normalized
            .split_once("->")
            .ok_or_else(|| UFuncError::SignatureParse {
                detail: format!(
                    "signature '{}' must contain exactly one '->' separator",
                    normalized
                ),
            })?;

    if outputs_raw.contains("->") {
        return Err(UFuncError::SignatureParse {
            detail: format!(
                "signature '{}' must contain exactly one '->' separator",
                normalized
            ),
        });
    }

    let inputs = parse_signature_groups(inputs_raw, "input signature")?;
    let outputs = parse_signature_groups(outputs_raw, "output signature")?;
    let canonical = format!(
        "{}->{}",
        canonicalize_signature_groups(&inputs),
        canonicalize_signature_groups(&outputs)
    );
    Ok(Some(GufuncSignature {
        inputs,
        outputs,
        canonical,
    }))
}

pub fn validate_override_payload_class(payload_class: &str) -> Result<(), UFuncError> {
    let normalized = payload_class.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(UFuncError::OverridePrecedenceViolation {
            detail: "override payload class must not be empty".to_string(),
        });
    }

    if matches!(
        normalized.as_str(),
        "ndarray" | "notimplemented" | "not_implemented" | "override_result"
    ) {
        Ok(())
    } else {
        Err(UFuncError::OverridePrecedenceViolation {
            detail: format!("unsupported override payload class '{payload_class}'"),
        })
    }
}

pub fn register_custom_loop(loop_name: &str) -> Result<(), UFuncError> {
    let loop_name = loop_name.trim();
    if loop_name.is_empty() {
        return Err(UFuncError::LoopRegistryInvalid {
            detail: "custom loop name must not be empty".to_string(),
        });
    }

    Err(UFuncError::LoopRegistryInvalid {
        detail: format!("custom loop registration unsupported for '{loop_name}'"),
    })
}

pub fn plan_binary_dispatch(
    lhs: &UFuncArray,
    rhs: &UFuncArray,
) -> Result<BinaryDispatchPlan, UFuncError> {
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(UFuncError::Shape)?;
    let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
    let out_dtype = promote(lhs.dtype(), rhs.dtype());
    Ok(BinaryDispatchPlan {
        out_shape,
        out_count,
        out_dtype,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct UFuncArray {
    shape: Vec<usize>,
    values: Vec<f64>,
    dtype: DType,
}

impl UFuncArray {
    pub fn new(shape: Vec<usize>, values: Vec<f64>, dtype: DType) -> Result<Self, UFuncError> {
        let expected = element_count(&shape).map_err(UFuncError::Shape)?;
        if values.len() != expected {
            return Err(UFuncError::InvalidInputLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            shape,
            values,
            dtype,
        })
    }

    #[must_use]
    pub fn scalar(value: f64, dtype: DType) -> Self {
        Self {
            shape: Vec::new(),
            values: vec![value],
            dtype,
        }
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn elementwise_binary(&self, rhs: &Self, op: BinaryOp) -> Result<Self, UFuncError> {
        let plan = plan_binary_dispatch(self, rhs)?;
        let out_shape = plan.out_shape;
        let out_count = plan.out_count;
        let out_dtype = plan.out_dtype;

        if self.shape == rhs.shape {
            let values = self
                .values
                .iter()
                .zip(&rhs.values)
                .map(|(&lhs, &rhs)| op.apply(lhs, rhs))
                .collect::<Vec<_>>();
            return Ok(Self {
                shape: out_shape,
                values,
                dtype: out_dtype,
            });
        }

        let lhs_strides = contiguous_strides_elems(&self.shape);
        let rhs_strides = contiguous_strides_elems(&rhs.shape);
        let lhs_axis_steps =
            aligned_broadcast_axis_steps(out_shape.len(), &self.shape, &lhs_strides);
        let rhs_axis_steps =
            aligned_broadcast_axis_steps(out_shape.len(), &rhs.shape, &rhs_strides);

        let mut out_multi = vec![0usize; out_shape.len()];
        let mut lhs_flat = 0usize;
        let mut rhs_flat = 0usize;
        let mut out_values = Vec::with_capacity(out_count);

        for flat in 0..out_count {
            out_values.push(op.apply(self.values[lhs_flat], rhs.values[rhs_flat]));

            // Increment the output index as an odometer and adjust source flat
            // indices incrementally. This avoids re-unraveling and re-mapping
            // every output element.
            if flat + 1 == out_count || out_shape.is_empty() {
                continue;
            }

            for axis in (0..out_shape.len()).rev() {
                out_multi[axis] += 1;
                lhs_flat += lhs_axis_steps[axis];
                rhs_flat += rhs_axis_steps[axis];

                if out_multi[axis] < out_shape[axis] {
                    break;
                }

                out_multi[axis] = 0;
                lhs_flat -= lhs_axis_steps[axis] * out_shape[axis];
                rhs_flat -= rhs_axis_steps[axis] * out_shape[axis];
            }
        }

        Ok(Self {
            shape: out_shape,
            values: out_values,
            dtype: out_dtype,
        })
    }

    #[must_use]
    pub fn elementwise_unary(&self, op: UnaryOp) -> Self {
        let values = self.values.iter().map(|&v| op.apply(v)).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        }
    }

    pub fn reduce_sum(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        let out_dtype = promote_for_sum_reduction(self.dtype);
        match axis {
            None => {
                let sum: f64 = self.values.iter().copied().sum();
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![sum],
                    dtype: out_dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![0.0f64; out_count];
                reduce_sum_axis_contiguous(&self.values, &self.shape, axis, &mut out_values);

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: out_dtype,
                })
            }
        }
    }

    pub fn reduce_prod(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        let out_dtype = promote_for_sum_reduction(self.dtype);
        match axis {
            None => {
                let prod: f64 = self.values.iter().copied().product();
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![prod],
                    dtype: out_dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![1.0f64; out_count];
                reduce_fold_axis_contiguous(
                    &self.values,
                    &self.shape,
                    axis,
                    &mut out_values,
                    |acc, v| acc * v,
                );

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: out_dtype,
                })
            }
        }
    }

    pub fn reduce_min(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let min = self
                    .values
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![min],
                    dtype: self.dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![f64::INFINITY; out_count];
                reduce_fold_axis_contiguous(
                    &self.values,
                    &self.shape,
                    axis,
                    &mut out_values,
                    f64::min,
                );

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: self.dtype,
                })
            }
        }
    }

    pub fn reduce_max(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let max = self
                    .values
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![max],
                    dtype: self.dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![f64::NEG_INFINITY; out_count];
                reduce_fold_axis_contiguous(
                    &self.values,
                    &self.shape,
                    axis,
                    &mut out_values,
                    f64::max,
                );

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: self.dtype,
                })
            }
        }
    }

    pub fn reduce_mean(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        let out_dtype = promote_for_mean_reduction(self.dtype);
        match axis {
            None => {
                let n = self.values.len() as f64;
                let sum: f64 = self.values.iter().copied().sum();
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![sum / n],
                    dtype: out_dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let axis_len = self.shape[axis] as f64;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![0.0f64; out_count];
                reduce_sum_axis_contiguous(&self.values, &self.shape, axis, &mut out_values);
                for v in &mut out_values {
                    *v /= axis_len;
                }

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: out_dtype,
                })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UFuncError {
    Shape(ShapeError),
    InvalidInputLength { expected: usize, actual: usize },
    AxisOutOfBounds { axis: isize, ndim: usize },
    SignatureConflict { sig: String, signature: String },
    SignatureParse { detail: String },
    FixedSignatureInvalid { detail: String },
    OverridePrecedenceViolation { detail: String },
    LoopRegistryInvalid { detail: String },
}

impl std::fmt::Display for UFuncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(err) => write!(f, "shape error: {err}"),
            Self::InvalidInputLength { expected, actual } => {
                write!(
                    f,
                    "invalid input length expected={expected} actual={actual}"
                )
            }
            Self::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis {axis} out of bounds for ndim={ndim}")
            }
            Self::SignatureConflict { sig, signature } => {
                write!(
                    f,
                    "ufunc signature conflict: sig='{sig}' differs from signature='{signature}'"
                )
            }
            Self::SignatureParse { detail } => {
                write!(f, "ufunc signature parse failed: {detail}")
            }
            Self::FixedSignatureInvalid { detail } => {
                write!(f, "ufunc fixed signature invalid: {detail}")
            }
            Self::OverridePrecedenceViolation { detail } => {
                write!(f, "ufunc override precedence violation: {detail}")
            }
            Self::LoopRegistryInvalid { detail } => {
                write!(f, "ufunc loop registry invalid: {detail}")
            }
        }
    }
}

impl std::error::Error for UFuncError {}

impl UFuncError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::Shape(_) => "ufunc_shape_contract_violation",
            Self::InvalidInputLength { .. } => "ufunc_invalid_input_length",
            Self::AxisOutOfBounds { .. } => "ufunc_axis_out_of_bounds",
            Self::SignatureConflict { .. } => "ufunc_signature_conflict",
            Self::SignatureParse { .. } => "ufunc_signature_parse_failed",
            Self::FixedSignatureInvalid { .. } => "ufunc_fixed_signature_invalid",
            Self::OverridePrecedenceViolation { .. } => "ufunc_override_precedence_violation",
            Self::LoopRegistryInvalid { .. } => "ufunc_loop_registry_invalid",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UFuncLogRecord {
    pub fixture_id: String,
    pub seed: u64,
    pub mode: UFuncRuntimeMode,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub reason_code: String,
    pub passed: bool,
}

impl UFuncLogRecord {
    #[must_use]
    pub fn is_replay_complete(&self) -> bool {
        !self.fixture_id.trim().is_empty()
            && !self.mode.as_str().is_empty()
            && !self.env_fingerprint.trim().is_empty()
            && !self.reason_code.trim().is_empty()
            && !self.artifact_refs.is_empty()
            && self
                .artifact_refs
                .iter()
                .all(|artifact| !artifact.trim().is_empty())
    }
}

#[must_use]
fn canonicalize_signature_groups(groups: &[Vec<String>]) -> String {
    groups
        .iter()
        .map(|group| format!("({})", group.join(",")))
        .collect::<Vec<_>>()
        .join(",")
}

fn validate_core_dim_identifier(token: &str) -> Result<(), UFuncError> {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return Err(UFuncError::SignatureParse {
            detail: "signature dimension token must not be empty".to_string(),
        });
    };

    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(UFuncError::SignatureParse {
            detail: format!("signature dimension '{}' must start with [A-Za-z_]", token),
        });
    }

    if chars.any(|ch| !(ch.is_ascii_alphanumeric() || ch == '_')) {
        return Err(UFuncError::SignatureParse {
            detail: format!(
                "signature dimension '{}' must contain only [A-Za-z0-9_]",
                token
            ),
        });
    }

    Ok(())
}

fn parse_signature_dims(raw: &str) -> Result<Vec<String>, UFuncError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let mut dims = Vec::new();
    for token in trimmed.split(',') {
        let dim = token.trim();
        if dim.is_empty() {
            return Err(UFuncError::SignatureParse {
                detail: "signature dimension list contains an empty token".to_string(),
            });
        }
        validate_core_dim_identifier(dim)?;
        dims.push(dim.to_string());
    }
    Ok(dims)
}

fn parse_signature_groups(raw: &str, side: &str) -> Result<Vec<Vec<String>>, UFuncError> {
    let bytes = raw.as_bytes();
    let mut idx = 0usize;
    let mut groups = Vec::new();

    while idx < bytes.len() {
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= bytes.len() {
            break;
        }

        if bytes[idx] != b'(' {
            return Err(UFuncError::SignatureParse {
                detail: format!("{side} must start tuple groups with '(' at byte {idx}"),
            });
        }
        idx += 1;
        let tuple_start = idx;
        let mut depth = 1usize;
        while idx < bytes.len() && depth > 0 {
            match bytes[idx] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            idx += 1;
        }

        if depth != 0 {
            return Err(UFuncError::SignatureParse {
                detail: format!("{side} has an unclosed tuple group"),
            });
        }

        let tuple_body = &raw[tuple_start..idx - 1];
        if tuple_body.contains('(') || tuple_body.contains(')') {
            return Err(UFuncError::SignatureParse {
                detail: format!("{side} contains nested tuple groups, which are unsupported"),
            });
        }
        groups.push(parse_signature_dims(tuple_body)?);

        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= bytes.len() {
            break;
        }

        if bytes[idx] != b',' {
            return Err(UFuncError::SignatureParse {
                detail: format!("{side} expected ',' between tuple groups at byte {idx}"),
            });
        }
        idx += 1;
    }

    if groups.is_empty() {
        return Err(UFuncError::SignatureParse {
            detail: format!("{side} must contain at least one tuple group"),
        });
    }
    Ok(groups)
}

#[must_use]
fn contiguous_strides_elems(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for (idx, &dim) in shape.iter().enumerate().rev() {
        strides[idx] = stride;
        stride = stride.saturating_mul(dim);
    }
    strides
}

#[must_use]
fn aligned_broadcast_axis_steps(
    out_ndim: usize,
    src_shape: &[usize],
    src_strides: &[usize],
) -> Vec<usize> {
    if out_ndim == 0 {
        return Vec::new();
    }

    let mut axis_steps = vec![0usize; out_ndim];
    let offset = out_ndim - src_shape.len();

    for (axis, (&dim, &stride)) in src_shape.iter().zip(src_strides).enumerate() {
        axis_steps[axis + offset] = if dim == 1 { 0 } else { stride };
    }

    axis_steps
}

fn reduce_sum_axis_contiguous(
    values: &[f64],
    shape: &[usize],
    axis: usize,
    out_values: &mut [f64],
) {
    debug_assert!(axis < shape.len());
    if out_values.is_empty() {
        return;
    }

    let axis_len = shape[axis];
    if axis_len == 0 {
        return;
    }

    let inner = shape[axis + 1..].iter().copied().product::<usize>();
    let outer = shape[..axis].iter().copied().product::<usize>();

    let mut out_flat = 0usize;
    for outer_idx in 0..outer {
        let base = outer_idx * axis_len * inner;
        for inner_idx in 0..inner {
            let mut sum = 0.0f64;
            let mut offset = base + inner_idx;
            for _ in 0..axis_len {
                sum += values[offset];
                offset += inner;
            }
            out_values[out_flat] = sum;
            out_flat += 1;
        }
    }
}

fn reduce_fold_axis_contiguous(
    values: &[f64],
    shape: &[usize],
    axis: usize,
    out_values: &mut [f64],
    fold: impl Fn(f64, f64) -> f64,
) {
    debug_assert!(axis < shape.len());
    if out_values.is_empty() {
        return;
    }

    let axis_len = shape[axis];
    if axis_len == 0 {
        return;
    }

    let inner = shape[axis + 1..].iter().copied().product::<usize>();
    let outer = shape[..axis].iter().copied().product::<usize>();

    let mut out_flat = 0usize;
    for outer_idx in 0..outer {
        let base = outer_idx * axis_len * inner;
        for inner_idx in 0..inner {
            let mut offset = base + inner_idx;
            let mut acc = values[offset];
            offset += inner;
            for _ in 1..axis_len {
                acc = fold(acc, values[offset]);
                offset += inner;
            }
            out_values[out_flat] = acc;
            out_flat += 1;
        }
    }
}

#[must_use]
fn reduced_shape(shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(idx, &dim)| if idx == axis { 1 } else { dim })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(idx, &dim)| (idx != axis).then_some(dim))
            .collect()
    }
}

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize, UFuncError> {
    let ndim_i128 = i128::try_from(ndim).expect("ndim should always fit into i128");
    let axis_i128 = i128::try_from(axis).expect("isize should always fit into i128");
    let normalized = if axis_i128 < 0 {
        ndim_i128 + axis_i128
    } else {
        axis_i128
    };
    if normalized < 0 || normalized >= ndim_i128 {
        return Err(UFuncError::AxisOutOfBounds { axis, ndim });
    }
    usize::try_from(normalized).map_err(|_| UFuncError::AxisOutOfBounds { axis, ndim })
}

#[cfg(test)]
mod tests {
    use super::{
        BinaryOp, UFUNC_PACKET_REASON_CODES, UFuncArray, UFuncError, UFuncLogRecord,
        UFuncRuntimeMode, UnaryOp, normalize_signature_keywords, parse_gufunc_signature,
        plan_binary_dispatch, register_custom_loop, validate_override_payload_class,
    };
    use fnp_dtype::{DType, promote};
    use fnp_ndarray::broadcast_shape;

    fn packet005_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-005/fixture_manifest.json".to_string(),
            "artifacts/phase2c/FNP-P2C-005/parity_gate.yaml".to_string(),
        ]
    }

    #[test]
    fn broadcasted_add_matches_expected_values() {
        let lhs = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64).expect("rhs");

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Add)
            .expect("broadcasted add");

        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn scalar_broadcast_mul() {
        let lhs = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::scalar(0.5, DType::F64);

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Mul)
            .expect("scalar mul");

        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn division_by_zero_yields_infinity_semantics() {
        let lhs = UFuncArray::new(vec![2], vec![1.0, -1.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![1], vec![0.0], DType::F64).expect("rhs");

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Div)
            .expect("div should execute");

        assert!(out.values()[0].is_infinite() && out.values()[0].is_sign_positive());
        assert!(out.values()[1].is_infinite() && out.values()[1].is_sign_negative());
    }

    #[test]
    fn reduce_sum_axis_none_keepdims_false() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_sum(None, false).expect("sum");
        assert_eq!(out.shape(), &[]);
        assert_eq!(out.values(), &[21.0]);
    }

    #[test]
    fn reduce_sum_axis_none_keepdims_true() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_sum(None, true).expect("sum");
        assert_eq!(out.shape(), &[1, 1]);
        assert_eq!(out.values(), &[21.0]);
    }

    #[test]
    fn reduce_sum_axis_keepdims_variants() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out_drop = arr.reduce_sum(Some(1), false).expect("sum axis=1");
        assert_eq!(out_drop.shape(), &[2]);
        assert_eq!(out_drop.values(), &[6.0, 15.0]);

        let out_keep = arr.reduce_sum(Some(1), true).expect("sum axis=1 keepdims");
        assert_eq!(out_keep.shape(), &[2, 1]);
        assert_eq!(out_keep.values(), &[6.0, 15.0]);
    }

    #[test]
    fn reduce_sum_axis_zero_preserves_c_order() {
        let arr = UFuncArray::new(
            vec![2, 3, 2],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            DType::F64,
        )
        .expect("arr");

        let out_drop = arr.reduce_sum(Some(0), false).expect("sum axis=0");
        assert_eq!(out_drop.shape(), &[3, 2]);
        assert_eq!(out_drop.values(), &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);

        let out_keep = arr.reduce_sum(Some(0), true).expect("sum axis=0 keepdims");
        assert_eq!(out_keep.shape(), &[1, 3, 2]);
        assert_eq!(out_keep.values(), &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn reduce_sum_empty_axis_returns_zero_initialized_outputs() {
        let arr = UFuncArray::new(vec![2, 0, 3], Vec::new(), DType::F64).expect("arr");
        let out = arr.reduce_sum(Some(1), false).expect("sum axis=1");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn mixed_dtype_high_rank_add_promotes_and_broadcasts() {
        let lhs =
            UFuncArray::new(vec![2, 1, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I32).expect("lhs");
        let rhs = UFuncArray::new(vec![1, 3, 1], vec![0.5, -1.0, 2.0], DType::F32).expect("rhs");

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Add)
            .expect("broadcasted mixed-dtype add");

        assert_eq!(out.shape(), &[2, 3, 2]);
        assert_eq!(out.dtype(), DType::F64);
        assert_eq!(
            out.values(),
            &[1.5, 2.5, 0.0, 1.0, 3.0, 4.0, 3.5, 4.5, 2.0, 3.0, 5.0, 6.0]
        );
    }

    #[test]
    fn mixed_dtype_bool_i64_mul_promotes_to_i64() {
        let lhs =
            UFuncArray::new(vec![2, 1, 2], vec![1.0, 0.0, 0.0, 1.0], DType::Bool).expect("lhs");
        let rhs = UFuncArray::new(vec![1, 3, 1], vec![10.0, -2.0, 7.0], DType::I64).expect("rhs");

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Mul)
            .expect("broadcasted mixed-dtype mul");

        assert_eq!(out.shape(), &[2, 3, 2]);
        assert_eq!(out.dtype(), DType::I64);
        assert_eq!(
            out.values(),
            &[
                10.0, 0.0, -2.0, 0.0, 7.0, 0.0, 0.0, 10.0, 0.0, -2.0, 0.0, 7.0
            ]
        );
    }

    #[test]
    fn reduce_sum_axis_one_keepdims_promotes_i32_to_i64() {
        let arr = UFuncArray::new(
            vec![2, 2, 2],
            vec![-3.0, 1.0, 5.0, -7.0, 9.0, 11.0, -13.0, 15.0],
            DType::I32,
        )
        .expect("arr");

        let out = arr.reduce_sum(Some(1), true).expect("sum axis=1 keepdims");
        assert_eq!(out.shape(), &[2, 1, 2]);
        assert_eq!(out.values(), &[2.0, -6.0, -4.0, 26.0]);
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn reduce_sum_promotes_bool_to_i64() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], DType::Bool)
            .expect("arr");

        let out = arr.reduce_sum(None, false).expect("sum all");
        assert_eq!(out.values(), &[4.0]);
        assert_eq!(out.dtype(), DType::I64);

        let axis0 = arr.reduce_sum(Some(0), false).expect("sum axis=0");
        assert_eq!(axis0.values(), &[1.0, 1.0, 2.0]);
        assert_eq!(axis0.dtype(), DType::I64);
    }

    #[test]
    fn reduce_sum_preserves_f32_dtype() {
        let arr =
            UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F32).expect("arr");

        let out = arr.reduce_sum(None, false).expect("sum all");
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn reduce_sum_preserves_i64_dtype() {
        let arr =
            UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I64).expect("arr");

        let out = arr.reduce_sum(None, false).expect("sum all");
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn reduce_sum_propagates_non_finite_values() {
        let arr = UFuncArray::new(
            vec![2, 2, 2],
            vec![
                1.0,
                f64::INFINITY,
                -1.0,
                2.0,
                3.0,
                f64::NEG_INFINITY,
                f64::NAN,
                4.0,
            ],
            DType::F64,
        )
        .expect("arr");

        let axis_one = arr.reduce_sum(Some(1), true).expect("sum axis=1 keepdims");
        assert_eq!(axis_one.shape(), &[2, 1, 2]);
        assert_eq!(axis_one.values()[0], 0.0);
        assert!(axis_one.values()[1].is_infinite() && axis_one.values()[1].is_sign_positive());
        assert!(axis_one.values()[2].is_nan());
        assert!(axis_one.values()[3].is_infinite() && axis_one.values()[3].is_sign_negative());

        let full = arr.reduce_sum(None, false).expect("sum all");
        assert_eq!(full.shape(), &[]);
        assert!(full.values()[0].is_nan());
    }

    #[test]
    fn invalid_input_length_is_rejected() {
        let err = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0], DType::F64)
            .expect_err("should reject length mismatch");
        assert!(matches!(err, UFuncError::InvalidInputLength { .. }));
    }

    #[test]
    fn normalize_signature_keywords_accepts_matching_sig_and_signature() {
        let normalized = normalize_signature_keywords(
            Some(" (batch,m),(m,n)->(batch,n) "),
            Some("(batch,m),(m,n)->(batch,n)"),
        )
        .expect("matching signature keywords should normalize");
        assert_eq!(normalized.as_deref(), Some("(batch,m),(m,n)->(batch,n)"));
    }

    #[test]
    fn parse_gufunc_signature_conflict_is_rejected() {
        let err = parse_gufunc_signature(Some("(i)->(i)"), Some("(j)->(j)"))
            .expect_err("conflicting sig/signature should fail");
        assert!(matches!(err, UFuncError::SignatureConflict { .. }));
        assert_eq!(err.reason_code(), "ufunc_signature_conflict");
    }

    #[test]
    fn parse_gufunc_signature_normalizes_grammar() {
        let signature = parse_gufunc_signature(Some(" (batch,m),(m,n) -> (batch,n) "), None)
            .expect("valid signature should parse")
            .expect("signature should be present");
        assert_eq!(signature.inputs.len(), 2);
        assert_eq!(signature.outputs.len(), 1);
        assert_eq!(signature.inputs[0], vec!["batch", "m"]);
        assert_eq!(signature.inputs[1], vec!["m", "n"]);
        assert_eq!(signature.outputs[0], vec!["batch", "n"]);
        assert_eq!(signature.canonical(), "(batch,m),(m,n)->(batch,n)");
    }

    #[test]
    fn parse_gufunc_signature_invalid_grammar_is_rejected() {
        let err = parse_gufunc_signature(Some("(i,j)->i,j"), None)
            .expect_err("invalid grammar must fail");
        assert!(matches!(err, UFuncError::SignatureParse { .. }));
        assert_eq!(err.reason_code(), "ufunc_signature_parse_failed");
    }

    #[test]
    fn dispatch_plan_is_deterministic() {
        let lhs = UFuncArray::new(
            vec![2, 1, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            DType::I32,
        )
        .expect("lhs");
        let rhs =
            UFuncArray::new(vec![1, 4, 1], vec![10.0, 20.0, 30.0, 40.0], DType::F32).expect("rhs");

        let first = plan_binary_dispatch(&lhs, &rhs).expect("dispatch plan");
        let second = plan_binary_dispatch(&lhs, &rhs).expect("dispatch plan repeat");
        assert_eq!(first, second);
        assert_eq!(first.out_shape, vec![2, 4, 3]);
        assert_eq!(first.out_count, 24);
        assert_eq!(first.out_dtype, DType::F64);
    }

    #[test]
    fn override_payload_validation_rejects_unknown_class() {
        let err = validate_override_payload_class("bogus_payload")
            .expect_err("unknown override payload class must fail");
        assert!(matches!(
            err,
            UFuncError::OverridePrecedenceViolation { .. }
        ));
        assert_eq!(err.reason_code(), "ufunc_override_precedence_violation");
    }

    #[test]
    fn custom_loop_registration_is_fail_closed() {
        let err = register_custom_loop("fused_add_loop")
            .expect_err("loop registration is unsupported in packet-D boundary");
        assert!(matches!(err, UFuncError::LoopRegistryInvalid { .. }));
        assert_eq!(err.reason_code(), "ufunc_loop_registry_invalid");
    }

    #[test]
    fn axis_out_of_bounds_is_rejected() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("arr");
        let err = arr
            .reduce_sum(Some(2), false)
            .expect_err("axis should be invalid");
        assert!(matches!(
            err,
            UFuncError::AxisOutOfBounds { axis: 2, ndim: 2 }
        ));
        assert_eq!(err.reason_code(), "ufunc_axis_out_of_bounds");
    }

    #[test]
    fn negative_axis_wraps_from_end() {
        let arr = UFuncArray::new(vec![2, 2, 2], (1..=8).map(f64::from).collect(), DType::F64)
            .expect("arr");
        let out = arr
            .reduce_sum(Some(-1), false)
            .expect("axis=-1 should be valid");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[3.0, 7.0, 11.0, 15.0]);
    }

    #[test]
    fn negative_axis_out_of_bounds_is_rejected() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("arr");
        let err = arr
            .reduce_sum(Some(-3), false)
            .expect_err("axis should be invalid");
        assert!(matches!(
            err,
            UFuncError::AxisOutOfBounds { axis: -3, ndim: 2 }
        ));
        assert_eq!(err.reason_code(), "ufunc_axis_out_of_bounds");
    }

    #[test]
    fn reason_code_registry_matches_packet_contract() {
        assert_eq!(
            UFUNC_PACKET_REASON_CODES,
            [
                "ufunc_shape_contract_violation",
                "ufunc_invalid_input_length",
                "ufunc_axis_out_of_bounds",
                "ufunc_division_by_zero_observed",
                "ufunc_broadcast_selector_determinism",
                "ufunc_reduce_keepdims_contract",
                "ufunc_reduce_axis_contract",
                "ufunc_scalar_broadcast_contract",
                "ufunc_dtype_promotion_contract",
                "ufunc_signature_conflict",
                "ufunc_signature_parse_failed",
                "ufunc_fixed_signature_invalid",
                "ufunc_override_precedence_violation",
                "gufunc_loop_exception_propagated",
                "ufunc_loop_registry_invalid",
                "ufunc_policy_unknown_metadata",
            ]
        );
    }

    #[test]
    fn ufunc_error_reason_codes_cover_adversarial_paths() {
        let len_err = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0], DType::F64)
            .expect_err("invalid length should fail");
        assert_eq!(len_err.reason_code(), "ufunc_invalid_input_length");

        let lhs = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("rhs");
        let shape_err = lhs
            .elementwise_binary(&rhs, BinaryOp::Add)
            .expect_err("incompatible broadcast should fail");
        assert_eq!(shape_err.reason_code(), "ufunc_shape_contract_violation");
    }

    #[test]
    fn elementwise_binary_property_grid_is_deterministic() {
        let cases = [
            (
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                DType::F64,
                vec![3],
                vec![10.0, 20.0, 30.0],
                DType::F64,
            ),
            (
                vec![1, 3],
                vec![2.0, 4.0, 8.0],
                DType::I32,
                vec![2, 1, 3],
                vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                DType::F32,
            ),
            (
                vec![],
                vec![4.0],
                DType::I64,
                vec![2, 2],
                vec![1.0, 2.0, 3.0, 4.0],
                DType::I32,
            ),
        ];

        let ops = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div];

        for (lhs_shape, lhs_values, lhs_dtype, rhs_shape, rhs_values, rhs_dtype) in cases {
            let lhs =
                UFuncArray::new(lhs_shape.clone(), lhs_values.clone(), lhs_dtype).expect("lhs");
            let rhs =
                UFuncArray::new(rhs_shape.clone(), rhs_values.clone(), rhs_dtype).expect("rhs");
            let expected_dtype = promote(lhs_dtype, rhs_dtype);
            let expected_shape =
                broadcast_shape(&lhs_shape, &rhs_shape).expect("broadcast should work");

            for op in ops {
                let first = lhs
                    .elementwise_binary(&rhs, op)
                    .expect("operation should succeed");
                let second = lhs
                    .elementwise_binary(&rhs, op)
                    .expect("operation should be deterministic");
                assert_eq!(first.shape(), second.shape());
                assert_eq!(first.values(), second.values());
                assert_eq!(first.dtype(), second.dtype());
                assert_eq!(first.dtype(), expected_dtype);
                assert_eq!(first.shape(), expected_shape.as_slice());
            }
        }
    }

    #[test]
    fn reduce_sum_keepdims_shape_contract_holds() {
        let arr = UFuncArray::new(vec![2, 3, 4], (1..=24).map(f64::from).collect(), DType::F64)
            .expect("arr");

        for axis in 0_isize..3_isize {
            let keep = arr
                .reduce_sum(Some(axis), true)
                .expect("keepdims reduction should succeed");
            let drop = arr
                .reduce_sum(Some(axis), false)
                .expect("dropdims reduction should succeed");
            assert_eq!(keep.shape().len(), arr.shape().len());
            assert_eq!(drop.shape().len(), arr.shape().len() - 1);
            assert_eq!(keep.values(), drop.values());
        }
    }

    #[test]
    fn reduce_prod_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_prod(None, false).expect("prod all");
        assert_eq!(out.shape(), &[]);
        assert!((out.values()[0] - 720.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_prod_axis_one_keepdims() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_prod(Some(1), true).expect("prod axis=1");
        assert_eq!(out.shape(), &[2, 1]);
        assert!((out.values()[0] - 6.0).abs() < 1e-10);
        assert!((out.values()[1] - 120.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_prod_promotes_i32_to_i64() {
        let arr =
            UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I32).expect("arr");
        let out = arr.reduce_prod(None, false).expect("prod");
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn reduce_min_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 4.0, 1.5, 5.0, 2.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_min(None, false).expect("min all");
        assert_eq!(out.shape(), &[]);
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_min_axis_zero() {
        let arr = UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 4.0, 1.5, 5.0, 2.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_min(Some(0), false).expect("min axis=0");
        assert_eq!(out.shape(), &[3]);
        assert!((out.values()[0] - 1.5).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_min_preserves_dtype() {
        let arr =
            UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I32).expect("arr");
        let out = arr.reduce_min(None, false).expect("min");
        assert_eq!(out.dtype(), DType::I32);
    }

    #[test]
    fn reduce_max_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 4.0, 1.5, 5.0, 2.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_max(None, false).expect("max all");
        assert_eq!(out.shape(), &[]);
        assert!((out.values()[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_max_axis_one_keepdims() {
        let arr = UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 4.0, 1.5, 5.0, 2.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_max(Some(1), true).expect("max axis=1");
        assert_eq!(out.shape(), &[2, 1]);
        assert!((out.values()[0] - 4.0).abs() < 1e-10);
        assert!((out.values()[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_max_preserves_dtype() {
        let arr =
            UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::Bool).expect("arr");
        let out = arr.reduce_max(None, false).expect("max");
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn reduce_mean_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_mean(None, false).expect("mean all");
        assert_eq!(out.shape(), &[]);
        assert!((out.values()[0] - 3.5).abs() < 1e-10);
    }

    #[test]
    fn reduce_mean_axis_one() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_mean(Some(1), false).expect("mean axis=1");
        assert_eq!(out.shape(), &[2]);
        assert!((out.values()[0] - 2.0).abs() < 1e-10);
        assert!((out.values()[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_mean_promotes_i32_to_f64() {
        let arr =
            UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I32).expect("arr");
        let out = arr.reduce_mean(None, false).expect("mean");
        assert_eq!(out.dtype(), DType::F64);
    }

    #[test]
    fn reduce_mean_preserves_f32_dtype() {
        let arr =
            UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F32).expect("arr");
        let out = arr.reduce_mean(None, false).expect("mean");
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn reduce_mean_keepdims_shape_contract() {
        let arr = UFuncArray::new(vec![2, 3, 4], (1..=24).map(f64::from).collect(), DType::F64)
            .expect("arr");

        for axis in 0_isize..3_isize {
            let keep = arr
                .reduce_mean(Some(axis), true)
                .expect("keepdims reduction");
            let drop = arr
                .reduce_mean(Some(axis), false)
                .expect("dropdims reduction");
            assert_eq!(keep.shape().len(), arr.shape().len());
            assert_eq!(drop.shape().len(), arr.shape().len() - 1);
            assert_eq!(keep.values(), drop.values());
        }
    }

    #[test]
    fn unary_abs_mixed_signs() {
        let arr = UFuncArray::new(
            vec![2, 3],
            vec![-3.0, 1.0, -4.0, 1.5, -5.0, 0.0],
            DType::F64,
        )
        .expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Abs);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[3.0, 1.0, 4.0, 1.5, 5.0, 0.0]);
        assert_eq!(out.dtype(), DType::F64);
    }

    #[test]
    fn unary_negative() {
        let arr =
            UFuncArray::new(vec![4], vec![1.0, -2.0, 0.0, 3.5], DType::F64).expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Negative);
        assert_eq!(out.values(), &[-1.0, 2.0, 0.0, -3.5]);
    }

    #[test]
    fn unary_sign() {
        let arr = UFuncArray::new(
            vec![5],
            vec![-3.0, 0.0, 4.0, f64::NAN, -0.0],
            DType::F64,
        )
        .expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Sign);
        assert_eq!(out.values()[0], -1.0);
        assert_eq!(out.values()[1], 0.0);
        assert_eq!(out.values()[2], 1.0);
        assert!(out.values()[3].is_nan());
        assert_eq!(out.values()[4], 0.0);
    }

    #[test]
    fn unary_sqrt() {
        let arr =
            UFuncArray::new(vec![4], vec![0.0, 1.0, 4.0, 9.0], DType::F64).expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Sqrt);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
        assert!((out.values()[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn unary_square() {
        let arr =
            UFuncArray::new(vec![4], vec![-2.0, 0.0, 3.0, -1.5], DType::F64).expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Square);
        assert!((out.values()[0] - 4.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.0).abs() < 1e-10);
        assert!((out.values()[2] - 9.0).abs() < 1e-10);
        assert!((out.values()[3] - 2.25).abs() < 1e-10);
    }

    #[test]
    fn unary_exp() {
        let arr =
            UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Exp);
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
        assert!((out.values()[1] - std::f64::consts::E).abs() < 1e-10);
        assert!((out.values()[2] - 1.0 / std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn unary_log() {
        let arr =
            UFuncArray::new(vec![3], vec![1.0, std::f64::consts::E, 10.0], DType::F64)
                .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Log);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 10.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn unary_log2() {
        let arr =
            UFuncArray::new(vec![4], vec![1.0, 2.0, 4.0, 8.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Log2);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
        assert!((out.values()[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn unary_log10() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 10.0, 100.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Log10);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn unary_sin_cos_identity() {
        let arr = UFuncArray::new(
            vec![4],
            vec![0.0, 0.5, 1.0, 1.5],
            DType::F64,
        )
        .expect("arr");

        let s = arr.elementwise_unary(UnaryOp::Sin);
        let c = arr.elementwise_unary(UnaryOp::Cos);
        for i in 0..4 {
            let identity = s.values()[i] * s.values()[i] + c.values()[i] * c.values()[i];
            assert!(
                (identity - 1.0).abs() < 1e-10,
                "sin^2 + cos^2 != 1 at index {i}"
            );
        }
    }

    #[test]
    fn unary_tan() {
        let arr =
            UFuncArray::new(vec![3], vec![0.0, 0.25, 0.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Tan);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.25_f64.tan()).abs() < 1e-10);
        assert!((out.values()[2] - 0.5_f64.tan()).abs() < 1e-10);
    }

    #[test]
    fn unary_preserves_dtype() {
        let arr =
            UFuncArray::new(vec![2], vec![1.0, 4.0], DType::F32).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Sqrt);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn binary_power() {
        let lhs = UFuncArray::new(vec![3], vec![2.0, 3.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![3.0, 2.0, 0.5], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Power)
            .expect("power");
        assert!((out.values()[0] - 8.0).abs() < 1e-10);
        assert!((out.values()[1] - 9.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn binary_remainder_sign_follows_divisor() {
        let lhs = UFuncArray::new(vec![4], vec![7.0, -7.0, 7.0, -7.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![3.0, 3.0, -3.0, -3.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Remainder)
            .expect("remainder");
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
        assert!((out.values()[1] - 2.0).abs() < 1e-10);
        assert!((out.values()[2] - (-2.0)).abs() < 1e-10);
        assert!((out.values()[3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn binary_minimum() {
        let lhs = UFuncArray::new(vec![3], vec![1.0, 5.0, 3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![4.0, 2.0, 6.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Minimum)
            .expect("minimum");
        assert_eq!(out.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn binary_maximum() {
        let lhs = UFuncArray::new(vec![3], vec![1.0, 5.0, 3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![4.0, 2.0, 6.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Maximum)
            .expect("maximum");
        assert_eq!(out.values(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn binary_minimum_nan_propagation() {
        let lhs =
            UFuncArray::new(vec![3], vec![1.0, f64::NAN, 3.0], DType::F64).expect("lhs");
        let rhs =
            UFuncArray::new(vec![3], vec![f64::NAN, 2.0, f64::NAN], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Minimum)
            .expect("minimum");
        assert!(out.values()[0].is_nan());
        assert!(out.values()[1].is_nan());
        assert!(out.values()[2].is_nan());
    }

    #[test]
    fn unary_floor() {
        let arr = UFuncArray::new(
            vec![5],
            vec![1.7, -1.7, 2.5, -2.5, 0.0],
            DType::F64,
        )
        .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Floor);
        assert_eq!(out.values(), &[1.0, -2.0, 2.0, -3.0, 0.0]);
    }

    #[test]
    fn unary_ceil() {
        let arr = UFuncArray::new(
            vec![5],
            vec![1.7, -1.7, 2.5, -2.5, 0.0],
            DType::F64,
        )
        .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Ceil);
        assert_eq!(out.values(), &[2.0, -1.0, 3.0, -2.0, 0.0]);
    }

    #[test]
    fn unary_round_bankers() {
        let arr = UFuncArray::new(
            vec![6],
            vec![0.5, 1.5, 2.5, 3.5, 4.5, -0.5],
            DType::F64,
        )
        .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Round);
        // Banker's rounding: 0.5->0, 1.5->2, 2.5->2, 3.5->4, 4.5->4, -0.5->0
        assert_eq!(out.values(), &[0.0, 2.0, 2.0, 4.0, 4.0, 0.0]);
    }

    #[test]
    fn unary_reciprocal() {
        let arr =
            UFuncArray::new(vec![4], vec![1.0, 2.0, 4.0, 0.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Reciprocal);
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.5).abs() < 1e-10);
        assert!((out.values()[2] - 0.25).abs() < 1e-10);
        assert!((out.values()[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn unary_sinh() {
        let arr =
            UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Sinh);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0_f64.sinh()).abs() < 1e-10);
        assert!((out.values()[2] - (-1.0_f64).sinh()).abs() < 1e-10);
    }

    #[test]
    fn unary_cosh() {
        let arr =
            UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Cosh);
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0_f64.cosh()).abs() < 1e-10);
        assert!((out.values()[2] - 1.0_f64.cosh()).abs() < 1e-10); // cosh is even
    }

    #[test]
    fn unary_tanh() {
        let arr =
            UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Tanh);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0_f64.tanh()).abs() < 1e-10);
        assert!((out.values()[2] - (-1.0_f64).tanh()).abs() < 1e-10);
    }

    #[test]
    fn unary_arcsin() {
        let arr =
            UFuncArray::new(vec![3], vec![0.0, 0.5, 1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Arcsin);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.5_f64.asin()).abs() < 1e-10);
        assert!((out.values()[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn unary_arccos() {
        let arr =
            UFuncArray::new(vec![3], vec![1.0, 0.5, 0.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Arccos);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.5_f64.acos()).abs() < 1e-10);
        assert!((out.values()[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn unary_arctan() {
        let arr =
            UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Arctan);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        assert!((out.values()[2] - (-std::f64::consts::FRAC_PI_4)).abs() < 1e-10);
    }

    #[test]
    fn binary_arctan2() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, -1.0, -1.0, 1.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![1.0, 1.0, -1.0, -1.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Arctan2)
            .expect("arctan2");
        assert!((out.values()[0] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        assert!((out.values()[1] - (-std::f64::consts::FRAC_PI_4)).abs() < 1e-10);
    }

    #[test]
    fn packet005_log_record_is_replay_complete() {
        let record = UFuncLogRecord {
            fixture_id: "UP-005-elementwise-deterministic".to_string(),
            seed: 5001,
            mode: UFuncRuntimeMode::Strict,
            env_fingerprint: "fnp-ufunc-tests".to_string(),
            artifact_refs: packet005_artifacts(),
            reason_code: "ufunc_broadcast_selector_determinism".to_string(),
            passed: true,
        };
        assert!(record.is_replay_complete());
    }

    #[test]
    fn packet005_log_record_rejects_missing_required_fields() {
        let missing = UFuncLogRecord {
            fixture_id: String::new(),
            seed: 5002,
            mode: UFuncRuntimeMode::Hardened,
            env_fingerprint: String::new(),
            artifact_refs: Vec::new(),
            reason_code: String::new(),
            passed: false,
        };
        assert!(!missing.is_replay_complete());
    }

    #[test]
    fn packet005_reason_codes_round_trip_into_replay_logs() {
        for (idx, reason_code) in UFUNC_PACKET_REASON_CODES.iter().enumerate() {
            let record = UFuncLogRecord {
                fixture_id: format!("UP-005-{idx}"),
                seed: 5100 + u64::try_from(idx).expect("small index"),
                mode: UFuncRuntimeMode::Strict,
                env_fingerprint: "fnp-ufunc-tests".to_string(),
                artifact_refs: packet005_artifacts(),
                reason_code: (*reason_code).to_string(),
                passed: true,
            };
            assert!(record.is_replay_complete());
            assert_eq!(record.reason_code, *reason_code);
        }
    }
}
