#![forbid(unsafe_code)]

/// Canonical subset of NumPy-like dtypes for the first conformance wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I32,
    I64,
    F32,
    F64,
}

impl DType {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    #[must_use]
    pub const fn item_size(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "bool" => Some(Self::Bool),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }
}

/// Deterministic promotion table for V1-scoped dtypes.
#[must_use]
pub const fn promote(lhs: DType, rhs: DType) -> DType {
    use DType::{Bool, F32, F64, I32, I64};

    match (lhs, rhs) {
        (Bool, x) | (x, Bool) => x,
        (I32, I32) => I32,
        (I32, I64) | (I64, I32) | (I64, I64) => I64,
        (F32, F32) => F32,
        (F64, F64) => F64,
        (F32, I32) | (I32, F32) => F64,
        (F32, I64) | (I64, F32) => F64,
        (F64, _) | (_, F64) => F64,
    }
}

#[must_use]
pub const fn can_cast_lossless(src: DType, dst: DType) -> bool {
    use DType::{Bool, F32, F64, I32, I64};

    matches!(
        (src, dst),
        (Bool, Bool)
            | (Bool, I32 | I64 | F32 | F64)
            | (I32, I32 | I64 | F64)
            | (I64, I64 | F64)
            | (F32, F32 | F64)
            | (F64, F64)
    )
}

#[cfg(test)]
mod tests {
    use super::{DType, can_cast_lossless, promote};

    #[test]
    fn promotion_is_commutative() {
        let dtypes = [DType::Bool, DType::I32, DType::I64, DType::F32, DType::F64];

        for &lhs in &dtypes {
            for &rhs in &dtypes {
                assert_eq!(promote(lhs, rhs), promote(rhs, lhs), "{lhs:?}/{rhs:?}");
            }
        }
    }

    #[test]
    fn promotion_expectations_hold() {
        assert_eq!(promote(DType::Bool, DType::I32), DType::I32);
        assert_eq!(promote(DType::I32, DType::I64), DType::I64);
        assert_eq!(promote(DType::I32, DType::F32), DType::F64);
        assert_eq!(promote(DType::F32, DType::F64), DType::F64);
    }

    #[test]
    fn cast_matrix_smoke() {
        assert!(can_cast_lossless(DType::Bool, DType::F64));
        assert!(can_cast_lossless(DType::I32, DType::I64));
        assert!(!can_cast_lossless(DType::I64, DType::I32));
        assert!(!can_cast_lossless(DType::F64, DType::F32));
    }
}
