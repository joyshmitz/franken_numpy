#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    C,
    F,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    InvalidItemSize,
    InvalidDimension(isize),
    MultipleUnknownDimensions,
    Overflow,
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidWindowDimension {
        axis: usize,
        window: usize,
        dim: usize,
    },
    NegativeStride(isize),
    OutOfBoundsView {
        required_nbytes: usize,
        available_nbytes: usize,
    },
    IncompatibleBroadcast {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    IncompatibleElementCount {
        old: usize,
        new: usize,
    },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidItemSize => write!(f, "item size must be > 0"),
            Self::InvalidDimension(dim) => write!(f, "invalid dimension {dim}"),
            Self::MultipleUnknownDimensions => write!(f, "only one -1 dimension is allowed"),
            Self::Overflow => write!(f, "size arithmetic overflow"),
            Self::RankMismatch { expected, actual } => {
                write!(f, "rank mismatch expected={expected} actual={actual}")
            }
            Self::InvalidWindowDimension { axis, window, dim } => write!(
                f,
                "invalid sliding window on axis={axis} window={window} dim={dim}"
            ),
            Self::NegativeStride(stride) => {
                write!(f, "negative strides are not supported stride={stride}")
            }
            Self::OutOfBoundsView {
                required_nbytes,
                available_nbytes,
            } => write!(
                f,
                "view exceeds base storage required_nbytes={required_nbytes} available_nbytes={available_nbytes}"
            ),
            Self::IncompatibleBroadcast { lhs, rhs } => {
                write!(f, "cannot broadcast {:?} with {:?}", lhs, rhs)
            }
            Self::IncompatibleElementCount { old, new } => {
                write!(f, "element count mismatch old={old} new={new}")
            }
        }
    }
}

impl std::error::Error for ShapeError {}

#[must_use]
pub fn can_broadcast(lhs: &[usize], rhs: &[usize]) -> bool {
    broadcast_shape(lhs, rhs).is_ok()
}

pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let nd = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(nd);

    for idx in 0..nd {
        let l = *lhs.get(lhs.len().wrapping_sub(1 + idx)).unwrap_or(&1);
        let r = *rhs.get(rhs.len().wrapping_sub(1 + idx)).unwrap_or(&1);

        let merged = if l == r {
            l
        } else if l == 1 {
            r
        } else if r == 1 {
            l
        } else {
            return Err(ShapeError::IncompatibleBroadcast {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
        };

        out.push(merged);
    }

    out.reverse();
    Ok(out)
}

pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
    let mut acc = Vec::new();
    for shape in shapes {
        acc = broadcast_shape(&acc, shape)?;
    }
    Ok(acc)
}

pub fn element_count(shape: &[usize]) -> Result<usize, ShapeError> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(ShapeError::Overflow)
    })
}

pub fn fix_unknown_dimension(
    new_shape: &[isize],
    old_element_count: usize,
) -> Result<Vec<usize>, ShapeError> {
    let mut known_product: usize = 1;
    let mut unknown_index: Option<usize> = None;
    let mut out = Vec::with_capacity(new_shape.len());

    for (idx, &dim) in new_shape.iter().enumerate() {
        match dim {
            -1 => {
                if unknown_index.replace(idx).is_some() {
                    return Err(ShapeError::MultipleUnknownDimensions);
                }
                out.push(0);
            }
            d if d < -1 => return Err(ShapeError::InvalidDimension(d)),
            d => {
                let d = usize::try_from(d).map_err(|_| ShapeError::InvalidDimension(d))?;
                known_product = known_product.checked_mul(d).ok_or(ShapeError::Overflow)?;
                out.push(d);
            }
        }
    }

    match unknown_index {
        Some(idx) => {
            if known_product == 0 || !old_element_count.is_multiple_of(known_product) {
                return Err(ShapeError::IncompatibleElementCount {
                    old: old_element_count,
                    new: known_product,
                });
            }
            out[idx] = old_element_count / known_product;
        }
        None => {
            if known_product != old_element_count {
                return Err(ShapeError::IncompatibleElementCount {
                    old: old_element_count,
                    new: known_product,
                });
            }
        }
    }

    Ok(out)
}

pub fn contiguous_strides(
    shape: &[usize],
    item_size: usize,
    order: MemoryOrder,
) -> Result<Vec<isize>, ShapeError> {
    if item_size == 0 {
        return Err(ShapeError::InvalidItemSize);
    }

    let mut strides_bytes = vec![0usize; shape.len()];

    match order {
        MemoryOrder::C => {
            let mut stride = item_size;
            for (i, &dim) in shape.iter().enumerate().rev() {
                strides_bytes[i] = stride;
                stride = stride.checked_mul(dim).ok_or(ShapeError::Overflow)?;
            }
        }
        MemoryOrder::F => {
            let mut stride = item_size;
            for (i, &dim) in shape.iter().enumerate() {
                strides_bytes[i] = stride;
                stride = stride.checked_mul(dim).ok_or(ShapeError::Overflow)?;
            }
        }
    }

    strides_bytes
        .into_iter()
        .map(|s| isize::try_from(s).map_err(|_| ShapeError::Overflow))
        .collect()
}

pub fn broadcast_strides(
    src_shape: &[usize],
    src_strides: &[isize],
    dst_shape: &[usize],
) -> Result<Vec<isize>, ShapeError> {
    if src_shape.len() != src_strides.len() {
        return Err(ShapeError::RankMismatch {
            expected: src_shape.len(),
            actual: src_strides.len(),
        });
    }

    let merged = broadcast_shape(src_shape, dst_shape)?;
    if merged != dst_shape {
        return Err(ShapeError::IncompatibleBroadcast {
            lhs: src_shape.to_vec(),
            rhs: dst_shape.to_vec(),
        });
    }

    if dst_shape.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = vec![0isize; dst_shape.len()];
    let offset = dst_shape.len().saturating_sub(src_shape.len());

    for (axis, (&src_dim, &src_stride)) in src_shape.iter().zip(src_strides).enumerate() {
        let out_axis = axis + offset;
        let dst_dim = dst_shape[out_axis];
        out[out_axis] = if src_dim == dst_dim {
            src_stride
        } else if src_dim == 1 {
            0
        } else {
            return Err(ShapeError::IncompatibleBroadcast {
                lhs: src_shape.to_vec(),
                rhs: dst_shape.to_vec(),
            });
        };
    }

    Ok(out)
}

fn required_view_nbytes(
    shape: &[usize],
    strides: &[isize],
    item_size: usize,
) -> Result<usize, ShapeError> {
    if item_size == 0 {
        return Err(ShapeError::InvalidItemSize);
    }
    if shape.len() != strides.len() {
        return Err(ShapeError::RankMismatch {
            expected: shape.len(),
            actual: strides.len(),
        });
    }
    if shape.contains(&0) {
        return Ok(0);
    }

    let mut max_offset = 0usize;
    for (&dim, &stride) in shape.iter().zip(strides) {
        if stride < 0 {
            return Err(ShapeError::NegativeStride(stride));
        }
        if dim <= 1 {
            continue;
        }
        let step = usize::try_from(stride).map_err(|_| ShapeError::NegativeStride(stride))?;
        let axis_span = step.checked_mul(dim - 1).ok_or(ShapeError::Overflow)?;
        max_offset = max_offset
            .checked_add(axis_span)
            .ok_or(ShapeError::Overflow)?;
    }

    max_offset
        .checked_add(item_size)
        .ok_or(ShapeError::Overflow)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdLayout {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub item_size: usize,
}

impl NdLayout {
    pub fn contiguous(
        shape: Vec<usize>,
        item_size: usize,
        order: MemoryOrder,
    ) -> Result<Self, ShapeError> {
        let strides = contiguous_strides(&shape, item_size, order)?;
        Ok(Self {
            shape,
            strides,
            item_size,
        })
    }

    pub fn nbytes(&self) -> Result<usize, ShapeError> {
        element_count(&self.shape)?
            .checked_mul(self.item_size)
            .ok_or(ShapeError::Overflow)
    }

    pub fn as_strided(&self, shape: Vec<usize>, strides: Vec<isize>) -> Result<Self, ShapeError> {
        let required_nbytes = required_view_nbytes(&shape, &strides, self.item_size)?;
        let available_nbytes = self.nbytes()?;
        if required_nbytes > available_nbytes {
            return Err(ShapeError::OutOfBoundsView {
                required_nbytes,
                available_nbytes,
            });
        }
        Ok(Self {
            shape,
            strides,
            item_size: self.item_size,
        })
    }

    pub fn broadcast_to(&self, shape: Vec<usize>) -> Result<Self, ShapeError> {
        let strides = broadcast_strides(&self.shape, &self.strides, &shape)?;
        Ok(Self {
            shape,
            strides,
            item_size: self.item_size,
        })
    }

    pub fn sliding_window_view(&self, window_shape: Vec<usize>) -> Result<Self, ShapeError> {
        if window_shape.len() != self.shape.len() {
            return Err(ShapeError::RankMismatch {
                expected: self.shape.len(),
                actual: window_shape.len(),
            });
        }

        let mut shape = Vec::with_capacity(self.shape.len() * 2);
        for (axis, (&dim, &window)) in self.shape.iter().zip(&window_shape).enumerate() {
            if window == 0 || window > dim {
                return Err(ShapeError::InvalidWindowDimension { axis, window, dim });
            }
            shape.push(dim - window + 1);
        }
        shape.extend(window_shape.iter().copied());

        let mut strides = Vec::with_capacity(self.strides.len() * 2);
        strides.extend(self.strides.iter().copied());
        strides.extend(self.strides.iter().copied());

        Ok(Self {
            shape,
            strides,
            item_size: self.item_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MemoryOrder, NdLayout, ShapeError, broadcast_shape, broadcast_shapes, broadcast_strides,
        can_broadcast, contiguous_strides, element_count, fix_unknown_dimension,
    };

    #[test]
    fn broadcast_shape_matches_numpy_style() {
        let out = broadcast_shape(&[8, 1, 6, 1], &[7, 1, 5]).expect("broadcast should succeed");
        assert_eq!(out, vec![8, 7, 6, 5]);
    }

    #[test]
    fn broadcast_shape_rejects_incompatible_shapes() {
        let err = broadcast_shape(&[4, 3], &[5, 3]).expect_err("should fail");
        assert!(matches!(err, ShapeError::IncompatibleBroadcast { .. }));
        assert!(!can_broadcast(&[4, 3], &[5, 3]));
    }

    #[test]
    fn broadcast_many_shapes() {
        let shapes: [&[usize]; 3] = [&[3, 1], &[1, 7], &[5, 3, 7]];
        let out = broadcast_shapes(&shapes).expect("broadcast should succeed");
        assert_eq!(out, vec![5, 3, 7]);
    }

    #[test]
    fn contiguous_strides_c_and_f_orders() {
        let c = contiguous_strides(&[2, 3, 4], 8, MemoryOrder::C).expect("c-order");
        let f = contiguous_strides(&[2, 3, 4], 8, MemoryOrder::F).expect("f-order");
        assert_eq!(c, vec![96, 32, 8]);
        assert_eq!(f, vec![8, 16, 48]);
    }

    #[test]
    fn unknown_dimension_resolution() {
        let resolved = fix_unknown_dimension(&[2, -1, 3], 24).expect("should infer -1 dimension");
        assert_eq!(resolved, vec![2, 4, 3]);
    }

    #[test]
    fn unknown_dimension_validation() {
        let err = fix_unknown_dimension(&[-1, -1], 8).expect_err("only one unknown allowed");
        assert!(matches!(err, ShapeError::MultipleUnknownDimensions));
    }

    #[test]
    fn layout_nbytes_is_consistent() {
        let layout = NdLayout::contiguous(vec![5, 6], 4, MemoryOrder::C).expect("layout");
        assert_eq!(layout.nbytes().expect("nbytes"), 120);
        assert_eq!(element_count(&layout.shape).expect("elements"), 30);
    }

    #[test]
    fn broadcast_shape_is_commutative_for_sample_shapes() {
        let shapes: &[&[usize]] = &[
            &[],
            &[1],
            &[3],
            &[1, 4],
            &[2, 1, 4],
            &[2, 3, 4],
            &[1, 1, 5, 1],
        ];

        for lhs in shapes {
            for rhs in shapes {
                let ab = broadcast_shape(lhs, rhs);
                let ba = broadcast_shape(rhs, lhs);
                match (ab, ba) {
                    (Ok(left), Ok(right)) => assert_eq!(left, right),
                    (Err(_), Err(_)) => {}
                    (left, right) => assert_eq!(
                        left.is_ok(),
                        right.is_ok(),
                        "broadcast commutativity mismatch: {left:?} vs {right:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn element_count_scalar_shape_is_one() {
        assert_eq!(element_count(&[]).expect("scalar count"), 1);
    }

    #[test]
    fn element_count_overflow_is_rejected() {
        let err = element_count(&[usize::MAX, 2]).expect_err("overflow should fail");
        assert!(matches!(err, ShapeError::Overflow));
    }

    #[test]
    fn contiguous_strides_rejects_zero_item_size() {
        let err = contiguous_strides(&[2, 3], 0, MemoryOrder::C).expect_err("must reject");
        assert!(matches!(err, ShapeError::InvalidItemSize));
    }

    #[test]
    fn invalid_negative_dimension_is_rejected() {
        let err = fix_unknown_dimension(&[2, -2], 2).expect_err("negative dimension should fail");
        assert!(matches!(err, ShapeError::InvalidDimension(-2)));
    }

    #[test]
    fn zero_known_product_with_unknown_dimension_is_rejected() {
        let err = fix_unknown_dimension(&[0, -1], 0).expect_err("zero known product is invalid");
        assert!(matches!(
            err,
            ShapeError::IncompatibleElementCount { old: 0, new: 0 }
        ));
    }

    #[test]
    fn layout_nbytes_overflow_is_rejected() {
        let layout = NdLayout {
            shape: vec![usize::MAX],
            strides: vec![1],
            item_size: 2,
        };
        let err = layout.nbytes().expect_err("nbytes overflow should fail");
        assert!(matches!(err, ShapeError::Overflow));
    }

    #[test]
    fn broadcast_strides_zeroes_broadcast_axes() {
        let base = NdLayout::contiguous(vec![2, 1, 3], 8, MemoryOrder::C).expect("layout");
        let out = broadcast_strides(&base.shape, &base.strides, &[2, 4, 3]).expect("strides");
        assert_eq!(out, vec![24, 0, 8]);
    }

    #[test]
    fn broadcast_strides_rejects_rank_mismatch() {
        let err = broadcast_strides(&[2, 3], &[24], &[2, 3])
            .expect_err("shape/stride rank mismatch should fail");
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn broadcast_strides_rejects_incompatible_target() {
        let err = broadcast_strides(&[2, 3], &[24, 8], &[2, 2])
            .expect_err("incompatible target shape should fail");
        assert!(matches!(err, ShapeError::IncompatibleBroadcast { .. }));
    }

    #[test]
    fn broadcast_to_produces_zero_stride_view() {
        let base = NdLayout::contiguous(vec![2, 1, 3], 8, MemoryOrder::C).expect("layout");
        let out = base.broadcast_to(vec![2, 4, 3]).expect("broadcast_to");
        assert_eq!(out.shape, vec![2, 4, 3]);
        assert_eq!(out.strides, vec![24, 0, 8]);
    }

    #[test]
    fn broadcast_to_rejects_incompatible_target() {
        let base = NdLayout::contiguous(vec![2, 3], 8, MemoryOrder::C).expect("layout");
        let err = base
            .broadcast_to(vec![2, 2])
            .expect_err("incompatible broadcast should fail");
        assert!(matches!(err, ShapeError::IncompatibleBroadcast { .. }));
    }

    #[test]
    fn as_strided_accepts_in_bounds_view() {
        let base = NdLayout::contiguous(vec![4, 4], 8, MemoryOrder::C).expect("layout");
        let view = base
            .as_strided(vec![2, 2], vec![16, 8])
            .expect("valid strided view");
        assert_eq!(view.shape, vec![2, 2]);
        assert_eq!(view.strides, vec![16, 8]);
    }

    #[test]
    fn as_strided_allows_zero_extent_view() {
        let base = NdLayout::contiguous(vec![4], 8, MemoryOrder::C).expect("layout");
        let view = base
            .as_strided(vec![0], vec![8])
            .expect("zero-extent views should be in-bounds");
        assert_eq!(view.shape, vec![0]);
        assert_eq!(view.strides, vec![8]);
    }

    #[test]
    fn as_strided_rejects_out_of_bounds_view() {
        let base = NdLayout::contiguous(vec![4, 4], 8, MemoryOrder::C).expect("layout");
        let err = base
            .as_strided(vec![4, 4], vec![64, 8])
            .expect_err("view must fit in base storage");
        assert!(matches!(err, ShapeError::OutOfBoundsView { .. }));
    }

    #[test]
    fn as_strided_rejects_negative_stride() {
        let base = NdLayout::contiguous(vec![4], 8, MemoryOrder::C).expect("layout");
        let err = base
            .as_strided(vec![4], vec![-8])
            .expect_err("negative strides are unsupported");
        assert!(matches!(err, ShapeError::NegativeStride(-8)));
    }

    #[test]
    fn as_strided_rejects_rank_mismatch() {
        let base = NdLayout::contiguous(vec![4], 8, MemoryOrder::C).expect("layout");
        let err = base
            .as_strided(vec![2, 2], vec![8])
            .expect_err("shape/stride rank mismatch should fail");
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn sliding_window_view_builds_expected_shape_and_strides() {
        let base = NdLayout::contiguous(vec![4, 5], 8, MemoryOrder::C).expect("layout");
        let view = base
            .sliding_window_view(vec![2, 3])
            .expect("valid sliding window");
        assert_eq!(view.shape, vec![3, 3, 2, 3]);
        assert_eq!(view.strides, vec![40, 8, 40, 8]);
    }

    #[test]
    fn sliding_window_view_rejects_invalid_window_size() {
        let base = NdLayout::contiguous(vec![4, 5], 8, MemoryOrder::C).expect("layout");
        let err = base
            .sliding_window_view(vec![0, 3])
            .expect_err("window size zero should fail");
        assert!(matches!(
            err,
            ShapeError::InvalidWindowDimension {
                axis: 0,
                window: 0,
                dim: 4
            }
        ));
    }

    #[test]
    fn sliding_window_view_rejects_rank_mismatch() {
        let base = NdLayout::contiguous(vec![4, 5], 8, MemoryOrder::C).expect("layout");
        let err = base
            .sliding_window_view(vec![2])
            .expect_err("window rank must match input rank");
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }
}
