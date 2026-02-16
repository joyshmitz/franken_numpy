#![forbid(unsafe_code)]

use core::fmt;
use std::collections::BTreeSet;

pub const IO_PACKET_ID: &str = "FNP-P2C-009";
pub const NPY_MAGIC_PREFIX: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];
pub const NPZ_MAGIC_PREFIX: [u8; 4] = [b'P', b'K', 0x03, 0x04];

pub const MAX_HEADER_BYTES: usize = 65_536;
pub const MAX_ARCHIVE_MEMBERS: usize = 4_096;
pub const MAX_ARCHIVE_UNCOMPRESSED_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const MAX_DISPATCH_RETRIES: usize = 8;
pub const MAX_MEMMAP_VALIDATION_RETRIES: usize = 64;

pub const IO_PACKET_REASON_CODES: [&str; 10] = [
    "io_magic_invalid",
    "io_header_schema_invalid",
    "io_dtype_descriptor_invalid",
    "io_write_contract_violation",
    "io_read_payload_incomplete",
    "io_pickle_policy_violation",
    "io_memmap_contract_violation",
    "io_load_dispatch_invalid",
    "io_npz_archive_contract_violation",
    "io_policy_unknown_metadata",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IORuntimeMode {
    Strict,
    Hardened,
}

impl IORuntimeMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hardened => "hardened",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOSupportedDType {
    Bool,
    I32,
    I64,
    F32,
    F64,
    Object,
}

impl IOSupportedDType {
    #[must_use]
    pub const fn descr(self) -> &'static str {
        match self {
            Self::Bool => "|b1",
            Self::I32 => "<i4",
            Self::I64 => "<i8",
            Self::F32 => "<f4",
            Self::F64 => "<f8",
            Self::Object => "|O",
        }
    }

    pub fn decode(descr: &str) -> Result<Self, IOError> {
        match descr {
            "|b1" => Ok(Self::Bool),
            "<i4" => Ok(Self::I32),
            "<i8" => Ok(Self::I64),
            "<f4" => Ok(Self::F32),
            "<f8" => Ok(Self::F64),
            "|O" => Ok(Self::Object),
            _ => Err(IOError::DTypeDescriptorInvalid),
        }
    }

    #[must_use]
    pub const fn item_size(self) -> Option<usize> {
        match self {
            Self::Bool => Some(1),
            Self::I32 | Self::F32 => Some(4),
            Self::I64 | Self::F64 => Some(8),
            Self::Object => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemmapMode {
    ReadOnly,
    ReadWrite,
    Write,
    CopyOnWrite,
}

impl MemmapMode {
    pub fn parse(token: &str) -> Result<Self, IOError> {
        match token {
            "r" => Ok(Self::ReadOnly),
            "r+" => Ok(Self::ReadWrite),
            "w+" => Ok(Self::Write),
            "c" => Ok(Self::CopyOnWrite),
            _ => Err(IOError::MemmapContractViolation(
                "invalid memmap mode token",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadDispatch {
    Npy,
    Npz,
    Pickle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IOError {
    MagicInvalid,
    HeaderSchemaInvalid(&'static str),
    DTypeDescriptorInvalid,
    WriteContractViolation(&'static str),
    ReadPayloadIncomplete(&'static str),
    PicklePolicyViolation,
    MemmapContractViolation(&'static str),
    LoadDispatchInvalid(&'static str),
    NpzArchiveContractViolation(&'static str),
    PolicyUnknownMetadata(&'static str),
}

impl IOError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::MagicInvalid => "io_magic_invalid",
            Self::HeaderSchemaInvalid(_) => "io_header_schema_invalid",
            Self::DTypeDescriptorInvalid => "io_dtype_descriptor_invalid",
            Self::WriteContractViolation(_) => "io_write_contract_violation",
            Self::ReadPayloadIncomplete(_) => "io_read_payload_incomplete",
            Self::PicklePolicyViolation => "io_pickle_policy_violation",
            Self::MemmapContractViolation(_) => "io_memmap_contract_violation",
            Self::LoadDispatchInvalid(_) => "io_load_dispatch_invalid",
            Self::NpzArchiveContractViolation(_) => "io_npz_archive_contract_violation",
            Self::PolicyUnknownMetadata(_) => "io_policy_unknown_metadata",
        }
    }
}

impl fmt::Display for IOError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MagicInvalid => write!(f, "invalid or unsupported npy/npz magic/version"),
            Self::HeaderSchemaInvalid(msg) => write!(f, "{msg}"),
            Self::DTypeDescriptorInvalid => write!(f, "dtype descriptor is invalid or unsupported"),
            Self::WriteContractViolation(msg) => write!(f, "{msg}"),
            Self::ReadPayloadIncomplete(msg) => write!(f, "{msg}"),
            Self::PicklePolicyViolation => write!(f, "pickle/object payload rejected by policy"),
            Self::MemmapContractViolation(msg) => write!(f, "{msg}"),
            Self::LoadDispatchInvalid(msg) => write!(f, "{msg}"),
            Self::NpzArchiveContractViolation(msg) => write!(f, "{msg}"),
            Self::PolicyUnknownMetadata(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for IOError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyHeader {
    pub shape: Vec<usize>,
    pub fortran_order: bool,
    pub descr: IOSupportedDType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IOLogRecord {
    pub ts_utc: String,
    pub suite_id: String,
    pub test_id: String,
    pub packet_id: String,
    pub fixture_id: String,
    pub mode: IORuntimeMode,
    pub seed: u64,
    pub input_digest: String,
    pub output_digest: String,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub duration_ms: u64,
    pub outcome: String,
    pub reason_code: String,
}

impl IOLogRecord {
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

        if self.packet_id != IO_PACKET_ID {
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

        IO_PACKET_REASON_CODES
            .iter()
            .any(|code| *code == self.reason_code)
    }
}

fn element_count(shape: &[usize]) -> Result<usize, IOError> {
    shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
        .ok_or(IOError::HeaderSchemaInvalid(
            "shape element-count overflowed",
        ))
}

pub fn validate_magic_version(payload: &[u8]) -> Result<(u8, u8), IOError> {
    if payload.len() < 8 {
        return Err(IOError::MagicInvalid);
    }
    if payload[..6] != NPY_MAGIC_PREFIX {
        return Err(IOError::MagicInvalid);
    }

    let version = (payload[6], payload[7]);
    if version == (1, 0) || version == (2, 0) || version == (3, 0) {
        Ok(version)
    } else {
        Err(IOError::MagicInvalid)
    }
}

pub fn validate_header_schema(
    shape: &[usize],
    fortran_order: bool,
    descr: &str,
    header_len: usize,
) -> Result<NpyHeader, IOError> {
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(IOError::HeaderSchemaInvalid(
            "header bytes must be within bounded budget",
        ));
    }
    if shape.len() > 32 {
        return Err(IOError::HeaderSchemaInvalid(
            "shape rank exceeds packet validation budget",
        ));
    }

    let _ = fortran_order;
    let _ = element_count(shape)?;
    let descr = IOSupportedDType::decode(descr)?;

    Ok(NpyHeader {
        shape: shape.to_vec(),
        fortran_order,
        descr,
    })
}

pub fn validate_descriptor_roundtrip(dtype: IOSupportedDType) -> Result<(), IOError> {
    let encoded = dtype.descr();
    let decoded = IOSupportedDType::decode(encoded)?;
    if decoded == dtype {
        Ok(())
    } else {
        Err(IOError::DTypeDescriptorInvalid)
    }
}

pub fn validate_write_contract(
    shape: &[usize],
    value_count: usize,
    dtype: IOSupportedDType,
) -> Result<usize, IOError> {
    let expected_count = element_count(shape).map_err(|_| {
        IOError::WriteContractViolation("failed to compute element count for write path")
    })?;
    if value_count != expected_count {
        return Err(IOError::WriteContractViolation(
            "value_count does not match shape element count",
        ));
    }

    let item_size = dtype.item_size().ok_or(IOError::WriteContractViolation(
        "object dtype requires explicit pickle/object policy path",
    ))?;

    expected_count
        .checked_mul(item_size)
        .ok_or(IOError::WriteContractViolation(
            "write byte count overflowed",
        ))
}

pub fn validate_read_payload(
    shape: &[usize],
    payload_len_bytes: usize,
    dtype: IOSupportedDType,
) -> Result<usize, IOError> {
    let item_size = dtype.item_size().ok_or(IOError::ReadPayloadIncomplete(
        "object dtype payload requires pickle/object decode path",
    ))?;
    let expected_count = element_count(shape)
        .map_err(|_| IOError::ReadPayloadIncomplete("failed to compute expected element count"))?;
    let expected_bytes =
        expected_count
            .checked_mul(item_size)
            .ok_or(IOError::ReadPayloadIncomplete(
                "expected payload bytes overflowed",
            ))?;

    if payload_len_bytes != expected_bytes {
        return Err(IOError::ReadPayloadIncomplete(
            "payload bytes must exactly match expected shape/dtype footprint",
        ));
    }

    Ok(expected_count)
}

pub fn enforce_pickle_policy(dtype: IOSupportedDType, allow_pickle: bool) -> Result<(), IOError> {
    if dtype == IOSupportedDType::Object && !allow_pickle {
        return Err(IOError::PicklePolicyViolation);
    }
    Ok(())
}

pub fn validate_memmap_contract(
    mode: MemmapMode,
    dtype: IOSupportedDType,
    file_len_bytes: usize,
    expected_bytes: usize,
    validation_retries: usize,
) -> Result<(), IOError> {
    if validation_retries > MAX_MEMMAP_VALIDATION_RETRIES {
        return Err(IOError::MemmapContractViolation(
            "memmap validation retries exceeded bounded budget",
        ));
    }
    if dtype == IOSupportedDType::Object {
        return Err(IOError::MemmapContractViolation(
            "object dtype is invalid for memmap path",
        ));
    }
    if file_len_bytes < expected_bytes {
        return Err(IOError::MemmapContractViolation(
            "backing file is too small for requested mapping",
        ));
    }
    if mode == MemmapMode::Write && expected_bytes == 0 {
        return Err(IOError::MemmapContractViolation(
            "write memmap requires non-empty expected byte footprint",
        ));
    }
    Ok(())
}

pub fn classify_load_dispatch(
    payload_prefix: &[u8],
    allow_pickle: bool,
) -> Result<LoadDispatch, IOError> {
    if payload_prefix.len() >= 4 && payload_prefix[..4] == NPZ_MAGIC_PREFIX {
        return Ok(LoadDispatch::Npz);
    }

    if payload_prefix.len() >= 6 && payload_prefix[..6] == NPY_MAGIC_PREFIX {
        return Ok(LoadDispatch::Npy);
    }

    if allow_pickle && payload_prefix.first().copied() == Some(0x80) {
        return Ok(LoadDispatch::Pickle);
    }

    Err(IOError::LoadDispatchInvalid(
        "payload prefix does not map to allowed npy/npz/pickle branch",
    ))
}

pub fn synthesize_npz_member_names(
    positional_count: usize,
    keyword_names: &[&str],
) -> Result<Vec<String>, IOError> {
    let member_count = positional_count.checked_add(keyword_names.len()).ok_or(
        IOError::NpzArchiveContractViolation("archive member count overflowed"),
    )?;
    if member_count == 0 || member_count > MAX_ARCHIVE_MEMBERS {
        return Err(IOError::NpzArchiveContractViolation(
            "archive member count is outside bounded limits",
        ));
    }

    let mut names = Vec::with_capacity(member_count);
    let mut seen = BTreeSet::new();

    for idx in 0..positional_count {
        let name = format!("arr_{idx}");
        let _ = seen.insert(name.clone());
        names.push(name);
    }

    for &name in keyword_names {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(IOError::NpzArchiveContractViolation(
                "keyword member name cannot be empty",
            ));
        }
        if !seen.insert(trimmed.to_string()) {
            return Err(IOError::NpzArchiveContractViolation(
                "archive member names must be unique",
            ));
        }
        names.push(trimmed.to_string());
    }

    Ok(names)
}

pub fn validate_npz_archive_budget(
    member_count: usize,
    uncompressed_bytes: usize,
    dispatch_retries: usize,
) -> Result<(), IOError> {
    if member_count == 0 || member_count > MAX_ARCHIVE_MEMBERS {
        return Err(IOError::NpzArchiveContractViolation(
            "archive member count is outside bounded limits",
        ));
    }
    if uncompressed_bytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES {
        return Err(IOError::NpzArchiveContractViolation(
            "archive decoded size exceeded bounded budget",
        ));
    }
    if dispatch_retries > MAX_DISPATCH_RETRIES {
        return Err(IOError::LoadDispatchInvalid(
            "dispatch retries exceeded bounded budget",
        ));
    }
    Ok(())
}

pub fn validate_io_policy_metadata(mode: &str, class: &str) -> Result<(), IOError> {
    let known_mode = mode == "strict" || mode == "hardened";
    let known_class = class == "known_compatible_low_risk"
        || class == "known_compatible_high_risk"
        || class == "known_incompatible_semantics"
        || class == "unknown_semantics";

    if !known_mode || !known_class {
        return Err(IOError::PolicyUnknownMetadata(
            "unknown mode/class metadata rejected fail-closed",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        IO_PACKET_ID, IO_PACKET_REASON_CODES, IOError, IOLogRecord, IORuntimeMode,
        IOSupportedDType, LoadDispatch, MAX_ARCHIVE_MEMBERS, MAX_DISPATCH_RETRIES,
        MAX_HEADER_BYTES, MAX_MEMMAP_VALIDATION_RETRIES, MemmapMode, NPY_MAGIC_PREFIX,
        NPZ_MAGIC_PREFIX, classify_load_dispatch, enforce_pickle_policy,
        synthesize_npz_member_names, validate_descriptor_roundtrip, validate_header_schema,
        validate_io_policy_metadata, validate_magic_version, validate_memmap_contract,
        validate_npz_archive_budget, validate_read_payload, validate_write_contract,
    };

    fn packet009_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-009/contract_table.md".to_string(),
            "artifacts/phase2c/FNP-P2C-009/unit_property_evidence.json".to_string(),
        ]
    }

    #[test]
    fn reason_code_registry_matches_packet_contract() {
        assert_eq!(
            IO_PACKET_REASON_CODES,
            [
                "io_magic_invalid",
                "io_header_schema_invalid",
                "io_dtype_descriptor_invalid",
                "io_write_contract_violation",
                "io_read_payload_incomplete",
                "io_pickle_policy_violation",
                "io_memmap_contract_violation",
                "io_load_dispatch_invalid",
                "io_npz_archive_contract_violation",
                "io_policy_unknown_metadata",
            ]
        );
    }

    #[test]
    fn magic_version_accepts_supported_tuples() {
        let mut payload = [0u8; 8];
        payload[..6].copy_from_slice(&NPY_MAGIC_PREFIX);

        for version in [(1, 0), (2, 0), (3, 0)] {
            payload[6] = version.0;
            payload[7] = version.1;
            assert_eq!(
                validate_magic_version(&payload).expect("supported tuple"),
                version
            );
        }
    }

    #[test]
    fn magic_version_rejects_corrupt_prefix_and_unknown_tuple() {
        let err = validate_magic_version(&[0u8; 4]).expect_err("short payload");
        assert_eq!(err.reason_code(), "io_magic_invalid");

        let mut payload = [0u8; 8];
        payload[..6].copy_from_slice(&NPY_MAGIC_PREFIX);
        payload[6] = 9;
        payload[7] = 9;
        let err = validate_magic_version(&payload).expect_err("unsupported tuple");
        assert_eq!(err.reason_code(), "io_magic_invalid");
    }

    #[test]
    fn header_schema_accepts_valid_and_rejects_invalid_budget() {
        let header = validate_header_schema(&[2, 3], false, "<f8", 128).expect("valid header");
        assert_eq!(header.shape, vec![2, 3]);
        assert_eq!(header.descr, IOSupportedDType::F64);

        let err = validate_header_schema(&[2, 3], true, "<f8", MAX_HEADER_BYTES + 1)
            .expect_err("oversized header");
        assert_eq!(err.reason_code(), "io_header_schema_invalid");
    }

    #[test]
    fn descriptor_roundtrip_covers_all_supported_dtypes() {
        let dtypes = [
            IOSupportedDType::Bool,
            IOSupportedDType::I32,
            IOSupportedDType::I64,
            IOSupportedDType::F32,
            IOSupportedDType::F64,
            IOSupportedDType::Object,
        ];

        for dtype in dtypes {
            validate_descriptor_roundtrip(dtype).expect("descriptor roundtrip");
        }

        let err = IOSupportedDType::decode(">i4").expect_err("unsupported descriptor");
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");
    }

    #[test]
    fn write_contract_property_grid_is_deterministic() {
        for seed in 1usize..=128usize {
            let rows = (seed % 17) + 1;
            let cols = (seed % 11) + 1;
            let shape = [rows, cols];
            let value_count = rows * cols;
            let bytes_first =
                validate_write_contract(&shape, value_count, IOSupportedDType::F64).expect("write");
            let bytes_second =
                validate_write_contract(&shape, value_count, IOSupportedDType::F64).expect("write");
            assert_eq!(bytes_first, bytes_second);
            assert_eq!(bytes_first, value_count * 8);
        }
    }

    #[test]
    fn write_contract_rejects_count_mismatch() {
        let err = validate_write_contract(&[3, 3], 8, IOSupportedDType::F64)
            .expect_err("shape/count mismatch");
        assert_eq!(err.reason_code(), "io_write_contract_violation");
    }

    #[test]
    fn read_payload_requires_exact_shape_footprint() {
        let count = validate_read_payload(&[2, 3], 6 * 8, IOSupportedDType::F64).expect("valid");
        assert_eq!(count, 6);

        let short = validate_read_payload(&[2, 3], 5 * 8, IOSupportedDType::F64)
            .expect_err("truncated payload");
        assert_eq!(short.reason_code(), "io_read_payload_incomplete");

        let long = validate_read_payload(&[2, 3], 7 * 8, IOSupportedDType::F64)
            .expect_err("extra trailing bytes");
        assert_eq!(long.reason_code(), "io_read_payload_incomplete");
    }

    #[test]
    fn pickle_policy_gate_rejects_object_when_disallowed() {
        let err = enforce_pickle_policy(IOSupportedDType::Object, false)
            .expect_err("object payload must be rejected");
        assert_eq!(err.reason_code(), "io_pickle_policy_violation");
        enforce_pickle_policy(IOSupportedDType::Object, true).expect("explicit allow_pickle");
    }

    #[test]
    fn memmap_contract_enforces_dtype_mode_and_retry_budget() {
        validate_memmap_contract(MemmapMode::ReadOnly, IOSupportedDType::F64, 4096, 1024, 0)
            .expect("valid memmap");
        let parsed = MemmapMode::parse("r+").expect("valid mode parse");
        assert_eq!(parsed, MemmapMode::ReadWrite);

        let object_err = validate_memmap_contract(
            MemmapMode::ReadOnly,
            IOSupportedDType::Object,
            4096,
            1024,
            0,
        )
        .expect_err("object memmap is invalid");
        assert_eq!(object_err.reason_code(), "io_memmap_contract_violation");

        let retry_err = validate_memmap_contract(
            MemmapMode::ReadOnly,
            IOSupportedDType::F64,
            4096,
            1024,
            MAX_MEMMAP_VALIDATION_RETRIES + 1,
        )
        .expect_err("retry budget exceeded");
        assert_eq!(retry_err.reason_code(), "io_memmap_contract_violation");
    }

    #[test]
    fn load_dispatch_selects_expected_branches() {
        let npz = classify_load_dispatch(&NPZ_MAGIC_PREFIX, false).expect("npz branch");
        assert_eq!(npz, LoadDispatch::Npz);

        let npy = classify_load_dispatch(&NPY_MAGIC_PREFIX, false).expect("npy branch");
        assert_eq!(npy, LoadDispatch::Npy);

        let pickle = classify_load_dispatch(&[0x80, 0x05, 0x95], true).expect("pickle branch");
        assert_eq!(pickle, LoadDispatch::Pickle);

        let err = classify_load_dispatch(&[0x80, 0x05, 0x95], false).expect_err("policy reject");
        assert_eq!(err.reason_code(), "io_load_dispatch_invalid");
    }

    #[test]
    fn npz_member_name_contract_enforces_uniqueness_and_budget() {
        let names =
            synthesize_npz_member_names(2, &["weights", "bias"]).expect("valid member names");
        assert_eq!(
            names,
            vec![
                "arr_0".to_string(),
                "arr_1".to_string(),
                "weights".to_string(),
                "bias".to_string()
            ]
        );

        let duplicate = synthesize_npz_member_names(1, &["arr_0"]).expect_err("duplicate name");
        assert_eq!(duplicate.reason_code(), "io_npz_archive_contract_violation");

        let too_many = synthesize_npz_member_names(MAX_ARCHIVE_MEMBERS + 1, &[])
            .expect_err("member budget exceeded");
        assert_eq!(too_many.reason_code(), "io_npz_archive_contract_violation");
    }

    #[test]
    fn npz_archive_budget_enforces_limits() {
        validate_npz_archive_budget(4, 1024, MAX_DISPATCH_RETRIES).expect("budget within limits");

        let huge = validate_npz_archive_budget(4, usize::MAX, 0).expect_err("decoded size too big");
        assert_eq!(huge.reason_code(), "io_npz_archive_contract_violation");

        let retries =
            validate_npz_archive_budget(4, 1024, MAX_DISPATCH_RETRIES + 1).expect_err("retries");
        assert_eq!(retries.reason_code(), "io_load_dispatch_invalid");
    }

    #[test]
    fn policy_metadata_is_fail_closed_for_unknowns() {
        validate_io_policy_metadata("strict", "known_compatible_low_risk").expect("known strict");
        validate_io_policy_metadata("hardened", "unknown_semantics").expect("known hardened");

        let err = validate_io_policy_metadata("mystery", "known_compatible_low_risk")
            .expect_err("unknown mode");
        assert_eq!(err.reason_code(), "io_policy_unknown_metadata");
    }

    #[test]
    fn packet009_log_record_is_replay_complete() {
        let record = IOLogRecord {
            ts_utc: "2026-02-16T00:00:00Z".to_string(),
            suite_id: "fnp-io::tests".to_string(),
            test_id: "UP-009-header-schema".to_string(),
            packet_id: IO_PACKET_ID.to_string(),
            fixture_id: "UP-009-header-schema".to_string(),
            mode: IORuntimeMode::Strict,
            seed: 9009,
            input_digest: "sha256:input".to_string(),
            output_digest: "sha256:output".to_string(),
            env_fingerprint: "fnp-io-unit-tests".to_string(),
            artifact_refs: packet009_artifacts(),
            duration_ms: 2,
            outcome: "pass".to_string(),
            reason_code: "io_header_schema_invalid".to_string(),
        };
        assert!(record.is_replay_complete());
    }

    #[test]
    fn packet009_log_record_rejects_missing_fields() {
        let record = IOLogRecord {
            ts_utc: String::new(),
            suite_id: String::new(),
            test_id: String::new(),
            packet_id: "wrong-packet".to_string(),
            fixture_id: String::new(),
            mode: IORuntimeMode::Hardened,
            seed: 9010,
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
    fn packet009_reason_codes_round_trip_into_logs() {
        for (idx, reason_code) in IO_PACKET_REASON_CODES.iter().enumerate() {
            let seed = u64::try_from(idx).expect("small index");
            let record = IOLogRecord {
                ts_utc: "2026-02-16T00:00:00Z".to_string(),
                suite_id: "fnp-io::tests".to_string(),
                test_id: format!("UP-009-{idx}"),
                packet_id: IO_PACKET_ID.to_string(),
                fixture_id: format!("UP-009-{idx}"),
                mode: IORuntimeMode::Strict,
                seed: 20_000 + seed,
                input_digest: "sha256:input".to_string(),
                output_digest: "sha256:output".to_string(),
                env_fingerprint: "fnp-io-unit-tests".to_string(),
                artifact_refs: packet009_artifacts(),
                duration_ms: 1,
                outcome: "pass".to_string(),
                reason_code: (*reason_code).to_string(),
            };
            assert!(record.is_replay_complete());
            assert_eq!(record.reason_code, *reason_code);
        }
    }

    #[test]
    fn io_error_reason_code_mapping_is_stable() {
        let err = IOError::HeaderSchemaInvalid("bad header");
        assert_eq!(err.reason_code(), "io_header_schema_invalid");
        let err = IOError::MagicInvalid;
        assert_eq!(err.reason_code(), "io_magic_invalid");
    }
}
