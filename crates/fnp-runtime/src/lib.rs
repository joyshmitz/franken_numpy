#![forbid(unsafe_code)]

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityClass {
    KnownCompatible,
    KnownIncompatible,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionAction {
    Allow,
    FullValidate,
    FailClosed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionEvent {
    pub ts_millis: u128,
    pub mode: RuntimeMode,
    pub class: CompatibilityClass,
    pub risk_score: f64,
    pub action: DecisionAction,
    pub note: String,
}

#[derive(Debug, Default, Clone)]
pub struct EvidenceLedger {
    events: Vec<DecisionEvent>,
}

impl EvidenceLedger {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, event: DecisionEvent) {
        self.events.push(event);
    }

    #[must_use]
    pub fn events(&self) -> &[DecisionEvent] {
        &self.events
    }

    #[must_use]
    pub fn last(&self) -> Option<&DecisionEvent> {
        self.events.last()
    }
}

#[must_use]
pub fn decide_compatibility(
    mode: RuntimeMode,
    class: CompatibilityClass,
    risk_score: f64,
    hardened_validation_threshold: f64,
) -> DecisionAction {
    match class {
        CompatibilityClass::KnownIncompatible | CompatibilityClass::Unknown => {
            DecisionAction::FailClosed
        }
        CompatibilityClass::KnownCompatible => match mode {
            RuntimeMode::Strict => DecisionAction::Allow,
            RuntimeMode::Hardened => {
                if risk_score >= hardened_validation_threshold {
                    DecisionAction::FullValidate
                } else {
                    DecisionAction::Allow
                }
            }
        },
    }
}

pub fn decide_and_record(
    ledger: &mut EvidenceLedger,
    mode: RuntimeMode,
    class: CompatibilityClass,
    risk_score: f64,
    hardened_validation_threshold: f64,
    note: impl Into<String>,
) -> DecisionAction {
    let action = decide_compatibility(mode, class, risk_score, hardened_validation_threshold);
    let ts_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    ledger.record(DecisionEvent {
        ts_millis,
        mode,
        class,
        risk_score,
        action,
        note: note.into(),
    });
    action
}

#[cfg(feature = "asupersync")]
pub mod asupersync_integration {
    /// Marker function proving asupersync linkage is available for runtime plumbing.
    #[must_use]
    pub fn runtime_tag() -> &'static str {
        "asupersync"
    }
}

#[cfg(feature = "frankentui")]
pub mod frankentui_integration {
    /// Marker function proving FrankenTUI linkage is available for observability UIs.
    #[must_use]
    pub fn ui_tag() -> &'static str {
        "frankentui"
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompatibilityClass, DecisionAction, EvidenceLedger, RuntimeMode, decide_and_record,
        decide_compatibility,
    };

    #[test]
    fn strict_mode_allows_only_known_compatible() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Strict,
                CompatibilityClass::KnownCompatible,
                0.2,
                0.5,
            ),
            DecisionAction::Allow
        );
        assert_eq!(
            decide_compatibility(RuntimeMode::Strict, CompatibilityClass::Unknown, 0.2, 0.5),
            DecisionAction::FailClosed
        );
    }

    #[test]
    fn hardened_mode_escalates_risky_inputs() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.9,
                0.7,
            ),
            DecisionAction::FullValidate
        );
    }

    #[test]
    fn records_evidence() {
        let mut ledger = EvidenceLedger::new();
        let action = decide_and_record(
            &mut ledger,
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            0.1,
            0.7,
            "broadcast_request",
        );
        assert_eq!(action, DecisionAction::Allow);
        assert_eq!(ledger.events().len(), 1);
        let event = ledger.last().expect("event should be present");
        assert_eq!(event.note, "broadcast_request");
    }
}
