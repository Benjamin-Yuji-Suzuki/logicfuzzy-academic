//! # error.rs
//!
//! Error types for the Mamdani fuzzy engine.

use std::fmt;

/// Errors that can occur during a [`MamdaniEngine`](crate::MamdaniEngine) inference cycle.
///
/// Returned by [`compute`](crate::MamdaniEngine::compute) and
/// [`explain`](crate::MamdaniEngine::explain).
#[derive(Debug, Clone, PartialEq)]
pub enum FuzzyError {
    /// A registered antecedent variable has no crisp input value.
    ///
    /// Call `engine.set_input(name, value)` before `compute()`.
    MissingInput(String),

    /// A crisp input value is outside the variable's universe of discourse.
    ///
    /// The value was clamped to the universe bounds before inference.
    InputOutOfRange {
        /// Antecedent variable name.
        variable: String,
        /// The out-of-range value that was provided.
        value: f64,
        /// Universe minimum.
        min: f64,
        /// Universe maximum.
        max: f64,
    },

    /// All rules had a firing degree of zero — no rule contributed to the output.
    ///
    /// This usually means the inputs are outside the covered regions of the
    /// membership functions, or the rule base is incomplete.
    NoRulesFired,
}

impl fmt::Display for FuzzyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FuzzyError::MissingInput(name) =>
                write!(f, "missing crisp input for antecedent '{name}' — call set_input() first"),
            FuzzyError::InputOutOfRange { variable, value, min, max } =>
                write!(f, "input '{variable}' = {value} is outside universe [{min}, {max}]; value was clamped"),
            FuzzyError::NoRulesFired =>
                write!(f, "no rule fired — all firing degrees are zero"),
        }
    }
}

impl std::error::Error for FuzzyError {}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn missing_input_display() {
        let e = FuzzyError::MissingInput("temperature".to_string());
        assert!(e.to_string().contains("temperature"));
        assert!(e.to_string().contains("set_input"));
    }

    #[test] fn out_of_range_display() {
        let e = FuzzyError::InputOutOfRange {
            variable: "temp".to_string(), value: 99.0, min: 0.0, max: 50.0,
        };
        let s = e.to_string();
        assert!(s.contains("temp"));
        assert!(s.contains("99"));
        assert!(s.contains("clamped"));
    }

    #[test] fn no_rules_fired_display() {
        let e = FuzzyError::NoRulesFired;
        assert!(e.to_string().contains("zero"));
    }

    #[test] fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(FuzzyError::NoRulesFired);
        assert!(e.to_string().len() > 0);
    }

    #[test] fn fuzzy_error_eq() {
        assert_eq!(FuzzyError::NoRulesFired, FuzzyError::NoRulesFired);
        assert_ne!(FuzzyError::NoRulesFired, FuzzyError::MissingInput("x".to_string()));
    }
}
