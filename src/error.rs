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
    MissingInput(String),

    /// A crisp input value is outside the variable's universe of discourse.
    InputOutOfRange {
        variable: String,
        value: f64,
        min: f64,
        max: f64,
    },

    /// All rules had a firing degree of zero — no rule contributed to the output.
    NoRulesFired {
        /// Human-readable explanations for why no rule fired.
        diagnostics: Vec<String>,
    },

    /// A variable with the same name is already registered.
    DuplicateVariable(String),
}

impl fmt::Display for FuzzyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FuzzyError::MissingInput(name) =>
                write!(f, "missing crisp input for antecedent '{name}' — call set_input() first"),
            FuzzyError::InputOutOfRange { variable, value, min, max } =>
                write!(f, "input '{variable}' = {value} is outside universe [{min}, {max}]; value was clamped"),
            FuzzyError::NoRulesFired { diagnostics } => {
                write!(f, "no rule fired — all firing degrees are zero")?;
                if !diagnostics.is_empty() {
                    write!(f, ". Diagnostics:")?;
                    for d in diagnostics {
                        write!(f, "\n  - {}", d)?;
                    }
                }
                Ok(())
            }
            FuzzyError::DuplicateVariable(name) =>
                write!(f, "variable '{}' is already registered", name),
        }
    }
}

impl std::error::Error for FuzzyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_input_display() {
        let e = FuzzyError::MissingInput("temperature".to_string());
        assert!(e.to_string().contains("temperature"));
        assert!(e.to_string().contains("set_input"));
    }

    #[test]
    fn out_of_range_display() {
        let e = FuzzyError::InputOutOfRange {
            variable: "temp".to_string(),
            value: 99.0,
            min: 0.0,
            max: 50.0,
        };
        let s = e.to_string();
        assert!(s.contains("temp"));
        assert!(s.contains("99"));
        assert!(s.contains("clamped"));
    }

    #[test]
    fn no_rules_fired_display_empty_diagnostics() {
        let e = FuzzyError::NoRulesFired {
            diagnostics: vec![],
        };
        assert!(e.to_string().contains("zero"));
        assert!(!e.to_string().contains("Diagnostics"));
    }

    #[test]
    fn no_rules_fired_display_with_diagnostics() {
        let e = FuzzyError::NoRulesFired {
            diagnostics: vec!["temperature out of range".into()],
        };
        let s = e.to_string();
        assert!(s.contains("Diagnostics"));
        assert!(s.contains("temperature out of range"));
    }

    #[test]
    fn duplicate_variable_display() {
        let e = FuzzyError::DuplicateVariable("temp".into());
        assert!(e.to_string().contains("already registered"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(FuzzyError::NoRulesFired {
            diagnostics: vec![],
        });
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn fuzzy_error_eq() {
        assert_eq!(
            FuzzyError::NoRulesFired {
                diagnostics: vec![]
            },
            FuzzyError::NoRulesFired {
                diagnostics: vec![]
            }
        );
        assert_ne!(
            FuzzyError::NoRulesFired {
                diagnostics: vec![]
            },
            FuzzyError::MissingInput("x".to_string())
        );
    }
}
