pub mod engine;
pub mod error;
pub mod explain;
pub mod macros;
pub mod membership;
pub mod rule;
pub mod svg;
pub mod variable;

pub use engine::DefuzzMethod;
pub use engine::MamdaniEngine;
pub use error::FuzzyError;
pub use explain::{CogTable, ExplainReport, FuzzifiedVariable, RuleFiring};
pub use membership::{gaussmf, interp_membership, trapmf, trimf, MembershipFn};
pub use rule::{Antecedent, Connector, Rule, RuleBuilder};
pub use variable::{AntecedentVar, ConsequentVar, FuzzyVariable, Term, Universe};
