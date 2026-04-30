pub mod error;
pub mod membership;
pub mod variable;
pub mod rule;
pub mod engine;
pub mod macros;
pub mod explain;
pub mod svg;

pub use membership::{trimf, trapmf, gaussmf, interp_membership, MembershipFn};
pub use variable::{Universe, Term, FuzzyVariable, AntecedentVar, ConsequentVar};
pub use rule::{Rule, RuleBuilder, Connector, Antecedent};
pub use engine::DefuzzMethod;
pub use engine::MamdaniEngine;
pub use explain::{ExplainReport, FuzzifiedVariable, RuleFiring, CogTable};
pub use error::FuzzyError;
