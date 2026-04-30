pub mod engine;
pub mod explain;
pub mod macros;
pub mod membership;
pub mod rule;
pub mod svg;
pub mod variable;

pub use engine::MamdaniEngine;
pub use explain::{CogTable, ExplainReport, FuzzifiedVariable, RuleFiring};
pub use membership::{gaussmf, interp_membership, trapmf, trimf, MembershipFn};
pub use rule::{Connector, Rule, RuleBuilder};
pub use variable::{Antecedent, Consequent, FuzzyVariable, Term, Universe};
