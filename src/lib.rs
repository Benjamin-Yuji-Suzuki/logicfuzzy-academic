//! logicfuzzy‑academic — pure‑Rust Mamdani Fuzzy Inference System.
//!
//! Designed as a functional equivalent of the Python
//! [scikit‑fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy) library,
//! with zero dependencies beyond Rust’s standard library.
//!
//! # Quick start
//!
//! ```
//! use logicfuzzy_academic::{MamdaniEngine, antecedent, consequent, rule, export_svg};
//!
//! let mut engine = MamdaniEngine::new();
//!
//! antecedent!(engine, "temperature", 0.0, 50.0, 501,
//!     "cold" => trimf [0.0,  0.0, 25.0],
//!     "warm" => trimf [0.0, 25.0, 50.0],
//!     "hot"  => trimf [25.0,50.0, 50.0],
//! );
//! antecedent!(engine, "humidity", 0.0, 100.0, 1001,
//!     "low"    => trimf [0.0,  0.0,  50.0],
//!     "medium" => trimf [0.0, 50.0, 100.0],
//!     "high" => trimf [50.0,100.0,100.0],
//! );
//! consequent!(engine, "fan_speed", 0.0, 100.0, 1001,
//!     "slow"   => trimf [0.0,  0.0,  50.0],
//!     "medium" => trimf [0.0, 50.0, 100.0],
//!     "fast"   => trimf [50.0,100.0,100.0],
//! );
//!
//! engine.add_rule(rule!(IF temperature IS cold AND humidity IS low    THEN fan_speed IS slow));
//! engine.add_rule(rule!(IF temperature IS warm AND humidity IS medium THEN fan_speed IS medium));
//! engine.add_rule(rule!(IF temperature IS hot  OR  humidity IS high   THEN fan_speed IS fast));
//! engine.add_rule(rule!(IF temperature IS cold AND humidity IS high   THEN fan_speed IS medium));
//!
//! engine.set_input("temperature", 45.0).unwrap();
//! engine.set_input("humidity",    90.0).unwrap();
//!
//! let result = engine.compute().unwrap();
//! println!("fan_speed = {:.2}%", result["fan_speed"]);
//!
//! export_svg!(engine, "output/", aggregated);
//! ```

pub mod engine;
pub mod error;
pub mod explain;
pub mod macros;
pub mod membership;
pub mod pso;
pub mod rule;
pub(crate) mod svg;
pub mod tsk;
pub(crate) mod util;
pub mod variable;

pub use engine::{DefuzzMethod, MamdaniEngine};
pub use error::FuzzyError;
pub use explain::{CogTable, ExplainReport, FuzzifiedVariable, RuleFiring};
pub use membership::{gaussmf, interp_membership, trapmf, trimf, MembershipFn};
pub use pso::{Particle, PsoConfig, PsoOptimizer, PsoState};
pub use rule::{Antecedent, Connector, Expression, Rule, RuleBuilder};
pub use tsk::{TskConsequent, TskEngine, TskRule};
pub use variable::{AntecedentVar, ConsequentVar, FuzzyVariable, Term, Universe};
