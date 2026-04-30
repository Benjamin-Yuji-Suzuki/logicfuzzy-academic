//! # rule.rs
//!
//! Fuzzy inference rules for the Mamdani system.
//! Equivalent to `ctrl.Rule` in scikit-fuzzy.
//!
//! New in v0.2.0:
//! - `NOT` negation per antecedent: `IF temperature IS NOT cold THEN ...`
//! - Rule weights: `rule.with_weight(0.8)` scales the firing degree
//! - Multiple consequents per rule: `THEN fan IS fast AND light IS bright`

use std::collections::HashMap;
use std::fmt;
use crate::variable::FuzzyVariable;

// ─────────────────────────────────────────────────────────────────
// Connector
// ─────────────────────────────────────────────────────────────────

/// Logical connector between antecedents.
///
/// - `And` → firing = MIN of membership degrees (t-norm)
/// - `Or`  → firing = MAX of membership degrees (s-norm)
#[derive(Debug, Clone, PartialEq)]
pub enum Connector {
    /// Fuzzy conjunction: min(μ₁, μ₂, …, μₙ)
    And,
    /// Fuzzy disjunction: max(μ₁, μ₂, …, μₙ)
    Or,
}

// ─────────────────────────────────────────────────────────────────
// Antecedent
// ─────────────────────────────────────────────────────────────────

/// A single antecedent condition in a fuzzy rule.
///
/// Stores the variable name, term label, and whether the condition is negated.
///
/// When `negated = true` the membership degree is complemented: `μ' = 1 - μ`.
/// This implements the standard fuzzy NOT operator.
#[derive(Debug, Clone)]
pub struct Antecedent {
    /// Name of the antecedent variable.
    pub var:     String,
    /// Linguistic term label.
    pub term:    String,
    /// Whether this antecedent is negated (`IS NOT`).
    pub negated: bool,
}

impl Antecedent {
    /// Creates a normal (non-negated) antecedent.
    pub fn new(var: impl Into<String>, term: impl Into<String>) -> Self {
        Self { var: var.into(), term: term.into(), negated: false }
    }

    /// Creates a negated antecedent (`IS NOT`).
    pub fn negated(var: impl Into<String>, term: impl Into<String>) -> Self {
        Self { var: var.into(), term: term.into(), negated: true }
    }

    /// Evaluates the membership degree for this antecedent, applying negation if set.
    pub fn eval(&self, inputs: &HashMap<String, f64>, vars: &HashMap<String, FuzzyVariable>) -> Option<f64> {
        let val = inputs.get(self.var.as_str())?;
        let var = vars.get(self.var.as_str())?;
        let mu  = var.membership_at(&self.term, *val);
        // Complemento fuzzy: NOT A = 1 - A
        Some(if self.negated { 1.0 - mu } else { mu })
    }
}

// ─────────────────────────────────────────────────────────────────
// Rule
// ─────────────────────────────────────────────────────────────────

/// A Mamdani fuzzy inference rule with optional NOT, weights, and multiple consequents.
///
/// # Example
/// ```
/// use logicfuzzy_academic::rule;
/// use logicfuzzy_academic::rule::Connector;
///
/// // Basic rule
/// let r = rule!(IF temperature IS hot THEN fan_speed IS fast);
/// assert_eq!(r.consequents()[0].0, "fan_speed");
///
/// // Rule with NOT
/// let r = rule!(IF temperature IS NOT cold THEN fan_speed IS fast);
/// assert!(r.antecedents_full()[0].negated);
///
/// // Rule with weight
/// let r = rule!(IF temperature IS hot THEN fan_speed IS fast).with_weight(0.8);
/// assert!((r.weight() - 0.8).abs() < 1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct Rule {
    /// Antecedents with variable, term, and optional NOT.
    antecedents: Vec<Antecedent>,
    /// Logical connector applied between antecedents.
    connector:   Connector,
    /// One or more (variable, term) pairs this rule concludes.
    consequents: Vec<(String, String)>,
    /// Weight [0.0, 1.0] scaling the firing degree. Default: 1.0.
    weight:      f64,
}

impl Rule {
    /// Creates a new rule from raw parts.
    ///
    /// Panics if `antecedents` is empty or `consequents` is empty.
    pub fn new(
        antecedents: Vec<Antecedent>,
        connector:   Connector,
        consequents: Vec<(String, String)>,
    ) -> Self {
        assert!(!antecedents.is_empty(), "Rule: antecedent list cannot be empty");
        assert!(!consequents.is_empty(), "Rule: consequent list cannot be empty");
        Self { antecedents, connector, consequents, weight: 1.0 }
    }

    /// Sets the rule weight and returns `self` for chaining.
    ///
    /// The weight scales the firing degree: `alpha = firing * weight`.
    /// Must be in [0.0, 1.0].
    ///
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule;
    /// let r = rule!(IF x IS a THEN y IS b).with_weight(0.75);
    /// assert!((r.weight() - 0.75).abs() < 1e-9);
    /// ```
    pub fn with_weight(mut self, weight: f64) -> Self {
        assert!((0.0..=1.0).contains(&weight),
            "Rule: weight must be in [0.0, 1.0], got {}", weight);
        self.weight = weight;
        self
    }

    /// Returns the full antecedent list including negation flags.
    pub fn antecedents_full(&self) -> &[Antecedent] {
        &self.antecedents
    }

    /// Returns `(var, term)` pairs for backward compatibility.
    pub fn antecedents(&self) -> Vec<(&str, &str)> {
        self.antecedents.iter().map(|a| (a.var.as_str(), a.term.as_str())).collect()
    }

    /// Returns the logical connector between antecedents.
    pub fn connector(&self) -> &Connector {
        &self.connector
    }

    /// Returns all consequent `(variable, term)` pairs.
    pub fn consequents(&self) -> &[(String, String)] {
        &self.consequents
    }

    /// Returns the first consequent's variable name.
    /// For single-consequent rules this is equivalent to the old `consequent_var()`.
    pub fn consequent_var(&self) -> &str {
        &self.consequents[0].0
    }

    /// Returns the first consequent's term label.
    pub fn consequent_term(&self) -> &str {
        &self.consequents[0].1
    }

    /// Returns the number of antecedents.
    pub fn antecedent_count(&self) -> usize {
        self.antecedents.len()
    }

    /// Returns the rule weight.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Evaluates the firing degree for this rule given crisp inputs.
    ///
    /// 1. Computes `μ(x)` for each antecedent (applying `NOT` if set).
    /// 2. Combines degrees with AND (min) or OR (max).
    /// 3. Multiplies by `weight`.
    ///
    /// Returns `0.0` if any variable or term is missing from the maps.
    ///
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{FuzzyVariable, Universe, Term, MembershipFn};
    /// use logicfuzzy_academic::rule::{Rule, Antecedent, Connector};
    /// use std::collections::HashMap;
    ///
    /// let mut temp = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
    /// temp.add_term(Term::new("hot", MembershipFn::Trimf([25.0, 50.0, 50.0])));
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("temperature".to_string(), 50.0);
    /// let mut vars = HashMap::new();
    /// vars.insert("temperature".to_string(), temp);
    ///
    /// let rule = Rule::new(
    ///     vec![Antecedent::new("temperature", "hot")],
    ///     Connector::And,
    ///     vec![("speed".to_string(), "fast".to_string())],
    /// );
    /// assert!((rule.firing_strength(&inputs, &vars) - 1.0).abs() < 1e-9);
    /// ```
    pub fn firing_strength(
        &self,
        inputs: &HashMap<String, f64>,
        antecedents: &HashMap<String, FuzzyVariable>,
    ) -> f64 {
        // Avalia grau de cada antecedente (com complemento se negado)
        let degrees: Vec<f64> = self.antecedents.iter()
            .filter_map(|a| a.eval(inputs, antecedents))
            .collect();

        if degrees.is_empty() { return 0.0; }

        // Aplica conector
        let firing = match self.connector {
            Connector::And => degrees.iter().cloned().fold(f64::INFINITY, f64::min),
            Connector::Or  => degrees.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        };

        // Aplica peso da regra
        (firing * self.weight).clamp(0.0, 1.0)
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let conn = match self.connector {
            Connector::And => "AND",
            Connector::Or  => "OR",
        };

        let ants: Vec<String> = self.antecedents.iter().map(|a| {
            if a.negated {
                format!("({} IS NOT {})", a.var, a.term)
            } else {
                format!("({} IS {})", a.var, a.term)
            }
        }).collect();

        let cons: Vec<String> = self.consequents.iter()
            .map(|(v, t)| format!("{} IS {}", v, t))
            .collect();

        let weight_str = if (self.weight - 1.0).abs() > 1e-9 {
            format!(" [w={:.2}]", self.weight)
        } else {
            String::new()
        };

        write!(f, "IF {} THEN {}{}",
            ants.join(&format!(" {} ", conn)),
            cons.join(" AND "),
            weight_str,
        )
    }
}

// ─────────────────────────────────────────────────────────────────
// RuleBuilder
// ─────────────────────────────────────────────────────────────────

/// Fluent builder for constructing fuzzy rules.
///
/// Supports normal and negated (`IS NOT`) antecedents,
/// multiple consequents, and rule weights.
///
/// # Example
/// ```
/// use logicfuzzy_academic::rule::{RuleBuilder, Antecedent};
///
/// let r = RuleBuilder::new()
///     .when("temperature", "hot")
///     .and_not("humidity", "high")
///     .then("fan_speed", "fast")
///     .also("light", "bright")
///     .weight(0.9)
///     .build();
///
/// assert_eq!(r.consequents().len(), 2);
/// assert!((r.weight() - 0.9).abs() < 1e-9);
/// assert!(r.antecedents_full()[1].negated);
/// ```
#[derive(Debug, Default)]
pub struct RuleBuilder {
    antecedents: Vec<Antecedent>,
    connector:   Option<Connector>,
    consequents: Vec<(String, String)>,
    weight:      f64,
}

impl RuleBuilder {
    /// Creates an empty builder with default weight 1.0.
    pub fn new() -> Self {
        Self { weight: 1.0, ..Default::default() }
    }

    /// Starts the rule with the first antecedent (normal, not negated).
    pub fn when(mut self, var: &str, term: &str) -> Self {
        self.antecedents.push(Antecedent::new(var, term));
        self
    }

    /// Starts the rule with the first antecedent negated (`IS NOT`).
    pub fn when_not(mut self, var: &str, term: &str) -> Self {
        self.antecedents.push(Antecedent::negated(var, term));
        self
    }

    /// Adds a normal antecedent with AND connector.
    pub fn and(mut self, var: &str, term: &str) -> Self {
        self.connector = Some(Connector::And);
        self.antecedents.push(Antecedent::new(var, term));
        self
    }

    /// Adds a negated antecedent with AND connector.
    pub fn and_not(mut self, var: &str, term: &str) -> Self {
        self.connector = Some(Connector::And);
        self.antecedents.push(Antecedent::negated(var, term));
        self
    }

    /// Adds a normal antecedent with OR connector.
    pub fn or(mut self, var: &str, term: &str) -> Self {
        self.connector = Some(Connector::Or);
        self.antecedents.push(Antecedent::new(var, term));
        self
    }

    /// Adds a negated antecedent with OR connector.
    pub fn or_not(mut self, var: &str, term: &str) -> Self {
        self.connector = Some(Connector::Or);
        self.antecedents.push(Antecedent::negated(var, term));
        self
    }

    /// Sets the first consequent and returns the builder.
    pub fn then(mut self, var: &str, term: &str) -> Self {
        self.consequents.push((var.to_string(), term.to_string()));
        self
    }

    /// Adds an additional consequent (for multi-consequent rules).
    pub fn also(mut self, var: &str, term: &str) -> Self {
        self.consequents.push((var.to_string(), term.to_string()));
        self
    }

    /// Sets the rule weight [0.0, 1.0].
    pub fn weight(mut self, w: f64) -> Self {
        self.weight = w;
        self
    }

    /// Builds and returns the `Rule`.
    ///
    /// Panics if no antecedent or no consequent was provided.
    pub fn build(self) -> Rule {
        let conn = self.connector.unwrap_or(Connector::And);
        let mut rule = Rule::new(self.antecedents, conn, self.consequents);
        rule.weight = self.weight;
        rule
    }
}

// Backward-compat: allow `.then()` to directly return a Rule for single-consequent rules
impl RuleBuilder {
    /// Convenience: sets the first (and only) consequent and immediately builds the rule.
    ///
    /// For multi-consequent rules, use `.then().also().build()` instead.
    pub fn done(self) -> Rule {
        self.build()
    }
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FuzzyVariable, Universe, Term, MembershipFn};

    fn make_vars() -> (HashMap<String, f64>, HashMap<String, FuzzyVariable>) {
        let mut temp = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
        temp.add_term(Term::new("cold", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        temp.add_term(Term::new("hot",  MembershipFn::Trimf([25.0,50.0, 50.0])));

        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 5.0);  // cold: 0.8, hot: 0.0

        let mut vars = HashMap::new();
        vars.insert("temperature".to_string(), temp);

        (inputs, vars)
    }

    // ── Connector ─────────────────────────────────────────────────

    #[test] fn connector_and_eq()  { assert_eq!(Connector::And, Connector::And); }
    #[test] fn connector_and_ne()  { assert_ne!(Connector::And, Connector::Or);  }

    // ── Antecedent ────────────────────────────────────────────────

    #[test] fn antecedent_normal_eval() {
        let (inputs, vars) = make_vars();
        let a = Antecedent::new("temperature", "cold");
        let mu = a.eval(&inputs, &vars).unwrap();
        assert!((mu - 0.8).abs() < 1e-6);
    }

    #[test] fn antecedent_negated_eval() {
        let (inputs, vars) = make_vars();
        let a = Antecedent::negated("temperature", "cold");
        let mu = a.eval(&inputs, &vars).unwrap();
        // NOT cold: 1 - 0.8 = 0.2
        assert!((mu - 0.2).abs() < 1e-6);
    }

    #[test] fn antecedent_missing_var_returns_none() {
        let inputs = HashMap::new();
        let vars   = HashMap::new();
        let a = Antecedent::new("missing", "term");
        assert!(a.eval(&inputs, &vars).is_none());
    }

    // ── Rule — basic ──────────────────────────────────────────────

    #[test] fn rule_one_antecedent() {
        let r = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("speed".to_string(), "slow".to_string())],
        );
        assert_eq!(r.antecedent_count(), 1);
        assert_eq!(r.consequent_var(),  "speed");
        assert_eq!(r.consequent_term(), "slow");
        assert!((r.weight() - 1.0).abs() < 1e-9);
    }

    #[test] fn rule_empty_antecedents_panics() {
        let result = std::panic::catch_unwind(|| {
            Rule::new(vec![], Connector::And, vec![("y".to_string(), "b".to_string())])
        });
        assert!(result.is_err());
    }

    #[test] fn rule_empty_consequents_panics() {
        let result = std::panic::catch_unwind(|| {
            Rule::new(
                vec![Antecedent::new("x", "a")],
                Connector::And,
                vec![],
            )
        });
        assert!(result.is_err());
    }

    // ── Rule — NOT ────────────────────────────────────────────────

    #[test] fn rule_not_antecedent_fires_higher_when_term_low() {
        // temperature=5 → cold=0.8 → NOT cold=0.2
        // A regra com NOT cold deve disparar com grau 0.2
        let (inputs, vars) = make_vars();
        let r = Rule::new(
            vec![Antecedent::negated("temperature", "cold")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        let alpha = r.firing_strength(&inputs, &vars);
        assert!((alpha - 0.2).abs() < 1e-6);
    }

    #[test] fn rule_not_antecedent_display() {
        let r = Rule::new(
            vec![Antecedent::negated("temperature", "cold")],
            Connector::And,
            vec![("speed".to_string(), "fast".to_string())],
        );
        let s = r.to_string();
        assert!(s.contains("IS NOT"), "display deve mostrar IS NOT: {}", s);
    }

    // ── Rule — weight ─────────────────────────────────────────────

    #[test] fn rule_weight_scales_firing() {
        let (inputs, vars) = make_vars();
        // cold=0.8, weight=0.5 → firing=0.4
        let r = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        ).with_weight(0.5);
        let alpha = r.firing_strength(&inputs, &vars);
        assert!((alpha - 0.4).abs() < 1e-6);
    }

    #[test] fn rule_weight_zero_never_fires() {
        let (inputs, vars) = make_vars();
        let r = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        ).with_weight(0.0);
        assert_eq!(r.firing_strength(&inputs, &vars), 0.0);
    }

    #[test] fn rule_weight_invalid_panics() {
        let result = std::panic::catch_unwind(|| {
            Rule::new(
                vec![Antecedent::new("x", "a")],
                Connector::And,
                vec![("y".to_string(), "b".to_string())],
            ).with_weight(1.5)
        });
        assert!(result.is_err());
    }

    #[test] fn rule_weight_displayed_when_not_one() {
        let r = Rule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        ).with_weight(0.75);
        assert!(r.to_string().contains("w=0.75"));
    }

    #[test] fn rule_weight_not_displayed_when_one() {
        let r = Rule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        assert!(!r.to_string().contains('w'));
    }

    // ── Rule — multiple consequents ───────────────────────────────

    #[test] fn rule_multiple_consequents() {
        let r = Rule::new(
            vec![Antecedent::new("temperature", "hot")],
            Connector::And,
            vec![
                ("fan".to_string(),   "fast".to_string()),
                ("light".to_string(), "bright".to_string()),
            ],
        );
        assert_eq!(r.consequents().len(), 2);
        assert_eq!(r.consequents()[0].0, "fan");
        assert_eq!(r.consequents()[1].0, "light");
    }

    #[test] fn rule_multi_consequent_display() {
        let r = Rule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![
                ("y".to_string(), "b".to_string()),
                ("z".to_string(), "c".to_string()),
            ],
        );
        let s = r.to_string();
        assert!(s.contains("y IS b"), "faltou primeiro consequente: {}", s);
        assert!(s.contains("z IS c"), "faltou segundo consequente: {}", s);
    }

    // ── Firing strength ───────────────────────────────────────────

    #[test] fn and_uses_min() {
        let (mut inputs, mut vars) = make_vars();
        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0,100.0,1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0,0.0,50.0])));
        inputs.insert("humidity".to_string(), 20.0); // low=0.6
        vars.insert("humidity".to_string(), hum);

        // cold=0.8, low=0.6 → AND = min = 0.6
        let r = Rule::new(
            vec![Antecedent::new("temperature","cold"), Antecedent::new("humidity","low")],
            Connector::And,
            vec![("y".to_string(),"b".to_string())],
        );
        assert!((r.firing_strength(&inputs,&vars) - 0.6).abs() < 1e-6);
    }

    #[test] fn or_uses_max() {
        let (mut inputs, mut vars) = make_vars();
        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0,100.0,1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0,0.0,50.0])));
        inputs.insert("humidity".to_string(), 20.0); // low=0.6
        vars.insert("humidity".to_string(), hum);

        // cold=0.8, low=0.6 → OR = max = 0.8
        let r = Rule::new(
            vec![Antecedent::new("temperature","cold"), Antecedent::new("humidity","low")],
            Connector::Or,
            vec![("y".to_string(),"b".to_string())],
        );
        assert!((r.firing_strength(&inputs,&vars) - 0.8).abs() < 1e-6);
    }

    #[test] fn missing_variable_returns_zero() {
        let inputs = HashMap::new();
        let vars   = HashMap::new();
        let r = Rule::new(
            vec![Antecedent::new("missing","term")],
            Connector::And,
            vec![("y".to_string(),"b".to_string())],
        );
        assert_eq!(r.firing_strength(&inputs,&vars), 0.0);
    }

    // ── RuleBuilder ───────────────────────────────────────────────

    #[test] fn builder_basic() {
        let r = RuleBuilder::new()
            .when("temperature", "cold")
            .then("speed", "slow")
            .build();
        assert_eq!(r.antecedent_count(), 1);
        assert_eq!(r.consequent_var(), "speed");
    }

    #[test] fn builder_and_not() {
        let r = RuleBuilder::new()
            .when("temperature", "hot")
            .and_not("humidity", "low")
            .then("speed", "fast")
            .build();
        assert!(r.antecedents_full()[1].negated);
        assert_eq!(r.connector(), &Connector::And);
    }

    #[test] fn builder_or_not() {
        let r = RuleBuilder::new()
            .when("temperature", "hot")
            .or_not("humidity", "low")
            .then("speed", "fast")
            .build();
        assert!(r.antecedents_full()[1].negated);
        assert_eq!(r.connector(), &Connector::Or);
    }

    #[test] fn builder_multi_consequent() {
        let r = RuleBuilder::new()
            .when("x", "a")
            .then("y", "b")
            .also("z", "c")
            .build();
        assert_eq!(r.consequents().len(), 2);
    }

    #[test] fn builder_weight() {
        let r = RuleBuilder::new()
            .when("x", "a")
            .then("y", "b")
            .weight(0.6)
            .build();
        assert!((r.weight() - 0.6).abs() < 1e-9);
    }

    // ── Display ───────────────────────────────────────────────────

    #[test] fn display_contains_if_then() {
        let r = Rule::new(
            vec![Antecedent::new("temperature","cold")],
            Connector::And,
            vec![("speed".to_string(),"slow".to_string())],
        );
        let s = r.to_string();
        assert!(s.contains("IF"));
        assert!(s.contains("THEN"));
    }

    #[test] fn display_and_or() {
        let r_and = RuleBuilder::new().when("x","a").and("y","b").then("z","c").build();
        let r_or  = RuleBuilder::new().when("x","a").or("y","b").then("z","c").build();
        assert!(r_and.to_string().contains("AND"));
        assert!(r_or.to_string().contains("OR"));
    }
}
