//! # rule.rs
//!
//! Fuzzy inference rules for the Mamdani system.
//! Equivalent to `ctrl.Rule` in scikit-fuzzy.

use crate::variable::FuzzyVariable;
use std::collections::BTreeMap;
use std::fmt;

/// Logical connector between antecedents in a flat rule: AND (min) or OR (max).
/// # Example
/// ```
/// use logicfuzzy_academic::rule::Connector;
/// let c = Connector::And;
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Connector {
    /// Minimum t-norm: firing = min(μ₁, μ₂, …)
    And,
    /// Maximum s-norm: firing = max(μ₁, μ₂, …)
    Or,
}

/// A single antecedent condition: `variable IS [NOT] term`.
#[derive(Debug, Clone)]
pub struct Antecedent {
    pub var: String,
    pub term: String,
    pub negated: bool,
}

impl Antecedent {
    /// Creates a non-negated antecedent: `var IS term`.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::Antecedent;
    /// let a = Antecedent::new("temperature", "cold");
    /// assert!(!a.negated);
    /// ```
    pub fn new(var: impl Into<String>, term: impl Into<String>) -> Self {
        Self {
            var: var.into(),
            term: term.into(),
            negated: false,
        }
    }

    /// Creates a negated antecedent: `var IS NOT term` (complement: `1 − μ`).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::Antecedent;
    /// let a = Antecedent::negated("temperature", "cold");
    /// assert!(a.negated);
    /// ```
    pub fn negated(var: impl Into<String>, term: impl Into<String>) -> Self {
        Self {
            var: var.into(),
            term: term.into(),
            negated: true,
        }
    }

    /// Evaluates this antecedent given the current inputs and registered variables.
    /// Returns `None` if the variable or term is missing.
    pub fn eval(
        &self,
        inputs: &BTreeMap<String, f64>,
        vars: &BTreeMap<String, FuzzyVariable>,
    ) -> Option<f64> {
        let val = inputs.get(self.var.as_str())?;
        let var = vars.get(self.var.as_str())?;
        let term = var.get_term(&self.term)?;
        let mu = term.eval(*val);
        Some(if self.negated { 1.0 - mu } else { mu })
    }
}

/// A logical expression tree combining antecedents with AND/OR connectors.
/// Supports arbitrary nesting for mixed expressions like (A AND B) OR (C AND D).
#[derive(Debug, Clone)]
pub enum Expression {
    /// A single antecedent
    Term { antecedent: Antecedent },
    /// Logical AND of multiple sub-expressions
    And { operands: Vec<Expression> },
    /// Logical OR of multiple sub-expressions
    Or { operands: Vec<Expression> },
}

impl Expression {
    /// Creates a terminal expression from an Antecedent.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::{Expression, Antecedent};
    /// let expr = Expression::term(Antecedent::new("x", "a"));
    /// ```
    pub fn term(antecedent: Antecedent) -> Self {
        Expression::Term { antecedent }
    }

    /// Creates an AND expression from a list of sub-expressions.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::{Expression, Antecedent};
    /// let expr = Expression::and(vec![
    ///     Expression::term(Antecedent::new("x", "a")),
    ///     Expression::term(Antecedent::new("y", "b")),
    /// ]);
    /// ```
    pub fn and(operands: Vec<Expression>) -> Self {
        assert!(
            !operands.is_empty(),
            "Expression::And requires at least one operand"
        );
        Expression::And { operands }
    }

    /// Creates an OR expression from a list of sub-expressions.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::{Expression, Antecedent};
    /// let expr = Expression::or(vec![
    ///     Expression::term(Antecedent::new("x", "a")),
    ///     Expression::term(Antecedent::new("y", "b")),
    /// ]);
    /// ```
    pub fn or(operands: Vec<Expression>) -> Self {
        assert!(
            !operands.is_empty(),
            "Expression::Or requires at least one operand"
        );
        Expression::Or { operands }
    }

    /// Evaluates the expression tree.
    pub fn eval(
        &self,
        inputs: &BTreeMap<String, f64>,
        antecedents: &BTreeMap<String, FuzzyVariable>,
    ) -> f64 {
        match self {
            Expression::Term { antecedent } => antecedent.eval(inputs, antecedents).unwrap_or(0.0),
            Expression::And { operands } => operands
                .iter()
                .map(|op| op.eval(inputs, antecedents))
                .fold(f64::INFINITY, f64::min),
            Expression::Or { operands } => operands
                .iter()
                .map(|op| op.eval(inputs, antecedents))
                .fold(f64::NEG_INFINITY, f64::max),
        }
    }

    /// Collects all leaf antecedents of the expression tree.
    /// Used by `validate_rules()` to inspect expression-based rules.
    pub fn antecedents(&self) -> Vec<&Antecedent> {
        match self {
            Expression::Term { antecedent } => vec![antecedent],
            Expression::And { operands } | Expression::Or { operands } => {
                operands.iter().flat_map(|op| op.antecedents()).collect()
            }
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Term { antecedent } => {
                if antecedent.negated {
                    write!(f, "({} IS NOT {})", antecedent.var, antecedent.term)
                } else {
                    write!(f, "({} IS {})", antecedent.var, antecedent.term)
                }
            }
            Expression::And { operands } => {
                let parts: Vec<String> = operands.iter().map(|op| op.to_string()).collect();
                write!(f, "({})", parts.join(" AND "))
            }
            Expression::Or { operands } => {
                let parts: Vec<String> = operands.iter().map(|op| op.to_string()).collect();
                write!(f, "({})", parts.join(" OR "))
            }
        }
    }
}

/// A Mamdani inference rule: `IF <antecedents> THEN <consequents>`.
///
/// Rules can be created via the [`rule!`](crate::rule!) macro, [`RuleBuilder`], or [`Rule::from_expression`]
/// for arbitrary nested logic.
/// # Example
/// ```
/// use logicfuzzy_academic::rule::{Rule, Antecedent, Connector};
/// let rule = Rule::new(
///     vec![Antecedent::new("temp", "cold")],
///     Connector::And,
///     vec![("speed".into(), "slow".into())],
/// );
/// ```
#[derive(Debug, Clone)]
pub struct Rule {
    /// Optional expression tree; if present, it replaces antecedents + connector.
    expression: Option<Expression>,
    /// Backward-compatible flat representation; used when expression is None.
    antecedents: Vec<Antecedent>,
    connector: Connector,
    consequents: Vec<(String, String)>,
    weight: f64,
}

impl Rule {
    /// Create a rule from a list of antecedents and a single connector (backward-compatible).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::{Rule, Antecedent, Connector};
    /// let rule = Rule::new(
    ///     vec![Antecedent::new("temp", "cold")],
    ///     Connector::And,
    ///     vec![("speed".into(), "slow".into())],
    /// );
    /// ```
    pub fn new(
        antecedents: Vec<Antecedent>,
        connector: Connector,
        consequents: Vec<(String, String)>,
    ) -> Self {
        assert!(
            !antecedents.is_empty(),
            "Rule: antecedent list cannot be empty"
        );
        assert!(
            !consequents.is_empty(),
            "Rule: consequent list cannot be empty"
        );
        Self {
            expression: None,
            antecedents,
            connector,
            consequents,
            weight: 1.0,
        }
    }

    /// Create a rule from an expression tree (AST).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::{Rule, Expression, Antecedent};
    /// let expr = Expression::term(Antecedent::new("x", "a"));
    /// let rule = Rule::from_expression(expr, vec![("y".into(), "b".into())]);
    /// ```
    pub fn from_expression(expression: Expression, consequents: Vec<(String, String)>) -> Self {
        assert!(
            !consequents.is_empty(),
            "Rule: consequent list cannot be empty"
        );
        Self {
            expression: Some(expression),
            antecedents: vec![],
            connector: Connector::And, // not used
            consequents,
            weight: 1.0,
        }
    }

    /// Sets the rule weight, scaling the firing degree. Must be in `[0.0, 1.0]`.
    ///
    /// # Panics
    /// Panics if `weight` is outside `[0.0, 1.0]`.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::{Rule, Antecedent, Connector};
    /// let rule = Rule::new(
    ///     vec![Antecedent::new("x", "a")],
    ///     Connector::And,
    ///     vec![("y".into(), "b".into())],
    /// ).with_weight(0.8);
    /// assert!((rule.weight() - 0.8).abs() < 1e-9);
    /// ```
    pub fn with_weight(mut self, weight: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&weight),
            "Rule: weight must be in [0.0, 1.0], got {}",
            weight
        );
        self.weight = weight;
        self
    }

    /// Returns the antecedents as a slice of [`Antecedent`] structs (with negation flag).
    pub fn antecedents_full(&self) -> &[Antecedent] {
        &self.antecedents
    }

    /// Returns `(variable, term)` pairs for each antecedent.
    pub fn antecedents(&self) -> Vec<(&str, &str)> {
        self.antecedents
            .iter()
            .map(|a| (a.var.as_str(), a.term.as_str()))
            .collect()
    }

    /// Returns the logical connector (`And` / `Or`) used for flat antecedent lists.
    ///
    /// For rules created via [`Rule::from_expression`], this returns `And` by default,
    /// but the actual logic is determined by the [`Expression`] tree — use
    /// [`Rule::expression`] to inspect it.
    pub fn connector(&self) -> &Connector {
        &self.connector
    }

    /// Returns all consequents as `(variable, term)` pairs.
    pub fn consequents(&self) -> &[(String, String)] {
        &self.consequents
    }

    /// Returns the variable name of the first consequent.
    pub fn consequent_var(&self) -> &str {
        &self.consequents[0].0
    }

    /// Returns the term name of the first consequent.
    pub fn consequent_term(&self) -> &str {
        &self.consequents[0].1
    }

    /// Returns the number of antecedents in the flat list (0 for expression-based rules).
    pub fn antecedent_count(&self) -> usize {
        self.antecedents.len()
    }

    /// Returns the rule weight (default `1.0`).
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns the expression tree, if any.
    pub fn expression(&self) -> Option<&Expression> {
        self.expression.as_ref()
    }

    /// Returns `true` if this rule was created from an [`Expression`] tree.
    pub fn is_expression_based(&self) -> bool {
        self.expression.is_some()
    }

    /// Computes the firing strength of the rule given the current crisp inputs.
    ///
    /// Uses the expression tree if present; otherwise evaluates the flat antecedent list.
    /// Returns `0.0` immediately if any variable or term is missing.
    pub fn firing_strength(
        &self,
        inputs: &BTreeMap<String, f64>,
        antecedents: &BTreeMap<String, FuzzyVariable>,
    ) -> f64 {
        if let Some(expr) = &self.expression {
            return (expr.eval(inputs, antecedents) * self.weight).clamp(0.0, 1.0);
        }

        // Backward-compatible flat evaluation
        let mut degrees = Vec::with_capacity(self.antecedents.len());
        for ant in &self.antecedents {
            match ant.eval(inputs, antecedents) {
                Some(d) => degrees.push(d),
                None => return 0.0,
            }
        }

        if degrees.is_empty() {
            return 0.0;
        }

        let firing = match self.connector {
            Connector::And => degrees.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            Connector::Or => degrees.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        };

        (firing * self.weight).clamp(0.0, 1.0)
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let condition = if let Some(expr) = &self.expression {
            expr.to_string()
        } else {
            let conn = match self.connector {
                Connector::And => "AND",
                Connector::Or => "OR",
            };
            let ants: Vec<String> = self
                .antecedents
                .iter()
                .map(|a| {
                    if a.negated {
                        format!("({} IS NOT {})", a.var, a.term)
                    } else {
                        format!("({} IS {})", a.var, a.term)
                    }
                })
                .collect();
            ants.join(&format!(" {} ", conn))
        };

        let cons: Vec<String> = self
            .consequents
            .iter()
            .map(|(v, t)| format!("{} IS {}", v, t))
            .collect();

        let weight_str = if (self.weight - 1.0).abs() > 1e-9 {
            format!(" [w={:.2}]", self.weight)
        } else {
            String::new()
        };

        write!(
            f,
            "IF {} THEN {}{}",
            condition,
            cons.join(" AND "),
            weight_str
        )
    }
}

/// Fluent builder for constructing [`Rule`] instances.
///
/// # Example
/// ```
/// use logicfuzzy_academic::rule::RuleBuilder;
/// let rule = RuleBuilder::new()
///     .when("temperature", "hot")
///     .and("humidity", "high")
///     .then("fan_speed", "fast")
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct RuleBuilder {
    antecedents: Vec<Antecedent>,
    connector: Option<Connector>,
    expression: Option<Expression>,
    consequents: Vec<(String, String)>,
    weight: f64,
}

impl RuleBuilder {
    /// Creates a new `RuleBuilder` with default weight `1.0`.
    pub fn new() -> Self {
        Self {
            weight: 1.0,
            ..Default::default()
        }
    }

    /// Adds the first antecedent (`IF var IS term`).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// let rule = RuleBuilder::new()
    ///     .when("temperature", "cold")
    ///     .then("speed", "slow")
    ///     .build();
    /// ```
    pub fn when(mut self, var: &str, term: &str) -> Self {
        self.antecedents.push(Antecedent::new(var, term));
        self
    }

    /// Adds the first antecedent with negation (`IF var IS NOT term`).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// let rule = RuleBuilder::new()
    ///     .when_not("temperature", "cold")
    ///     .then("speed", "fast")
    ///     .build();
    /// ```
    pub fn when_not(mut self, var: &str, term: &str) -> Self {
        self.antecedents.push(Antecedent::negated(var, term));
        self
    }

    /// Accepts an `Expression` tree; this replaces any previous antecedents.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::{RuleBuilder, Expression, Antecedent};
    /// let expr = Expression::term(Antecedent::new("x", "a"));
    /// let rule = RuleBuilder::new()
    ///     .when_expr(expr)
    ///     .then("y", "b")
    ///     .build();
    /// ```
    pub fn when_expr(mut self, expr: Expression) -> Self {
        self.expression = Some(expr);
        self
    }

    /// Adds an AND antecedent (`AND var IS term`).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// let rule = RuleBuilder::new()
    ///     .when("x", "a")
    ///     .and("y", "b")
    ///     .then("z", "c")
    ///     .build();
    /// ```
    pub fn and(mut self, var: &str, term: &str) -> Self {
        self.connector = Some(Connector::And);
        self.antecedents.push(Antecedent::new(var, term));
        self
    }

    /// Adds an AND antecedent with negation (`AND var IS NOT term`).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// let rule = RuleBuilder::new()
    ///     .when("x", "a")
    ///     .and_not("y", "b")
    ///     .then("z", "c")
    ///     .build();
    /// ```
    pub fn and_not(mut self, var: &str, term: &str) -> Self {
        self.connector = Some(Connector::And);
        self.antecedents.push(Antecedent::negated(var, term));
        self
    }

    /// Adds an OR antecedent (`OR var IS term`).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// let rule = RuleBuilder::new()
    ///     .when("x", "a")
    ///     .or("y", "b")
    ///     .then("z", "c")
    ///     .build();
    /// ```
    pub fn or(mut self, var: &str, term: &str) -> Self {
        self.connector = Some(Connector::Or);
        self.antecedents.push(Antecedent::new(var, term));
        self
    }

    /// Adds an OR antecedent with negation (`OR var IS NOT term`).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// let rule = RuleBuilder::new()
    ///     .when("x", "a")
    ///     .or_not("y", "b")
    ///     .then("z", "c")
    ///     .build();
    /// ```
    pub fn or_not(mut self, var: &str, term: &str) -> Self {
        self.connector = Some(Connector::Or);
        self.antecedents.push(Antecedent::negated(var, term));
        self
    }

    /// Sets the primary consequent (`THEN var IS term`).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// let rule = RuleBuilder::new()
    ///     .when("x", "a")
    ///     .then("y", "b")
    ///     .build();
    /// assert_eq!(rule.consequent_var(), "y");
    /// ```
    pub fn then(mut self, var: &str, term: &str) -> Self {
        self.consequents.push((var.to_string(), term.to_string()));
        self
    }

    /// Adds an additional consequent (`AND var IS term` on the THEN side).
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// let rule = RuleBuilder::new()
    ///     .when("x", "a")
    ///     .then("y", "b")
    ///     .also("z", "c")
    ///     .build();
    /// assert_eq!(rule.consequents().len(), 2);
    /// ```
    pub fn also(mut self, var: &str, term: &str) -> Self {
        self.consequents.push((var.to_string(), term.to_string()));
        self
    }

    /// Sets the rule weight. Must be in `[0.0, 1.0]`.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// let rule = RuleBuilder::new()
    ///     .when("x", "a")
    ///     .then("y", "b")
    ///     .weight(0.5)
    ///     .build();
    /// ```
    pub fn weight(mut self, w: f64) -> Self {
        self.weight = w;
        self
    }

    /// Builds the [`Rule`]. Alias: [`done`](Self::done).
    pub fn build(self) -> Rule {
        if let Some(expr) = self.expression {
            Rule::from_expression(expr, self.consequents).with_weight(self.weight)
        } else {
            let conn = self.connector.unwrap_or(Connector::And);
            let rule = Rule::new(self.antecedents, conn, self.consequents);
            rule.with_weight(self.weight)
        }
    }

    /// Alias for [`build`](Self::build).
    pub fn done(self) -> Rule {
        self.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FuzzyVariable, MembershipFn, Term, Universe};

    fn make_vars() -> (BTreeMap<String, f64>, BTreeMap<String, FuzzyVariable>) {
        let mut temp = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
        temp.add_term(Term::new("cold", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        temp.add_term(Term::new("hot", MembershipFn::Trimf([25.0, 50.0, 50.0])));

        let mut inputs = BTreeMap::new();
        inputs.insert("temperature".to_string(), 5.0);

        let mut vars = BTreeMap::new();
        vars.insert("temperature".to_string(), temp);

        (inputs, vars)
    }

    #[test]
    fn connector_and_eq() {
        assert_eq!(Connector::And, Connector::And);
    }
    #[test]
    fn connector_and_ne() {
        assert_ne!(Connector::And, Connector::Or);
    }

    #[test]
    fn antecedent_normal_eval() {
        let (inputs, vars) = make_vars();
        let a = Antecedent::new("temperature", "cold");
        let mu = a.eval(&inputs, &vars).unwrap();
        assert!((mu - 0.8).abs() < 1e-6);
    }

    #[test]
    fn antecedent_negated_eval() {
        let (inputs, vars) = make_vars();
        let a = Antecedent::negated("temperature", "cold");
        let mu = a.eval(&inputs, &vars).unwrap();
        assert!((mu - 0.2).abs() < 1e-6);
    }

    #[test]
    fn antecedent_missing_var_returns_none() {
        let inputs = BTreeMap::new();
        let vars = BTreeMap::new();
        let a = Antecedent::new("missing", "term");
        assert!(a.eval(&inputs, &vars).is_none());
    }

    // ── Expression tests ──────────────────────────────────────────

    #[test]
    fn expression_term_eval() {
        let (inputs, vars) = make_vars();
        let expr = Expression::term(Antecedent::new("temperature", "cold"));
        let result = expr.eval(&inputs, &vars);
        assert!((result - 0.8).abs() < 1e-6);
    }

    #[test]
    fn expression_and_eval() {
        let (mut inputs, mut vars) = make_vars();
        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0, 100.0, 1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        inputs.insert("humidity".to_string(), 20.0);
        vars.insert("humidity".to_string(), hum);

        let expr = Expression::and(vec![
            Expression::term(Antecedent::new("temperature", "cold")),
            Expression::term(Antecedent::new("humidity", "low")),
        ]);
        let result = expr.eval(&inputs, &vars);
        // temperature cold = 0.8, humidity low = 0.6, min = 0.6
        assert!((result - 0.6).abs() < 1e-6);
    }

    #[test]
    fn expression_or_eval() {
        let (mut inputs, mut vars) = make_vars();
        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0, 100.0, 1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        inputs.insert("humidity".to_string(), 20.0);
        vars.insert("humidity".to_string(), hum);

        let expr = Expression::or(vec![
            Expression::term(Antecedent::new("temperature", "cold")),
            Expression::term(Antecedent::new("humidity", "low")),
        ]);
        let result = expr.eval(&inputs, &vars);
        // max(0.8, 0.6) = 0.8
        assert!((result - 0.8).abs() < 1e-6);
    }

    #[test]
    fn expression_nested_and_or() {
        let (mut inputs, mut vars) = make_vars();
        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0, 100.0, 1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        inputs.insert("humidity".to_string(), 20.0);
        vars.insert("humidity".to_string(), hum);

        // (temperature IS cold AND humidity IS low) OR (temperature IS hot)
        let expr = Expression::or(vec![
            Expression::and(vec![
                Expression::term(Antecedent::new("temperature", "cold")),
                Expression::term(Antecedent::new("humidity", "low")),
            ]),
            Expression::term(Antecedent::new("temperature", "hot")),
        ]);
        let result = expr.eval(&inputs, &vars);
        // min(0.8, 0.6)=0.6, hot=0.0 => max(0.6, 0.0)=0.6
        assert!((result - 0.6).abs() < 1e-6);
    }

    #[test]
    fn expression_display() {
        let expr = Expression::and(vec![
            Expression::term(Antecedent::new("x", "a")),
            Expression::term(Antecedent::new("y", "b")),
        ]);
        let s = expr.to_string();
        assert!(s.contains("x IS a"));
        assert!(s.contains("y IS b"));
        assert!(s.contains("AND"));
    }

    #[test]
    fn expression_antecedents_recursive() {
        let expr = Expression::and(vec![
            Expression::term(Antecedent::new("x", "a")),
            Expression::or(vec![
                Expression::term(Antecedent::new("y", "b")),
                Expression::term(Antecedent::new("z", "c")),
            ]),
        ]);
        let ants = expr.antecedents();
        assert_eq!(ants.len(), 3);
        assert!(ants.iter().any(|a| a.var == "x"));
        assert!(ants.iter().any(|a| a.var == "y"));
        assert!(ants.iter().any(|a| a.var == "z"));
    }

    // ── Rule with expression ──────────────────────────────────────

    #[test]
    fn rule_from_expression_basic() {
        let (inputs, vars) = make_vars();
        let expr = Expression::term(Antecedent::new("temperature", "cold"));
        let rule = Rule::from_expression(expr, vec![("speed".to_string(), "slow".to_string())]);
        let firing = rule.firing_strength(&inputs, &vars);
        assert!((firing - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rule_builder_with_expression() {
        let (mut inputs, mut vars) = make_vars();
        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0, 100.0, 1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        inputs.insert("humidity".to_string(), 20.0);
        vars.insert("humidity".to_string(), hum);

        let expr = Expression::or(vec![
            Expression::and(vec![
                Expression::term(Antecedent::new("temperature", "cold")),
                Expression::term(Antecedent::new("humidity", "low")),
            ]),
            Expression::term(Antecedent::new("temperature", "hot")),
        ]);

        let rule = RuleBuilder::new()
            .when_expr(expr)
            .then("speed", "fast")
            .build();

        assert!(rule.firing_strength(&inputs, &vars) > 0.5);
        let display = rule.to_string();
        assert!(display.contains("AND"));
        assert!(display.contains("OR"));
    }

    // ── Existing tests (adapted where necessary) ──────────────────

    #[test]
    fn rule_one_antecedent() {
        let r = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("speed".to_string(), "slow".to_string())],
        );
        assert_eq!(r.antecedent_count(), 1);
        assert_eq!(r.consequent_var(), "speed");
        assert_eq!(r.consequent_term(), "slow");
        assert!((r.weight() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rule_empty_antecedents_panics() {
        let result = std::panic::catch_unwind(|| {
            Rule::new(
                vec![],
                Connector::And,
                vec![("y".to_string(), "b".to_string())],
            )
        });
        assert!(result.is_err());
    }

    #[test]
    fn rule_empty_consequents_panics() {
        let result = std::panic::catch_unwind(|| {
            Rule::new(vec![Antecedent::new("x", "a")], Connector::And, vec![])
        });
        assert!(result.is_err());
    }

    #[test]
    fn rule_not_antecedent_fires_higher_when_term_low() {
        let (inputs, vars) = make_vars();
        let r = Rule::new(
            vec![Antecedent::negated("temperature", "cold")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        let alpha = r.firing_strength(&inputs, &vars);
        assert!((alpha - 0.2).abs() < 1e-6);
    }

    #[test]
    fn rule_not_antecedent_display() {
        let r = Rule::new(
            vec![Antecedent::negated("temperature", "cold")],
            Connector::And,
            vec![("speed".to_string(), "fast".to_string())],
        );
        let s = r.to_string();
        assert!(s.contains("IS NOT"));
    }

    #[test]
    fn rule_weight_scales_firing() {
        let (inputs, vars) = make_vars();
        let r = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        )
        .with_weight(0.5);
        let alpha = r.firing_strength(&inputs, &vars);
        assert!((alpha - 0.4).abs() < 1e-6);
    }

    #[test]
    fn rule_weight_zero_never_fires() {
        let (inputs, vars) = make_vars();
        let r = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        )
        .with_weight(0.0);
        assert_eq!(r.firing_strength(&inputs, &vars), 0.0);
    }

    #[test]
    fn rule_weight_invalid_panics() {
        let result = std::panic::catch_unwind(|| {
            Rule::new(
                vec![Antecedent::new("x", "a")],
                Connector::And,
                vec![("y".to_string(), "b".to_string())],
            )
            .with_weight(1.5)
        });
        assert!(result.is_err());
    }

    #[test]
    fn rule_weight_displayed_when_not_one() {
        let r = Rule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        )
        .with_weight(0.75);
        assert!(r.to_string().contains("w=0.75"));
    }

    #[test]
    fn rule_weight_not_displayed_when_one() {
        let r = Rule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        assert!(!r.to_string().contains('w'));
    }

    #[test]
    fn rule_multiple_consequents() {
        let r = Rule::new(
            vec![Antecedent::new("temperature", "hot")],
            Connector::And,
            vec![
                ("fan".to_string(), "fast".to_string()),
                ("light".to_string(), "bright".to_string()),
            ],
        );
        assert_eq!(r.consequents().len(), 2);
    }

    #[test]
    fn rule_multi_consequent_display() {
        let r = Rule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![
                ("y".to_string(), "b".to_string()),
                ("z".to_string(), "c".to_string()),
            ],
        );
        let s = r.to_string();
        assert!(s.contains("y IS b"));
        assert!(s.contains("z IS c"));
    }

    #[test]
    fn and_uses_min() {
        let (mut inputs, mut vars) = make_vars();
        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0, 100.0, 1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        inputs.insert("humidity".to_string(), 20.0);
        vars.insert("humidity".to_string(), hum);

        let r = Rule::new(
            vec![
                Antecedent::new("temperature", "cold"),
                Antecedent::new("humidity", "low"),
            ],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        assert!((r.firing_strength(&inputs, &vars) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn or_uses_max() {
        let (mut inputs, mut vars) = make_vars();
        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0, 100.0, 1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        inputs.insert("humidity".to_string(), 20.0);
        vars.insert("humidity".to_string(), hum);

        let r = Rule::new(
            vec![
                Antecedent::new("temperature", "cold"),
                Antecedent::new("humidity", "low"),
            ],
            Connector::Or,
            vec![("y".to_string(), "b".to_string())],
        );
        assert!((r.firing_strength(&inputs, &vars) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn missing_variable_returns_zero() {
        let inputs = BTreeMap::new();
        let vars = BTreeMap::new();
        let r = Rule::new(
            vec![Antecedent::new("missing", "term")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        assert_eq!(r.firing_strength(&inputs, &vars), 0.0);
    }

    #[test]
    fn firing_strength_missing_antecedent_returns_zero() {
        let (inputs, vars) = make_vars();
        let r = Rule::new(
            vec![
                Antecedent::new("temperature", "cold"),
                Antecedent::new("humidity", "low"),
            ],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        assert_eq!(r.firing_strength(&inputs, &vars), 0.0);
    }

    #[test]
    fn firing_strength_missing_term_in_existing_variable_returns_zero() {
        let (inputs, vars) = make_vars();
        let r = Rule::new(
            vec![Antecedent::new("temperature", "freezing")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        assert_eq!(r.firing_strength(&inputs, &vars), 0.0);
    }

    #[test]
    fn firing_strength_all_antecedents_present_works_normally() {
        let (mut inputs, mut vars) = make_vars();
        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0, 100.0, 1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        inputs.insert("humidity".to_string(), 20.0);
        vars.insert("humidity".to_string(), hum);

        let r = Rule::new(
            vec![
                Antecedent::new("temperature", "cold"),
                Antecedent::new("humidity", "low"),
            ],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        assert!((r.firing_strength(&inputs, &vars) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn builder_basic() {
        let r = RuleBuilder::new()
            .when("temperature", "cold")
            .then("speed", "slow")
            .build();
        assert_eq!(r.antecedent_count(), 1);
        assert_eq!(r.consequent_var(), "speed");
    }

    #[test]
    fn builder_and_not() {
        let r = RuleBuilder::new()
            .when("temperature", "hot")
            .and_not("humidity", "low")
            .then("speed", "fast")
            .build();
        assert!(r.antecedents_full()[1].negated);
        assert_eq!(r.connector(), &Connector::And);
    }

    #[test]
    fn builder_or_not() {
        let r = RuleBuilder::new()
            .when("temperature", "hot")
            .or_not("humidity", "low")
            .then("speed", "fast")
            .build();
        assert!(r.antecedents_full()[1].negated);
        assert_eq!(r.connector(), &Connector::Or);
    }

    #[test]
    fn builder_multi_consequent() {
        let r = RuleBuilder::new()
            .when("x", "a")
            .then("y", "b")
            .also("z", "c")
            .build();
        assert_eq!(r.consequents().len(), 2);
    }

    #[test]
    fn builder_weight() {
        let r = RuleBuilder::new()
            .when("x", "a")
            .then("y", "b")
            .weight(0.6)
            .build();
        assert!((r.weight() - 0.6).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "weight must be in [0.0, 1.0]")]
    fn rule_builder_invalid_weight_panics() {
        let _ = RuleBuilder::new()
            .when("x", "a")
            .then("y", "b")
            .weight(1.5)
            .build();
    }

    #[test]
    fn display_contains_if_then() {
        let r = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("speed".to_string(), "slow".to_string())],
        );
        let s = r.to_string();
        assert!(s.contains("IF"));
        assert!(s.contains("THEN"));
    }

    #[test]
    fn display_and_or() {
        let r_and = RuleBuilder::new()
            .when("x", "a")
            .and("y", "b")
            .then("z", "c")
            .build();
        let r_or = RuleBuilder::new()
            .when("x", "a")
            .or("y", "b")
            .then("z", "c")
            .build();
        assert!(r_and.to_string().contains("AND"));
        assert!(r_or.to_string().contains("OR"));
    }

    #[test]
    #[should_panic(expected = "weight must be in [0.0, 1.0]")]
    fn rule_weight_negative_panics() {
        Rule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        )
        .with_weight(-0.1);
    }

    #[test]
    fn is_expression_based_for_flat_rule() {
        let r = Rule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![("y".to_string(), "b".to_string())],
        );
        assert!(!r.is_expression_based());
    }

    #[test]
    fn is_expression_based_for_expression_rule() {
        let expr = Expression::term(Antecedent::new("x", "a"));
        let r = Rule::from_expression(expr, vec![("y".to_string(), "b".to_string())]);
        assert!(r.is_expression_based());
    }

    // ── NEW tests to kill rule.rs mutants ─────────────────────

    #[test]
    fn antecedent_count_more_than_one() {
        let r = Rule::new(
            vec![
                Antecedent::new("temp", "cold"),
                Antecedent::new("hum", "low"),
            ],
            Connector::And,
            vec![("speed".into(), "slow".into())],
        );
        assert_eq!(r.antecedent_count(), 2);
    }

    #[test]
    fn weight_boundary_near_one_does_not_show_weight() {
        // Use a weight exactly 1e-9 away from 1.0 so that
        // (1.0 - w).abs() == 1e-9, which is NOT > 1e-9.
        let w = 1.0 - 1e-9;
        let r = Rule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![("y".into(), "b".into())],
        )
        .with_weight(w);
        let s = r.to_string();
        assert!(
            !s.contains("[w="),
            "weight 1.0-1e-9 should not be displayed (difference not > 1e-9)"
        );
    }

    #[test]
    fn firing_strength_with_weight_not_one() {
        let (inputs, vars) = make_vars();
        // Use weight 0.3, mu_cold = 0.8 => firing = 0.24
        let r = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("y".into(), "b".into())],
        )
        .with_weight(0.3);
        let alpha = r.firing_strength(&inputs, &vars);
        assert!((alpha - 0.24).abs() < 1e-6);
    }

    #[test]
    fn firing_strength_weight_near_zero_must_be_tiny() {
        let (inputs, vars) = make_vars();            // mu_cold = 0.8
        let r = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("y".into(), "b".into())],
        )
        .with_weight(0.001);
        let alpha = r.firing_strength(&inputs, &vars);
        // Com weight 0.001, alpha = 0.8 * 0.001 = 0.0008
        assert!(alpha < 0.01, "Tiny weight must give tiny firing, got {}", alpha);
        assert!(alpha > 0.0,  "Tiny weight must not be zero, got {}", alpha);
    }

    /// Kill * -> / mutant in firing_strength.
    /// With weight=0.5 and mu=0.8, firing = 0.8*0.5 = 0.4.
    /// If mutant changes * to /, firing = 0.8/0.5 = 1.6 clamped to 1.0 -> different.
    #[test]
    fn firing_strength_weight_half_exact() {
        let (inputs, vars) = make_vars(); // mu_cold = 0.8
        let rule = Rule::new(
            vec![Antecedent::new("temperature", "cold")],
            Connector::And,
            vec![("y".into(), "b".into())],
        )
        .with_weight(0.5);
        let fs = rule.firing_strength(&inputs, &vars);
        // 0.8 * 0.5 = 0.4 (clamped stays 0.4)
        assert!((fs - 0.4).abs() < 1e-9,
            "Expected firing 0.4 with weight=0.5, got {}", fs);
    }

}
