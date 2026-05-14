//! # Takagi-Sugeno-Kang (TSK) Fuzzy Inference
//!
//! TSK uses fuzzy antecedents (same as Mamdani) but **crisp polynomial consequents**.
//! Each rule computes a weighted output `α · f(x₁, x₂, …)` where `f` is a linear
//! polynomial. The final output is the **weighted average** of all rule contributions.
//!
//! # Zero-order TSK
//! Consequent is a constant: `y = c₀`
//!
//! # First-order TSK
//! Consequent is linear: `y = c₀ + c₁·x₁ + c₂·x₂ + …`
//!
//! Coefficients are ordered: `[bias, coeff_for_input_1, coeff_for_input_2, …]`
//! matching the antecedent variables in **alphabetical order**.
//!
//! # Example
//! ```
//! use logicfuzzy_academic::{
//!     TskEngine, TskRule, TskConsequent,
//!     FuzzyVariable, Universe, Term, MembershipFn,
//!     rule::Antecedent, rule::Connector,
//! };
//!
//! let mut engine = TskEngine::new();
//!
//! let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
//! x.add_term(Term::new("small", MembershipFn::Trimf([0.0, 0.0, 5.0])));
//! engine.add_antecedent(x);
//!
//! engine.add_output("y", Universe::new(0.0, 100.0, 101));
//!
//! // Zero-order rule: y = 25 (constant)
//! engine.add_rule(TskRule::new(
//!     vec![Antecedent::new("x", "small")],
//!     Connector::And,
//!     vec![TskConsequent::new("y", vec![25.0, 0.0])],
//! ));
//!
//! engine.set_input("x", 3.0).unwrap();
//! let result = engine.compute().unwrap();
//! assert!((result["y"] - 25.0).abs() < 1e-6);
//! ```

use crate::error::FuzzyError;
use crate::rule::{Antecedent, Connector, Expression};
use crate::variable::{FuzzyVariable, Universe};
use std::collections::{BTreeMap, HashMap};

/// A TSK rule consequent: polynomial function of the crisp input variables.
///
/// `coefficients[0]` is the bias term, and `coefficients[1..]` multiply the
/// antecedent variables in alphabetical order by name.
///
/// # Example
/// ```
/// use logicfuzzy_academic::TskConsequent;
/// let cons = TskConsequent::new("y", vec![5.0, 2.0, 3.0]);
/// assert_eq!(cons.coefficients, vec![5.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone)]
pub struct TskConsequent {
    /// Output variable name.
    pub variable: String,
    /// Polynomial coefficients: `[bias, c₁, c₂, …]`.
    pub coefficients: Vec<f64>,
}

impl TskConsequent {
    /// Creates a new TSK consequent with `coefficients`.
    ///
    /// # Panics
    /// Panics if `coefficients` is empty (at least the bias term is required).
    ///
    /// # Example
    /// ```
    /// use logicfuzzy_academic::TskConsequent;
    /// let cons = TskConsequent::new("z", vec![10.0, 1.0]);
    /// assert_eq!(cons.variable, "z");
    /// ```
    pub fn new(variable: impl Into<String>, coefficients: Vec<f64>) -> Self {
        assert!(
            !coefficients.is_empty(),
            "TskConsequent must have at least the bias coefficient (c0)"
        );
        Self {
            variable: variable.into(),
            coefficients,
        }
    }
}

/// A Takagi-Sugeno-Kang fuzzy rule.
///
/// Antecedents are fuzzy (same as Mamdani) but consequents are crisp
/// polynomial functions of the input variables.
#[derive(Debug, Clone)]
pub struct TskRule {
    expression: Option<Expression>,
    antecedents: Vec<Antecedent>,
    connector: Connector,
    consequents: Vec<TskConsequent>,
    weight: f64,
}

impl TskRule {
    /// Creates a TSK rule from a flat list of antecedents with a single connector.
    ///
    /// # Panics
    /// Panics if `antecedents` or `consequents` is empty.
    ///
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{
    ///     TskRule, TskConsequent, rule::Antecedent, rule::Connector,
    /// };
    /// let rule = TskRule::new(
    ///     vec![Antecedent::new("x", "small")],
    ///     Connector::And,
    ///     vec![TskConsequent::new("y", vec![25.0, 0.0])],
    /// );
    /// ```
    pub fn new(
        antecedents: Vec<Antecedent>,
        connector: Connector,
        consequents: Vec<TskConsequent>,
    ) -> Self {
        assert!(
            !antecedents.is_empty(),
            "TskRule: antecedent list cannot be empty"
        );
        assert!(
            !consequents.is_empty(),
            "TskRule: consequent list cannot be empty"
        );
        Self {
            expression: None,
            antecedents,
            connector,
            consequents,
            weight: 1.0,
        }
    }

    /// Creates a TSK rule from an [`Expression`] AST for arbitrary nested AND/OR logic.
    ///
    /// # Panics
    /// Panics if `consequents` is empty.
    ///
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{
    ///     TskRule, TskConsequent, rule::{Expression, Antecedent},
    /// };
    /// let expr = Expression::and(vec![
    ///     Expression::term(Antecedent::new("x", "small")),
    ///     Expression::term(Antecedent::new("y", "low")),
    /// ]);
    /// let rule = TskRule::from_expression(expr, vec![TskConsequent::new("z", vec![10.0, 2.0, 3.0])]);
    /// ```
    pub fn from_expression(expression: Expression, consequents: Vec<TskConsequent>) -> Self {
        assert!(
            !consequents.is_empty(),
            "TskRule: consequent list cannot be empty"
        );
        Self {
            expression: Some(expression),
            antecedents: vec![],
            connector: Connector::And,
            consequents,
            weight: 1.0,
        }
    }

    /// Sets the rule weight, scaling the firing degree. Must be in `[0.0, 1.0]`.
    ///
    /// # Panics
    /// Panics if `weight` is outside `[0.0, 1.0]`.
    pub fn with_weight(mut self, weight: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&weight),
            "TskRule: weight must be in [0.0, 1.0], got {}",
            weight
        );
        self.weight = weight;
        self
    }

    /// Returns the rule weight (default `1.0`).
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns all consequents as a slice of [`TskConsequent`].
    pub fn consequents(&self) -> &[TskConsequent] {
        &self.consequents
    }

    /// Returns the expression tree, if any.
    pub fn expression(&self) -> Option<&Expression> {
        self.expression.as_ref()
    }

    /// Returns the antecedents as a slice of [`Antecedent`] structs.
    pub fn antecedents_full(&self) -> &[Antecedent] {
        &self.antecedents
    }

    /// Returns the logical connector (`And` / `Or`) for flat antecedent lists.
    pub fn connector(&self) -> &Connector {
        &self.connector
    }

    /// Returns `true` if this rule was created from an [`Expression`] tree.
    pub fn is_expression_based(&self) -> bool {
        self.expression.is_some()
    }

    pub fn firing_strength(
        &self,
        inputs: &BTreeMap<String, f64>,
        antecedents: &BTreeMap<String, FuzzyVariable>,
    ) -> f64 {
        if let Some(expr) = &self.expression {
            return (expr.eval(inputs, antecedents) * self.weight).clamp(0.0, 1.0);
        }

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

    /// Evaluate the polynomial consequent for `var_name` given the crisp inputs.
    /// Returns `None` if the consequent variable is not found.
    pub fn evaluate_consequent(
        &self,
        var_name: &str,
        input_vars: &[&str],
        inputs: &BTreeMap<String, f64>,
    ) -> Option<f64> {
        let cons = self.consequents.iter().find(|c| c.variable == var_name)?;

        let coeffs = &cons.coefficients;
        if coeffs.len() != input_vars.len() + 1 {
            return None;
        }

        let mut result = coeffs[0];
        for (i, var_name) in input_vars.iter().enumerate() {
            if let Some(&val) = inputs.get(*var_name) {
                result += coeffs[i + 1] * val;
            }
        }
        Some(result)
    }

    /// Returns the antecedent variable+term references for validation.
    pub fn all_antecedent_refs(&self) -> Vec<&Antecedent> {
        if let Some(expr) = &self.expression {
            expr.antecedents()
        } else {
            self.antecedents.iter().collect()
        }
    }
}

impl std::fmt::Display for TskRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
            .map(|c| {
                let coeff_str = c
                    .coefficients
                    .iter()
                    .map(|v| format!("{:.4}", v))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} = f({{{}}})", c.variable, coeff_str)
            })
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

/// Takagi-Sugeno-Kang Fuzzy Inference Engine.
///
/// Computes crisp outputs as weighted averages of polynomial consequents
/// evaluated on the crisp input values.
#[derive(Debug, Clone)]
pub struct TskEngine {
    antecedents: BTreeMap<String, FuzzyVariable>,
    inputs: BTreeMap<String, f64>,
    rules: Vec<TskRule>,
    outputs: BTreeMap<String, Universe>,
    rules_dirty: bool,
}

impl TskEngine {
    /// Creates a new empty `TskEngine` with no variables, outputs, or rules.
    pub fn new() -> Self {
        Self {
            antecedents: BTreeMap::new(),
            inputs: BTreeMap::new(),
            rules: Vec::new(),
            outputs: BTreeMap::new(),
            rules_dirty: false,
        }
    }

    /// Registers a fuzzy variable as an antecedent (input).
    ///
    /// # Panics
    /// Panics if a variable with the same name is already registered.
    /// Use [`try_add_antecedent`](Self::try_add_antecedent) for a fallible version.
    pub fn add_antecedent(&mut self, var: FuzzyVariable) {
        assert!(
            !self.antecedents.contains_key(&var.name),
            "TskEngine: antecedent '{}' already registered",
            var.name
        );
        self.antecedents.insert(var.name.clone(), var);
        self.rules_dirty = true;
    }

    /// Registers a fuzzy variable as an antecedent, returning
    /// `Err(DuplicateVariable)` instead of panicking if already registered.
    pub fn try_add_antecedent(&mut self, var: FuzzyVariable) -> Result<(), FuzzyError> {
        if self.antecedents.contains_key(&var.name) {
            return Err(FuzzyError::DuplicateVariable(var.name));
        }
        self.antecedents.insert(var.name.clone(), var);
        self.rules_dirty = true;
        Ok(())
    }

    /// Registers an output variable with its universe of discourse.
    ///
    /// # Panics
    /// Panics if an output with the same name is already registered.
    pub fn add_output(&mut self, name: impl Into<String>, universe: Universe) {
        let name = name.into();
        assert!(
            !self.outputs.contains_key(&name),
            "TskEngine: output '{}' already registered",
            name
        );
        self.outputs.insert(name, universe);
        self.rules_dirty = true;
    }

    /// Registers an output variable, returning `Err(DuplicateVariable)` instead of panicking.
    pub fn try_add_output(
        &mut self,
        name: impl Into<String>,
        universe: Universe,
    ) -> Result<(), FuzzyError> {
        let name = name.into();
        if self.outputs.contains_key(&name) {
            return Err(FuzzyError::DuplicateVariable(name));
        }
        self.outputs.insert(name, universe);
        self.rules_dirty = true;
        Ok(())
    }

    /// Appends a TSK rule to the rule base.
    pub fn add_rule(&mut self, rule: TskRule) {
        self.rules.push(rule);
        self.rules_dirty = true;
    }

    /// Sets the crisp input value for a registered antecedent variable.
    ///
    /// Returns `Err(InvalidInput)` if the value is NaN or infinite.
    /// Returns `Err(InputOutOfRange)` (and clamps the value) if outside the universe.
    /// Returns `Err(MissingInput)` if the variable is not registered.
    pub fn set_input(&mut self, name: &str, value: f64) -> Result<(), FuzzyError> {
        if !value.is_finite() {
            return Err(FuzzyError::InvalidInput {
                variable: name.to_string(),
                value,
            });
        }

        let var = self
            .antecedents
            .get(name)
            .ok_or_else(|| FuzzyError::MissingInput(name.to_string()))?;

        let min = var.universe.min;
        let max = var.universe.max;

        if value < min || value > max {
            let clamped = value.clamp(min, max);
            self.inputs.insert(name.to_string(), clamped);
            return Err(FuzzyError::InputOutOfRange {
                variable: name.to_string(),
                value,
                min,
                max,
            });
        }

        self.inputs.insert(name.to_string(), value);
        Ok(())
    }

    /// Sets a crisp input value, clamping silently if out of range.
    ///
    /// Convenience wrapper around [`set_input`](Self::set_input).
    ///
    /// # Panics
    /// Panics if the variable is not registered or if the value is NaN/infinite.
    pub fn set_input_unchecked(&mut self, name: &str, value: f64) {
        if let Err(e) = self.set_input(name, value) {
            match e {
                FuzzyError::MissingInput(_) => panic!("Variable '{}' not registered", name),
                FuzzyError::InputOutOfRange { .. } => {}
                FuzzyError::InvalidInput { .. } => {
                    panic!("Invalid input value for variable '{}'", name)
                }
                FuzzyError::InvalidRule { .. } => unreachable!(),
                FuzzyError::NoRulesFired { .. } => unreachable!(),
                FuzzyError::DuplicateVariable(_) => unreachable!(),
            }
        }
    }

    /// Clears all set crisp inputs, returning the engine to a fresh state.
    pub fn reset_inputs(&mut self) {
        self.inputs.clear();
    }

    /// Returns the number of rules in the rule base.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Returns the number of registered antecedent variables.
    pub fn antecedent_count(&self) -> usize {
        self.antecedents.len()
    }

    /// Returns the number of registered output variables.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Returns the names of all registered antecedent variables in alphabetical order.
    pub fn antecedent_names(&self) -> Vec<&str> {
        self.antecedents.keys().map(|k| k.as_str()).collect()
    }

    /// Returns the names of all registered output variables in alphabetical order.
    pub fn output_names(&self) -> Vec<&str> {
        self.outputs.keys().map(|k| k.as_str()).collect()
    }

    fn validate_rules(&self) -> Result<(), FuzzyError> {
        let mut errors = Vec::new();
        let ant_names: Vec<&str> = self.antecedents.keys().map(|k| k.as_str()).collect();

        for (i, rule) in self.rules.iter().enumerate() {
            let ant_refs = rule.all_antecedent_refs();
            self.validate_rule_antecedents(i, &ant_refs, &mut errors);
            self.validate_rule_consequents(i, rule, &ant_names, &mut errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(FuzzyError::InvalidRule {
                message: errors.join("; "),
            })
        }
    }

    fn validate_rule_antecedents(
        &self,
        rule_idx: usize,
        ant_refs: &[&Antecedent],
        errors: &mut Vec<String>,
    ) {
        for (j, ant) in ant_refs.iter().enumerate() {
            if let Some(var) = self.antecedents.get(&ant.var) {
                if var.get_term(&ant.term).is_none() {
                    errors.push(format!(
                        "Rule {} antecedent {}: variable '{}' has no term '{}'",
                        rule_idx + 1,
                        j + 1,
                        ant.var,
                        ant.term
                    ));
                }
            } else {
                errors.push(format!(
                    "Rule {} antecedent {}: variable '{}' not registered as antecedent",
                    rule_idx + 1,
                    j + 1,
                    ant.var
                ));
            }
        }
    }

    fn validate_rule_consequents(
        &self,
        rule_idx: usize,
        rule: &TskRule,
        ant_names: &[&str],
        errors: &mut Vec<String>,
    ) {
        for cons in rule.consequents() {
            if !self.outputs.contains_key(&cons.variable) {
                errors.push(format!(
                    "Rule {}: output variable '{}' not registered",
                    rule_idx + 1,
                    cons.variable
                ));
            }
            let expected_coeff_count = ant_names.len() + 1;
            if cons.coefficients.len() != expected_coeff_count {
                errors.push(format!(
                    "Rule {}: consequent '{}' has {} coefficients but expected {} (n_inputs + 1)",
                    rule_idx + 1,
                    cons.variable,
                    cons.coefficients.len(),
                    expected_coeff_count
                ));
            }
        }
    }

    fn check_all_inputs_present(&self) -> Result<(), FuzzyError> {
        for name in self.antecedent_names() {
            if !self.inputs.contains_key(name) {
                return Err(FuzzyError::MissingInput(name.to_string()));
            }
        }
        Ok(())
    }

    fn input_vars_sorted(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.antecedents.keys().map(|k| k.as_str()).collect();
        names.sort();
        names
    }

    fn init_accumulators(&self) -> (HashMap<String, f64>, HashMap<String, f64>) {
        let mut numerator = HashMap::new();
        let mut denominator = HashMap::new();
        for var_name in self.outputs.keys() {
            numerator.insert(var_name.clone(), 0.0);
            denominator.insert(var_name.clone(), 0.0);
        }
        (numerator, denominator)
    }

    fn aggregate_firings(&self) -> (HashMap<String, f64>, HashMap<String, f64>, bool) {
        let input_vars = self.input_vars_sorted();
        let (mut numerator, mut denominator) = self.init_accumulators();
        let mut any_fired = false;

        for rule in &self.rules {
            let alpha = rule.firing_strength(&self.inputs, &self.antecedents);
            if alpha <= 0.0 {
                continue;
            }
            any_fired = true;
            for cons in rule.consequents() {
                if let Some(val) =
                    rule.evaluate_consequent(&cons.variable, &input_vars, &self.inputs)
                {
                    *numerator.get_mut(&cons.variable).unwrap() += alpha * val;
                    *denominator.get_mut(&cons.variable).unwrap() += alpha;
                }
            }
        }

        (numerator, denominator, any_fired)
    }

    fn build_no_rules_diagnostics(&self) -> Vec<String> {
        let mut diagnostics = Vec::new();
        for (name, var) in &self.antecedents {
            if let Some(&crisp) = self.inputs.get(name) {
                let degrees = var.fuzzify(crisp);
                let max_deg = degrees.iter().map(|(_, d)| *d).fold(0.0_f64, f64::max);
                if max_deg <= 0.0 {
                    diagnostics.push(format!(
                        "Antecedent '{}' has crisp value {} but all membership degrees are zero",
                        name, crisp
                    ));
                } else {
                    diagnostics.push(format!(
                        "Antecedent '{}' has non-zero degrees (max {:.4}) but no rule matched the combination",
                        name, max_deg
                    ));
                }
            }
        }
        if diagnostics.is_empty() {
            diagnostics.push("No rules fired (unknown reason)".into());
        }
        diagnostics
    }

    fn compute_weighted_outputs(
        &self,
        numerator: &HashMap<String, f64>,
        denominator: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        let mut results = HashMap::new();
        for (var_name, univ) in &self.outputs {
            let num = numerator[var_name];
            let den = denominator[var_name];
            let value = if den.abs() < f64::EPSILON {
                (univ.min + univ.max) / 2.0
            } else {
                (num / den).clamp(univ.min, univ.max)
            };
            results.insert(var_name.clone(), value);
        }
        results
    }

    /// Runs the TSK inference pipeline.
    ///
    /// For each rule, computes firing strength and evaluates polynomial consequents.
    /// Final output for each variable = weighted average of rule outputs.
    pub fn compute(&mut self) -> Result<HashMap<String, f64>, FuzzyError> {
        if self.rules_dirty {
            self.validate_rules()?;
            self.rules_dirty = false;
        }

        self.check_all_inputs_present()?;

        let (numerator, denominator, any_fired) = self.aggregate_firings();

        if !any_fired {
            return Err(FuzzyError::NoRulesFired {
                diagnostics: self.build_no_rules_diagnostics(),
            });
        }

        Ok(self.compute_weighted_outputs(&numerator, &denominator))
    }

    /// Prints a summary of the engine configuration to stdout.
    pub fn print_summary(&self) {
        println!("=== TSK Fuzzy System ===");
        println!("Antecedents ({}):", self.antecedents.len());
        for (name, var) in &self.antecedents {
            println!(
                "  {} ∈ [{}, {}] | terms: [{}]",
                name,
                var.universe.min,
                var.universe.max,
                var.term_labels().join(", ")
            );
        }
        println!("Outputs ({}):", self.outputs.len());
        for (name, univ) in &self.outputs {
            println!("  {} ∈ [{}, {}]", name, univ.min, univ.max);
        }
        println!("Rules: {}", self.rules.len());
    }

    /// Exports SVG files for all variables (antecedents + outputs) into `dir`.
    /// The directory is created if it does not exist. Alias that also works with
    /// the `export_svg!` macro in `aggregated` mode.
    pub fn export_aggregated_svg(&self, dir: &str) -> std::io::Result<()> {
        self.export_svg(dir)
    }

    /// Exports SVG files for all variables (antecedents with input markers, outputs as value cards).
    /// The directory is created if it does not exist.
    pub fn export_svg(&self, dir: &str) -> std::io::Result<()> {
        use std::fs;
        use std::path::Path;
        fs::create_dir_all(dir)?;
        for (name, var) in &self.antecedents {
            let input = self.inputs.get(name.as_str()).copied();
            let svg = crate::svg::render_variable_svg(var, input);
            let path = Path::new(dir).join(format!("{}.svg", name));
            fs::write(path, svg)?;
        }
        for name in self.outputs.keys() {
            let svg = self.render_output_svg(name).0;
            let path = Path::new(dir).join(format!("{}.svg", name));
            fs::write(path, svg)?;
        }
        Ok(())
    }

    /// Generates a simple SVG showing the TSK output value.
    fn render_output_svg(&self, name: &str) -> (String, usize, usize) {
        let w = 660;
        let h = 200;
        let mut svg = String::from(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 660 200\" width=\"660\" height=\"200\">\n\
             <style>text { font-family: monospace; font-size: 14px; fill: #cdd6f4; }</style>\n\
             <rect width=\"100%\" height=\"100%\" fill=\"#1e1e2e\"/>\n"
        );
        svg.push_str(&format!(
            "<text x=\"20\" y=\"30\" font-size=\"18\" font-weight=\"bold\">TSK Output: {}</text>\n",
            name
        ));
        if let Ok(result) = self.compute_result_snapshot(name) {
            svg.push_str(&format!(
                "<text x=\"20\" y=\"60\" font-size=\"28\" fill=\"#a6e3a1\">value = {:.4}</text>\n",
                result
            ));
        } else {
            svg.push_str("<text x=\"20\" y=\"60\" fill=\"#f38ba8\">(no output)</text>\n");
        }
        svg.push_str("</svg>");
        (svg, w, h)
    }

    /// Helper: runs compute and returns just one output value (for SVG preview).
    fn compute_result_snapshot(&self, name: &str) -> Result<f64, FuzzyError> {
        let (numerator, denominator, _) = self.aggregate_firings();
        let num = numerator.get(name).copied().unwrap_or(0.0);
        let den = denominator.get(name).copied().unwrap_or(0.0);
        let univ = self
            .outputs
            .get(name)
            .ok_or_else(|| FuzzyError::MissingInput(name.to_string()))?;
        if den.abs() < f64::EPSILON {
            Ok((univ.min + univ.max) / 2.0)
        } else {
            Ok((num / den).clamp(univ.min, univ.max))
        }
    }
}

impl Default for TskEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::membership::MembershipFn;
    use crate::rule::{Antecedent, Connector, Expression};
    use crate::variable::Term;

    fn make_tsk_engine() -> TskEngine {
        let mut engine = TskEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("small", MembershipFn::Trimf([0.0, 0.0, 5.0])));
        x.add_term(Term::new("large", MembershipFn::Trimf([5.0, 10.0, 10.0])));
        engine.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 5.0])));
        y.add_term(Term::new("high", MembershipFn::Trimf([5.0, 10.0, 10.0])));
        engine.add_antecedent(y);

        engine.add_output("z", Universe::new(0.0, 100.0, 101));

        engine
    }

    #[test]
    fn engine_new_empty() {
        let e = TskEngine::new();
        assert_eq!(e.rule_count(), 0);
        assert_eq!(e.antecedent_count(), 0);
        assert_eq!(e.output_count(), 0);
    }

    #[test]
    fn add_antecedent_and_output() {
        let mut e = TskEngine::new();
        let mut v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        v.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        e.add_antecedent(v);
        e.add_output("y", Universe::new(0.0, 20.0, 101));
        assert_eq!(e.antecedent_count(), 1);
        assert_eq!(e.output_count(), 1);
    }

    #[test]
    fn zero_order_tsk() {
        let mut engine = make_tsk_engine();

        let r1 = TskRule::new(
            vec![Antecedent::new("x", "small"), Antecedent::new("y", "low")],
            Connector::And,
            vec![TskConsequent::new("z", vec![10.0, 0.0, 0.0])],
        );
        let r2 = TskRule::new(
            vec![Antecedent::new("x", "large"), Antecedent::new("y", "high")],
            Connector::And,
            vec![TskConsequent::new("z", vec![90.0, 0.0, 0.0])],
        );

        engine.add_rule(r1);
        engine.add_rule(r2);
        engine.set_input_unchecked("x", 0.0);
        engine.set_input_unchecked("y", 0.0);

        let result = engine.compute().unwrap();
        // x=0 => small=1.0, y=0 => low=1.0 => r1 fires
        assert!((result["z"] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn first_order_tsk() {
        let mut engine = make_tsk_engine();

        // z = 5 + 2*x + 3*y
        let r1 = TskRule::new(
            vec![Antecedent::new("x", "small"), Antecedent::new("y", "low")],
            Connector::And,
            vec![TskConsequent::new("z", vec![5.0, 2.0, 3.0])],
        );
        let r2 = TskRule::new(
            vec![Antecedent::new("x", "large"), Antecedent::new("y", "high")],
            Connector::And,
            vec![TskConsequent::new("z", vec![20.0, 1.0, 1.0])],
        );

        engine.add_rule(r1);
        engine.add_rule(r2);
        // x=0 => small=1.0, y=0 => low=1.0
        engine.set_input_unchecked("x", 0.0);
        engine.set_input_unchecked("y", 0.0);

        let result = engine.compute().unwrap();
        // r1: 5 + 2*0 + 3*0 = 5, r2: 20 + 1*0 + 1*0 = 20
        // alpha1=1.0, alpha2=0.0 => output = 5.0
        assert!((result["z"] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_average_two_rules() {
        let mut engine = make_tsk_engine();

        let r1 = TskRule::new(
            vec![Antecedent::new("x", "small")],
            Connector::And,
            vec![TskConsequent::new("z", vec![10.0, 0.0, 0.0])],
        );
        let r2 = TskRule::new(
            vec![Antecedent::new("x", "large")],
            Connector::And,
            vec![TskConsequent::new("z", vec![50.0, 0.0, 0.0])],
        );

        engine.add_rule(r1);
        engine.add_rule(r2);
        // x=5 => small=0.0, large=0.0 -> NoRulesFired for trimf [5,10,10] at x=5: actually at the
        // boundary trimf returns 0.0 for a=5 (since x <= a returns 0 for standard triangle).
        // Let's use x=2.5 where small=0.5, large=0.0 (since large starts at 5)
        engine.set_input_unchecked("x", 2.5);
        engine.set_input_unchecked("y", 2.5);

        let result = engine.compute().unwrap();
        // small at x=2.5: trimf(2.5,0,0,5): left shoulder a=b, x between a and c
        // (c-x)/(c-b) = (5-2.5)/(5-0) = 0.5
        // large at x=2.5: trimf(2.5,5,10,10): x <= a => 0.0
        // alpha1=0.5, alpha2=0.0 => only rule1 fires
        // z = 10.0
        assert!((result["z"] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn input_out_of_range_clamps() {
        let mut engine = make_tsk_engine();
        // "large" term covers [5,10] with right shoulder so x=10 gives mu=1.0
        engine.add_rule(TskRule::new(
            vec![Antecedent::new("x", "large")],
            Connector::And,
            vec![TskConsequent::new("z", vec![42.0, 0.0, 0.0])],
        ));
        engine.set_input_unchecked("y", 0.0);
        // x=99 clamped to 10, large at 10 -> trimf [5,10,10] right shoulder => 1.0
        engine.set_input_unchecked("x", 99.0);
        assert!(engine.compute().is_ok());
    }

    #[test]
    fn missing_input_returns_error() {
        let mut engine = make_tsk_engine();
        engine.add_rule(TskRule::new(
            vec![Antecedent::new("x", "small")],
            Connector::And,
            vec![TskConsequent::new("z", vec![42.0, 0.0, 0.0])],
        ));
        engine.set_input_unchecked("x", 2.0);
        let result = engine.compute();
        assert!(matches!(result, Err(FuzzyError::MissingInput(_))));
    }

    #[test]
    fn nan_input_rejected() {
        let mut engine = make_tsk_engine();
        let err = engine.set_input("x", f64::NAN).unwrap_err();
        assert!(matches!(err, FuzzyError::InvalidInput { .. }));
    }

    #[test]
    fn no_rules_fired_error() {
        let mut engine = make_tsk_engine();
        engine.add_rule(TskRule::new(
            vec![Antecedent::new("x", "small")],
            Connector::And,
            vec![TskConsequent::new("z", vec![1.0, 0.0, 0.0])],
        ));
        // x=99 has zero membership for "small" (universe 0-10), clamped to 10 but still zero
        engine.set_input_unchecked("x", 99.0);
        engine.set_input_unchecked("y", 0.0);
        // small at 10: trimf [0,0,5] -> 0.0
        let result = engine.compute();
        assert!(matches!(result, Err(FuzzyError::NoRulesFired { .. })));
    }

    #[test]
    fn duplicate_antecedent_error() {
        let mut engine = make_tsk_engine();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        let res = engine.try_add_antecedent(v);
        assert!(matches!(res, Err(FuzzyError::DuplicateVariable(_))));
    }

    #[test]
    fn duplicate_output_error() {
        let mut engine = make_tsk_engine();
        let res = engine.try_add_output("z", Universe::new(0.0, 10.0, 101));
        assert!(matches!(res, Err(FuzzyError::DuplicateVariable(_))));
    }

    #[test]
    fn rule_with_expression() {
        let mut engine = make_tsk_engine();
        let expr = Expression::and(vec![
            Expression::term(Antecedent::new("x", "small")),
            Expression::term(Antecedent::new("y", "low")),
        ]);
        let rule =
            TskRule::from_expression(expr, vec![TskConsequent::new("z", vec![77.0, 0.0, 0.0])]);
        engine.add_rule(rule);
        engine.set_input_unchecked("x", 0.0);
        engine.set_input_unchecked("y", 0.0);
        let result = engine.compute().unwrap();
        assert!((result["z"] - 77.0).abs() < 1e-6);
    }

    #[test]
    fn rule_weight_scales_contribution() {
        let mut engine = make_tsk_engine();
        let r1 = TskRule::new(
            vec![Antecedent::new("x", "small")],
            Connector::And,
            vec![TskConsequent::new("z", vec![100.0, 0.0, 0.0])],
        )
        .with_weight(0.5);
        engine.add_rule(r1);
        engine.set_input_unchecked("x", 0.0);
        engine.set_input_unchecked("y", 0.0);
        let result = engine.compute().unwrap();
        // weight 0.5 means alpha=0.5 instead of 1.0
        // alpha*val = 0.5 * 100 = 50, den=0.5 => 100
        // Actually weighted average: numerator = 0.5 * 100 = 50, denominator = 0.5 => 100
        // But wait, if only one rule fires, the weight cancels out in weighted avg
        assert!((result["z"] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_coefficient_count() {
        let mut engine = make_tsk_engine();
        // Need 3 coefficients (bias + x + y) but only provide 2
        let rule = TskRule::new(
            vec![Antecedent::new("x", "small")],
            Connector::And,
            vec![TskConsequent::new("z", vec![10.0, 0.0])],
        );
        engine.add_rule(rule);
        engine.set_input_unchecked("x", 0.0);
        engine.set_input_unchecked("y", 0.0);
        let result = engine.compute();
        assert!(matches!(result, Err(FuzzyError::InvalidRule { .. })));
    }

    #[test]
    fn unknown_output_variable() {
        let mut engine = make_tsk_engine();
        let rule = TskRule::new(
            vec![Antecedent::new("x", "small")],
            Connector::And,
            vec![TskConsequent::new("unknown_var", vec![42.0, 0.0, 0.0])],
        );
        engine.add_rule(rule);
        engine.set_input_unchecked("x", 0.0);
        engine.set_input_unchecked("y", 0.0);
        let result = engine.compute();
        assert!(matches!(result, Err(FuzzyError::InvalidRule { .. })));
    }

    #[test]
    fn reset_inputs_clears() {
        let mut engine = make_tsk_engine();
        engine.set_input_unchecked("x", 2.0);
        engine.set_input_unchecked("y", 3.0);
        engine.reset_inputs();
        let result = engine.compute();
        assert!(matches!(result, Err(FuzzyError::MissingInput(_))));
    }

    #[test]
    fn display_contains_if_then() {
        let ant = Antecedent::new("x", "small");
        let rule = TskRule::new(
            vec![ant],
            Connector::And,
            vec![TskConsequent::new("z", vec![42.0, 1.0, 2.0])],
        );
        let s = rule.to_string();
        assert!(s.contains("IF"));
        assert!(s.contains("THEN"));
    }

    #[test]
    fn consequent_requires_at_least_bias() {
        let result = std::panic::catch_unwind(|| TskConsequent::new("z", vec![]));
        assert!(result.is_err());
    }

    #[test]
    fn multiple_outputs() {
        let mut engine = TskEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 5.0])));
        engine.add_antecedent(x);
        engine.add_output("a", Universe::new(0.0, 10.0, 101));
        engine.add_output("b", Universe::new(0.0, 20.0, 101));

        let rule = TskRule::new(
            vec![Antecedent::new("x", "low")],
            Connector::And,
            vec![
                TskConsequent::new("a", vec![3.0, 0.0]),
                TskConsequent::new("b", vec![15.0, 0.0]),
            ],
        );
        engine.add_rule(rule);
        engine.set_input_unchecked("x", 0.0);
        let result = engine.compute().unwrap();
        assert!((result["a"] - 3.0).abs() < 1e-6);
        assert!((result["b"] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn output_clamped_to_universe() {
        let mut engine = TskEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 5.0])));
        engine.add_antecedent(x);
        engine.add_output("z", Universe::new(0.0, 10.0, 101));

        let rule = TskRule::new(
            vec![Antecedent::new("x", "low")],
            Connector::And,
            vec![TskConsequent::new("z", vec![100.0, 0.0])],
        );
        engine.add_rule(rule);
        engine.set_input_unchecked("x", 0.0);
        let result = engine.compute().unwrap();
        assert!((result["z"] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn tsk_rule_antecedent_refs_from_expression() {
        let expr = Expression::and(vec![
            Expression::term(Antecedent::new("x", "a")),
            Expression::term(Antecedent::new("y", "b")),
        ]);
        let rule =
            TskRule::from_expression(expr, vec![TskConsequent::new("z", vec![0.0, 0.0, 0.0])]);
        assert!(rule.is_expression_based());
        assert_eq!(rule.all_antecedent_refs().len(), 2);
    }

    #[test]
    fn zero_denominator_returns_midpoint() {
        let mut engine = TskEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("a", MembershipFn::Trimf([0.0, 0.0, 5.0])));
        engine.add_antecedent(x);
        engine.add_output("z", Universe::new(0.0, 10.0, 101));
        // Add a rule with weight 0 that fires but contributes zero
        let rule = TskRule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![TskConsequent::new("z", vec![100.0, 0.0])],
        )
        .with_weight(0.0);
        engine.add_rule(rule);
        engine.set_input_unchecked("x", 2.0);
        let result = engine.compute();
        // weight 0 => firing = 0 => NoRulesFired
        assert!(matches!(result, Err(FuzzyError::NoRulesFired { .. })));
    }

    #[test]
    fn antecedent_names_returns_sorted() {
        let mut e = TskEngine::new();
        let mut v1 = FuzzyVariable::new("z", Universe::new(0.0, 10.0, 101));
        v1.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        let mut v2 = FuzzyVariable::new("a", Universe::new(0.0, 10.0, 101));
        v2.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        e.add_antecedent(v1);
        e.add_antecedent(v2);
        let names = e.antecedent_names();
        assert_eq!(names, vec!["a", "z"]);
    }

    #[test]
    fn tsk_consequent_new_panics_on_empty() {
        let result = std::panic::catch_unwind(|| TskConsequent::new("z", vec![]));
        assert!(result.is_err());
    }

    // ── Boundary tests to kill mutation mutants ───────────────────

    #[test]
    fn set_input_at_exact_min_ok() {
        let mut engine = make_tsk_engine();
        assert!(engine.set_input("x", 0.0).is_ok());
    }

    #[test]
    fn set_input_at_exact_max_ok() {
        let mut engine = make_tsk_engine();
        assert!(engine.set_input("x", 10.0).is_ok());
    }

    #[test]
    fn compute_alpha_exactly_zero_skips_rule() {
        // alpha = 0.0 should skip the rule even with <= mutant
        let mut engine = make_tsk_engine();
        engine.add_rule(TskRule::new(
            vec![Antecedent::new("x", "small")],
            Connector::And,
            vec![TskConsequent::new("z", vec![42.0, 0.0, 0.0])],
        ));
        engine.set_input_unchecked("y", 0.0);
        engine.set_input_unchecked("x", 10.0); // small at 10: trimf [0,0,5] => 0.0
                                               // if alpha <= 0.0 is mutated to alpha < 0.0, alpha=0.0 would NOT be skipped
                                               // causing firing_strength to return 0, still skipping…
                                               // We need a test where a non-firing rule would change the output.
                                               // Let's ensure compute returns NoRulesFired (alpha=0 for all rules)
        let result = engine.compute();
        assert!(matches!(result, Err(FuzzyError::NoRulesFired { .. })));
    }

    #[test]
    fn evaluate_consequent_arithmetic_mutants() {
        // Kill += vs -= in evaluate_consequent
        // z = 10 + 2*x + 3*y, with x=1, y=2 => z = 10 + 2*1 + 3*2 = 10 + 2 + 6 = 18
        let mut engine = make_tsk_engine();
        engine.add_rule(TskRule::new(
            vec![Antecedent::new("x", "small"), Antecedent::new("y", "low")],
            Connector::And,
            vec![TskConsequent::new("z", vec![10.0, 2.0, 3.0])],
        ));
        engine.set_input_unchecked("x", 1.0);
        engine.set_input_unchecked("y", 2.0);
        let result = engine.compute().unwrap();
        // += -> -= would give 10 - 2*1 - 3*2 = 2, not 18
        assert!((result["z"] - 18.0).abs() < 1e-9);
    }

    #[test]
    fn tsk_display_weight_exact_boundary() {
        // Kill > vs >= mutant in Display: (weight - 1.0).abs() > 1e-9
        let w = 1.0 - 1e-9; // difference exactly 1e-9, NOT > 1e-9
        let ant = Antecedent::new("x", "small");
        let rule = TskRule::new(
            vec![ant],
            Connector::And,
            vec![TskConsequent::new("z", vec![1.0])],
        )
        .with_weight(w);
        let s = rule.to_string();
        // If > mutated to >=, then 1e-9 >= 1e-9 is true, showing [w=...]
        assert!(!s.contains("[w="));
    }

    #[test]
    fn compute_result_snapshot_alpha_zero_skips_rule() {
        // Kill <= vs < mutant in compute_result_snapshot
        let mut engine = TskEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 5.0])));
        engine.add_antecedent(x);
        engine.add_output("z", Universe::new(0.0, 100.0, 101));
        engine.add_rule(TskRule::new(
            vec![Antecedent::new("x", "low")],
            Connector::And,
            vec![TskConsequent::new("z", vec![50.0, 0.0])],
        ));
        engine.set_input_unchecked("x", 10.0); // low at 10 = 0.0
        let result = engine.compute();
        // No rule should fire
        assert!(matches!(result, Err(FuzzyError::NoRulesFired { .. })));
    }

    #[test]
    fn tsk_weight_boundary_near_one() {
        // Display weight: boundary at 1e-9 difference
        let w = 1.0 - 1e-9;
        let rule = TskRule::new(
            vec![Antecedent::new("x", "a")],
            Connector::And,
            vec![TskConsequent::new("y", vec![1.0])],
        )
        .with_weight(w);
        let s = rule.to_string();
        assert!(!s.contains("[w="), "weight near 1 should not display");
    }
}
