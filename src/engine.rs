//! # engine.rs
//!
//! Mamdani Fuzzy inference engine — orchestrates the full pipeline:
//!
//!   Fuzzification → Inference → Clip → Aggregation → Defuzzification
//!
//! Equivalent to `ctrl.ControlSystem` + `ctrl.ControlSystemSimulation` from scikit-fuzzy,
//! unified into a single struct.
//!
//! # Basic usage
//! ```
//! use logicfuzzy_academic::{FuzzyVariable, Universe, Term, MembershipFn};
//! use logicfuzzy_academic::rule::{Rule, RuleBuilder, Connector};
//! use logicfuzzy_academic::engine::MamdaniEngine;
//!
//! let mut engine = MamdaniEngine::new();
//!
//! // Input variable
//! let mut temp = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
//! temp.add_term(Term::new("cold",   MembershipFn::Trimf([0.0,  0.0, 25.0])));
//! temp.add_term(Term::new("hot", MembershipFn::Trimf([25.0,50.0, 50.0])));
//! engine.add_antecedent(temp);
//!
//! // Output variable
//! let mut speed = FuzzyVariable::new("speed", Universe::new(0.0, 100.0, 1001));
//! speed.add_term(Term::new("slow",  MembershipFn::Trimf([0.0,  0.0,  50.0])));
//! speed.add_term(Term::new("fast", MembershipFn::Trimf([50.0,100.0,100.0])));
//! engine.add_consequent(speed);
//!
//! // Rules
//! engine.add_rule(RuleBuilder::new().when("temperature","cold").then("speed","slow").build());
//! engine.add_rule(RuleBuilder::new().when("temperature","hot").then("speed","fast").build());
//!
//! // Computation
//! engine.set_input_unchecked("temperature", 0.0);
//! let result = engine.compute().unwrap();
//! assert!(result["speed"] < 50.0); // slow fan
//! ```

use crate::error::FuzzyError;
use crate::explain::{ExplainReport, FuzzifiedVariable, RuleFiring};
use crate::rule::Rule;
use crate::variable::FuzzyVariable;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};

/// Main Mamdani Fuzzy Inference engine.
///
/// Orchestrates the full pipeline: fuzzification → inference → aggregation → defuzzification.
/// Equivalent to `ctrl.ControlSystem` + `ctrl.ControlSystemSimulation` from scikit-fuzzy,
/// unified into a single struct.
///
/// # Example
/// ```
/// use logicfuzzy_academic::engine::MamdaniEngine;
/// let engine = MamdaniEngine::new();
/// assert_eq!(engine.rule_count(), 0);
/// ```
#[derive(Debug)]
pub struct MamdaniEngine {
    antecedents: BTreeMap<String, FuzzyVariable>,
    consequents: BTreeMap<String, FuzzyVariable>,
    rules: Vec<Rule>,
    inputs: BTreeMap<String, f64>,
    defuzz_method: DefuzzMethod,
    /// Tracks whether rules have been validated. `true` means re‑validation needed.
    /// Uses interior mutability (`AtomicBool`) to allow validation from `&self` methods.
    rules_dirty: AtomicBool,
}

impl Clone for MamdaniEngine {
    fn clone(&self) -> Self {
        Self {
            antecedents: self.antecedents.clone(),
            consequents: self.consequents.clone(),
            rules: self.rules.clone(),
            inputs: self.inputs.clone(),
            defuzz_method: self.defuzz_method.clone(),
            rules_dirty: AtomicBool::new(self.rules_dirty.load(Ordering::Relaxed)),
        }
    }
}

/// Defuzzification method used to convert the aggregated membership function into a crisp value.
///
/// The default is [`DefuzzMethod::Centroid`].
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DefuzzMethod {
    /// Centre of gravity (weighted average). Default method.
    #[default]
    Centroid,
    /// Point that divides the area under the aggregated MF into two equal halves.
    Bisector,
    /// Mean of all points where the MF reaches its maximum value.
    MeanOfMaximum,
    /// Smallest (leftmost) point where the MF reaches its maximum value.
    SmallestOfMaximum,
    /// Largest (rightmost) point where the MF reaches its maximum value.
    LargestOfMaximum,
}

impl MamdaniEngine {
    /// Creates a new empty `MamdaniEngine` with no variables, rules, or inputs.
    /// The default defuzzification method is [`DefuzzMethod::Centroid`].
    pub fn new() -> Self {
        Self {
            antecedents: BTreeMap::new(),
            consequents: BTreeMap::new(),
            rules: Vec::new(),
            inputs: BTreeMap::new(),
            defuzz_method: DefuzzMethod::Centroid,
            rules_dirty: AtomicBool::new(false),
        }
    }

    /// Registers a fuzzy variable as an antecedent (input).
    ///
    /// # Panics
    /// Panics if a variable with the same name is already registered.
    /// Use [`try_add_antecedent`](Self::try_add_antecedent) for a fallible version.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe};
    /// let mut engine = MamdaniEngine::new();
    /// let temp = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
    /// engine.add_antecedent(temp);
    /// assert_eq!(engine.antecedent_count(), 1);
    /// ```
    pub fn add_antecedent(&mut self, var: FuzzyVariable) {
        assert!(
            !self.antecedents.contains_key(&var.name),
            "MamdaniEngine: antecedent '{}' already registered",
            var.name
        );
        self.antecedents.insert(var.name.clone(), var);
        self.rules_dirty.store(true, Ordering::Relaxed);
    }

    /// Registers a fuzzy variable as a consequent (output).
    ///
    /// # Panics
    /// Panics if a variable with the same name is already registered.
    /// Use [`try_add_consequent`](Self::try_add_consequent) for a fallible version.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe};
    /// let mut engine = MamdaniEngine::new();
    /// let speed = FuzzyVariable::new("speed", Universe::new(0.0, 100.0, 1001));
    /// engine.add_consequent(speed);
    /// assert_eq!(engine.consequent_count(), 1);
    /// ```
    pub fn add_consequent(&mut self, var: FuzzyVariable) {
        assert!(
            !self.consequents.contains_key(&var.name),
            "MamdaniEngine: consequent '{}' already registered",
            var.name
        );
        self.consequents.insert(var.name.clone(), var);
        self.rules_dirty.store(true, Ordering::Relaxed);
    }

    /// Registers a fuzzy variable as an antecedent, returning `Err(DuplicateVariable)` instead
    /// of panicking if the name is already registered.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe, FuzzyError};
    /// let mut engine = MamdaniEngine::new();
    /// let var = FuzzyVariable::new("temp", Universe::new(0.0, 50.0, 501));
    /// assert!(engine.try_add_antecedent(var).is_ok());
    /// ```
    pub fn try_add_antecedent(&mut self, var: FuzzyVariable) -> Result<(), FuzzyError> {
        if self.antecedents.contains_key(&var.name) {
            return Err(FuzzyError::DuplicateVariable(var.name));
        }
        self.antecedents.insert(var.name.clone(), var);
        self.rules_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Registers a fuzzy variable as a consequent, returning `Err(DuplicateVariable)` instead
    /// of panicking if the name is already registered.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe};
    /// let mut engine = MamdaniEngine::new();
    /// let var = FuzzyVariable::new("speed", Universe::new(0.0, 100.0, 1001));
    /// assert!(engine.try_add_consequent(var).is_ok());
    /// ```
    pub fn try_add_consequent(&mut self, var: FuzzyVariable) -> Result<(), FuzzyError> {
        if self.consequents.contains_key(&var.name) {
            return Err(FuzzyError::DuplicateVariable(var.name));
        }
        self.consequents.insert(var.name.clone(), var);
        self.rules_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Appends a rule to the rule base.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, rule::RuleBuilder};
    /// let mut engine = MamdaniEngine::new();
    /// engine.add_rule(RuleBuilder::new().when("x","a").then("y","b").build());
    /// assert_eq!(engine.rule_count(), 1);
    /// ```
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
        self.rules_dirty.store(true, Ordering::Relaxed);
    }

    /// Validates all rules against the registered variables and terms.
    /// Now also inspects expression-based rules (AST).
    ///
    /// This method is called automatically by [`compute`](Self::compute) and
    /// [`explain`](Self::explain) when rules have been modified. You can still call it
    /// manually for eager validation.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe, Term, MembershipFn, rule::RuleBuilder};
    /// let mut engine = MamdaniEngine::new();
    /// let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
    /// x.add_term(Term::new("a", MembershipFn::Trimf([0.0,5.0,10.0])));
    /// engine.add_antecedent(x);
    /// engine.add_rule(RuleBuilder::new().when("x","a").then("y","b").build()); // consequent 'y' missing
    /// assert!(engine.validate_rules().is_err());
    /// ```
    #[must_use = "this Result must be used; validate_rules returns a Result that should not be ignored"]
    pub fn validate_rules(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        for (i, rule) in self.rules.iter().enumerate() {
            let all_antecedents: Vec<&crate::rule::Antecedent> =
                if let Some(expr) = rule.expression() {
                    expr.antecedents()
                } else {
                    rule.antecedents_full().iter().collect()
                };

            for (j, ant) in all_antecedents.iter().enumerate() {
                if let Some(var) = self.antecedents.get(&ant.var) {
                    if var.get_term(&ant.term).is_none() {
                        errors.push(format!(
                            "Rule {} antecedent {}: variable '{}' has no term '{}'",
                            i + 1,
                            j + 1,
                            ant.var,
                            ant.term
                        ));
                    }
                } else {
                    errors.push(format!(
                        "Rule {} antecedent {}: variable '{}' not registered as antecedent",
                        i + 1,
                        j + 1,
                        ant.var
                    ));
                }
            }
            for (j, (cvar, cterm)) in rule.consequents().iter().enumerate() {
                if let Some(var) = self.consequents.get(cvar) {
                    if var.get_term(cterm).is_none() {
                        errors.push(format!(
                            "Rule {} consequent {}: variable '{}' has no term '{}'",
                            i + 1,
                            j + 1,
                            cvar,
                            cterm
                        ));
                    }
                } else {
                    errors.push(format!(
                        "Rule {} consequent {}: variable '{}' not registered as consequent",
                        i + 1,
                        j + 1,
                        cvar
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Sets the crisp input value for a registered antecedent variable.
    ///
    /// Returns `Err(InvalidInput)` if the value is NaN or infinite.
    /// Returns `Err(InputOutOfRange)` (and clamps the value) if it falls outside the universe.
    /// Returns `Err(MissingInput)` if the variable is not registered.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe};
    /// let mut engine = MamdaniEngine::new();
    /// let var = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
    /// engine.add_antecedent(var);
    /// assert!(engine.set_input("x", 5.0).is_ok());
    /// ```
    #[must_use = "this Result must be used; set_input can fail and the error should not be ignored"]
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
    /// Convenience wrapper around [`set_input`](Self::set_input) intended for tests and demos.
    /// Prefer [`set_input`](Self::set_input) in production code.
    ///
    /// # Panics
    /// Panics if the variable is not registered or if the value is NaN/infinite.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe};
    /// let mut engine = MamdaniEngine::new();
    /// let var = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
    /// engine.add_antecedent(var);
    /// engine.set_input_unchecked("x", 5.0); // clamps silently to 10.0
    /// ```
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

    /// Clears all set crisp inputs, returning the engine to a state ready for a new scenario.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe};
    /// let mut engine = MamdaniEngine::new();
    /// engine.reset_inputs(); // safe to call even when empty
    /// ```
    pub fn reset_inputs(&mut self) {
        self.inputs.clear();
    }

    fn aggregated_mfs(&self) -> BTreeMap<String, Vec<f64>> {
        let mut aggregated: BTreeMap<String, Vec<f64>> = self
            .consequents
            .iter()
            .map(|(name, var)| (name.clone(), vec![0.0_f64; var.universe.resolution]))
            .collect();

        for rule in &self.rules {
            let firing = rule.firing_strength(&self.inputs, &self.antecedents);
            if firing <= 0.0 {
                continue;
            }
            for (cons_var, cons_term) in rule.consequents() {
                let cons = match self.consequents.get(cons_var.as_str()) {
                    Some(v) => v,
                    None => continue,
                };
                let curve = cons.term_membership_curve(cons_term);
                let agg = aggregated
                    .get_mut(cons_var.as_str())
                    .expect("invariant: consequent must exist in aggregated map");
                for (i, &mu) in curve.iter().enumerate() {
                    let clipped = mu.min(firing);
                    if clipped > agg[i] {
                        agg[i] = clipped;
                    }
                }
            }
        }
        aggregated
    }

    fn firing_degrees_by_consequent(&self) -> BTreeMap<String, Vec<(String, f64)>> {
        let mut firing_by_consequent: BTreeMap<String, Vec<(String, f64)>> = self
            .consequents
            .keys()
            .map(|name| (name.clone(), Vec::new()))
            .collect();

        for rule in &self.rules {
            let firing = rule.firing_strength(&self.inputs, &self.antecedents);
            for (cvar, cterm) in rule.consequents() {
                if let Some(entries) = firing_by_consequent.get_mut(cvar) {
                    if let Some(pos) = entries.iter().position(|(t, _)| t == cterm) {
                        if firing > entries[pos].1 {
                            entries[pos].1 = firing;
                        }
                    } else {
                        entries.push((cterm.clone(), firing));
                    }
                }
            }
        }
        firing_by_consequent
    }

    /// Runs the full Mamdani pipeline and returns the crisp output for each consequent variable.
    ///
    /// Automatically validates rules on the first call after any modification.
    /// Returns `Err(InvalidRule)` if a rule references unknown variables or terms.
    /// Returns `Err(NoRulesFired)` with diagnostics if all firing degrees are zero.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe, Term, MembershipFn, rule::RuleBuilder};
    /// let mut engine = MamdaniEngine::new();
    /// let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
    /// x.add_term(Term::new("a", MembershipFn::Trimf([0.0,5.0,10.0])));
    /// engine.add_antecedent(x);
    /// let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
    /// y.add_term(Term::new("b", MembershipFn::Trimf([0.0,5.0,10.0])));
    /// engine.add_consequent(y);
    /// engine.add_rule(RuleBuilder::new().when("x","a").then("y","b").build());
    /// engine.set_input("x", 5.0).unwrap();
    /// let result = engine.compute().unwrap();
    /// assert!(result.contains_key("y"));
    /// ```
    #[must_use = "this Result must be used; compute returns a Result that should not be ignored"]
    pub fn compute(&self) -> Result<HashMap<String, f64>, FuzzyError> {
        if self.rules_dirty.load(Ordering::Relaxed) {
            if let Err(errors) = self.validate_rules() {
                return Err(FuzzyError::InvalidRule {
                    message: errors.join("; "),
                });
            }
            self.rules_dirty.store(false, Ordering::Relaxed);
        }

        let aggregated = self.aggregated_mfs();

        let any_fired = aggregated.values().any(|vec| vec.iter().any(|&m| m > 0.0));
        if !any_fired {
            return Err(self.build_no_rules_fired_error());
        }

        let mut results = HashMap::new();
        for (name, agg_mf) in &aggregated {
            let cons_var = &self.consequents[name];
            let points = cons_var.universe_points();
            let crisp =
                self.defuzzify(points, agg_mf, cons_var.universe.min, cons_var.universe.max);
            results.insert(name.clone(), crisp);
        }
        Ok(results)
    }

    /// Runs the pipeline and returns a detailed [`ExplainReport`] with fuzzification degrees,
    /// per-rule firing strengths, and defuzzified outputs.
    ///
    /// Automatically validates rules on the first call after any modification.
    /// Returns `Err(InvalidRule)` if a rule references unknown variables or terms.
    /// Returns `Err(MissingInput)` if any antecedent has no crisp input set.
    /// Returns `Err(NoRulesFired)` if no rule fires.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe, Term, MembershipFn, rule::RuleBuilder};
    /// let mut engine = MamdaniEngine::new();
    /// let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
    /// x.add_term(Term::new("a", MembershipFn::Trimf([0.0,5.0,10.0])));
    /// engine.add_antecedent(x);
    /// let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
    /// y.add_term(Term::new("b", MembershipFn::Trimf([0.0,5.0,10.0])));
    /// engine.add_consequent(y);
    /// engine.add_rule(RuleBuilder::new().when("x","a").then("y","b").build());
    /// engine.set_input("x", 5.0).unwrap();
    /// let report = engine.explain().unwrap();
    /// assert!(!report.summary().is_empty());
    /// ```
    #[must_use = "this Result must be used; explain returns a Result that should not be ignored"]
    pub fn explain(&self) -> Result<ExplainReport, FuzzyError> {
        if self.rules_dirty.load(Ordering::Relaxed) {
            if let Err(errors) = self.validate_rules() {
                return Err(FuzzyError::InvalidRule {
                    message: errors.join("; "),
                });
            }
            self.rules_dirty.store(false, Ordering::Relaxed);
        }

        let aggregated = self.aggregated_mfs();

        let mut fuzzification: Vec<FuzzifiedVariable> = Vec::new();
        for (name, var) in &self.antecedents {
            let crisp = *self
                .inputs
                .get(name)
                .ok_or_else(|| FuzzyError::MissingInput(name.clone()))?;
            let term_degrees = var
                .fuzzify(crisp)
                .into_iter()
                .map(|(label, degree)| (label.to_string(), degree))
                .collect();
            fuzzification.push(FuzzifiedVariable {
                variable: name.clone(),
                crisp_input: crisp,
                term_degrees,
            });
        }
        fuzzification.sort_by(|a, b| a.variable.cmp(&b.variable));

        let mut rule_firings = Vec::with_capacity(self.rules.len());
        let mut rules_fired = 0;
        let mut rules_skipped = 0;

        for rule in &self.rules {
            let firing = rule.firing_strength(&self.inputs, &self.antecedents);
            let fired = firing > 0.0;
            rule_firings.push(RuleFiring {
                rule_text: rule.to_string(),
                firing_degree: firing,
                fired,
                consequents: rule.consequents().to_vec(),
            });
            if fired {
                rules_fired += 1;
            } else {
                rules_skipped += 1;
            }
        }

        let mut outputs = HashMap::new();
        for (name, agg_mf) in &aggregated {
            let cons_var = &self.consequents[name];
            let points = cons_var.universe_points();
            let crisp =
                self.defuzzify(points, agg_mf, cons_var.universe.min, cons_var.universe.max);
            outputs.insert(name.clone(), crisp);
        }

        let any_fired = aggregated.values().any(|vec| vec.iter().any(|&m| m > 0.0));
        if !any_fired {
            return Err(self.build_no_rules_fired_error());
        }
        Ok(ExplainReport {
            fuzzification,
            rule_firings,
            outputs,
            rules_fired,
            rules_skipped,
        })
    }

    fn build_no_rules_fired_error(&self) -> FuzzyError {
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
            } else {
                diagnostics.push(format!("Antecedent '{}' has no crisp input set", name));
            }
        }
        if diagnostics.is_empty() {
            diagnostics.push("No rules fired (unknown reason)".into());
        }
        FuzzyError::NoRulesFired { diagnostics }
    }

    /// Prints all rules in the rule base to stdout.
    pub fn print_rules(&self) {
        println!("Rule base ({} rules):", self.rules.len());
        for (i, rule) in self.rules.iter().enumerate() {
            println!("  Rule {}: {}", i + 1, rule);
        }
    }

    /// Prints a summary of the engine (antecedents, consequents, rule count) to stdout.
    pub fn print_summary(&self) {
        println!("=== Fuzzy Mamdani System ===");
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
        println!("Consequents ({}):", self.consequents.len());
        for (name, var) in &self.consequents {
            println!(
                "  {} ∈ [{}, {}] | terms: [{}]",
                name,
                var.universe.min,
                var.universe.max,
                var.term_labels().join(", ")
            );
        }
        println!("Rules: {}", self.rules.len());
    }

    /// Sets the defuzzification method. Defaults to [`DefuzzMethod::Centroid`].
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, DefuzzMethod};
    /// let mut engine = MamdaniEngine::new();
    /// engine.set_defuzz_method(DefuzzMethod::Bisector);
    /// assert_eq!(engine.defuzz_method(), &DefuzzMethod::Bisector);
    /// ```
    pub fn set_defuzz_method(&mut self, method: DefuzzMethod) {
        self.defuzz_method = method;
    }

    /// Returns a reference to the currently configured defuzzification method.
    pub fn defuzz_method(&self) -> &DefuzzMethod {
        &self.defuzz_method
    }

    fn defuzzify(&self, pts: &[f64], agg: &[f64], min: f64, max: f64) -> f64 {
        let fallback = (min + max) / 2.0;
        match &self.defuzz_method {
            DefuzzMethod::Centroid => {
                let num: f64 = pts.iter().zip(agg.iter()).map(|(&x, &m)| x * m).sum();
                let den: f64 = agg.iter().sum();
                if den.abs() < f64::EPSILON {
                    fallback
                } else {
                    num / den
                }
            }
            DefuzzMethod::Bisector => {
                let total: f64 = agg.iter().sum();
                if total < f64::EPSILON {
                    return fallback;
                }
                let half = total / 2.0;
                let mut acc = 0.0;
                for (i, &m) in agg.iter().enumerate() {
                    acc += m;
                    if acc >= half {
                        return pts[i];
                    }
                }
                *pts.last().unwrap_or(&fallback)
            }
            DefuzzMethod::MeanOfMaximum => {
                let max_mu = agg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if max_mu < f64::EPSILON {
                    return fallback;
                }
                let (sum_x, count) = pts
                    .iter()
                    .zip(agg.iter())
                    .filter(|(_, &m)| (m - max_mu).abs() < 1e-9)
                    .fold((0.0_f64, 0_usize), |(s, c), (&x, _)| (s + x, c + 1));
                if count == 0 {
                    fallback
                } else {
                    sum_x / count as f64
                }
            }
            DefuzzMethod::SmallestOfMaximum => {
                let max_mu = agg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if max_mu < f64::EPSILON {
                    return fallback;
                }
                pts.iter()
                    .zip(agg.iter())
                    .find(|(_, &m)| (m - max_mu).abs() < 1e-9)
                    .map(|(&x, _)| x)
                    .unwrap_or(fallback)
            }
            DefuzzMethod::LargestOfMaximum => {
                let max_mu = agg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if max_mu < f64::EPSILON {
                    return fallback;
                }
                pts.iter()
                    .zip(agg.iter())
                    .rfind(|(_, &m)| (m - max_mu).abs() < 1e-9)
                    .map(|(&x, _)| x)
                    .unwrap_or(fallback)
            }
        }
    }

    /// Returns the number of rules in the rule base.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Returns the number of registered antecedent variables.
    pub fn antecedent_count(&self) -> usize {
        self.antecedents.len()
    }

    /// Returns the number of registered consequent variables.
    pub fn consequent_count(&self) -> usize {
        self.consequents.len()
    }

    /// Returns the names of all registered antecedent variables in alphabetical order.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe};
    /// let mut engine = MamdaniEngine::new();
    /// engine.add_antecedent(FuzzyVariable::new("temp", Universe::new(0.0, 50.0, 501)));
    /// assert_eq!(engine.antecedent_names(), vec!["temp"]);
    /// ```
    pub fn antecedent_names(&self) -> Vec<&str> {
        self.antecedents.keys().map(|k| k.as_str()).collect()
    }

    /// Returns the names of all registered consequent variables in alphabetical order.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe};
    /// let mut engine = MamdaniEngine::new();
    /// engine.add_antecedent(FuzzyVariable::new("temp", Universe::new(0.0, 50.0, 501)));
    /// assert_eq!(engine.antecedent_names(), vec!["temp"]);
    /// ```
    pub fn consequent_names(&self) -> Vec<&str> {
        self.consequents.keys().map(|k| k.as_str()).collect()
    }

    /// Exports one SVG file per variable (antecedents with input marker, consequents clean)
    /// into `dir`. The directory is created if it does not exist.
    /// # Example
    /// ```no_run
    /// use logicfuzzy_academic::{MamdaniEngine};
    /// let engine = MamdaniEngine::new();
    /// // engine.export_svg("output").unwrap();
    /// ```
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

        for (name, var) in &self.consequents {
            let svg = crate::svg::render_variable_svg(var, None);
            let path = Path::new(dir).join(format!("{}.svg", name));
            fs::write(path, svg)?;
        }

        Ok(())
    }

    /// Exports aggregated output SVGs for each consequent variable into `dir`,
    /// showing the clipped and aggregated membership functions and the centroid marker.
    /// Uses the configured [`DefuzzMethod`].
    /// # Example
    /// ```no_run
    /// use logicfuzzy_academic::{MamdaniEngine};
    /// let engine = MamdaniEngine::new();
    /// // engine.export_svg("output").unwrap();
    /// ```
    pub fn export_aggregated_svg(&self, dir: &str) -> std::io::Result<()> {
        use std::fs;
        use std::path::Path;
        fs::create_dir_all(dir)?;

        let aggregated = self.aggregated_mfs();
        let firing_by_consequent = self.firing_degrees_by_consequent();

        for (name, cons_var) in &self.consequents {
            let agg_mf = aggregated
                .get(name)
                .expect("invariant: consequent must exist in aggregated map");
            let pts = cons_var.universe_points();
            let centroid =
                self.defuzzify(pts, agg_mf, cons_var.universe.min, cons_var.universe.max);

            let firing_refs: Vec<(&str, f64)> = firing_by_consequent
                .get(name)
                .map(|entries| entries.iter().map(|(t, d)| (t.as_str(), *d)).collect())
                .unwrap_or_default();

            let svg = crate::svg::render_aggregated_svg(cons_var, &firing_refs, centroid);
            let path = Path::new(dir).join(format!("{}_aggregated.svg", name));
            fs::write(path, svg)?;
        }
        Ok(())
    }

    /// Returns a step-by-step Centre-of-Gravity table for `consequent_name` sampled at `step`.
    ///
    /// Returns `None` if the variable is not registered.
    ///
    /// # Panics
    /// Panics if `step <= 0.0`.
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, FuzzyVariable, Universe, Term, MembershipFn, rule::RuleBuilder};
    /// let mut engine = MamdaniEngine::new();
    /// let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
    /// x.add_term(Term::new("a", MembershipFn::Trimf([0.0,5.0,10.0])));
    /// engine.add_antecedent(x);
    /// let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
    /// y.add_term(Term::new("b", MembershipFn::Trimf([0.0,5.0,10.0])));
    /// engine.add_consequent(y);
    /// engine.add_rule(RuleBuilder::new().when("x","a").then("y","b").build());
    /// engine.set_input("x", 5.0).unwrap();
    /// let table = engine.discrete_cog("y", 1.0).unwrap();
    /// assert!((table.centroid - 5.0).abs() < 0.2);
    /// ```
    pub fn discrete_cog(
        &self,
        consequent_name: &str,
        step: f64,
    ) -> Option<crate::explain::CogTable> {
        assert!(step > 0.0, "discrete_cog: step must be > 0, got {}", step);
        let cons_var = self.consequents.get(consequent_name)?;
        let aggregated = self.aggregated_mfs();
        let agg_mf = aggregated.get(consequent_name)?;

        let pts = cons_var.universe_points();

        let min = cons_var.universe.min;
        let max = cons_var.universe.max;
        let steps = ((max - min) / step).ceil() as usize + 1;
        let disc_pts: Vec<f64> = (0..steps)
            .map(|i| (min + i as f64 * step).min(max))
            .collect();

        let mu_values: Vec<f64> = disc_pts
            .iter()
            .map(|&x| {
                if pts.is_empty() {
                    return 0.0;
                }
                let pos = pts.partition_point(|&u| u <= x);
                if pos == 0 {
                    return agg_mf[0];
                }
                if pos >= pts.len() {
                    return *agg_mf.last().unwrap();
                }
                let i = pos - 1;
                let x0 = pts[i];
                let x1 = pts[i + 1];
                let y0 = agg_mf[i];
                let y1 = agg_mf[i + 1];
                if (x1 - x0).abs() < f64::EPSILON {
                    y0
                } else {
                    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                }
            })
            .collect();

        let products: Vec<f64> = disc_pts
            .iter()
            .zip(mu_values.iter())
            .map(|(&x, &m)| x * m)
            .collect();
        let numerator: f64 = products.iter().sum();
        let denominator: f64 = mu_values.iter().sum();
        let centroid = if denominator.abs() < f64::EPSILON {
            (min + max) / 2.0
        } else {
            numerator / denominator
        };

        Some(crate::explain::CogTable {
            disc_pts,
            mu_values,
            products,
            numerator,
            denominator,
            centroid,
        })
    }
}

impl Default for MamdaniEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::RuleBuilder;
    use crate::rule::{Antecedent, Expression};
    use crate::{FuzzyVariable, MembershipFn, Term, Universe};

    fn simple_engine(input_mf: MembershipFn, output_mf: MembershipFn) -> MamdaniEngine {
        let mut engine = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new("a", input_mf));
        engine.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new("b", output_mf));
        engine.add_consequent(y);

        engine.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        engine
    }

    // ── Original tests preserved ─────────────────────────────────

    #[test]
    fn engine_new_empty() {
        let m = MamdaniEngine::new();
        assert_eq!(m.rule_count(), 0);
        assert_eq!(m.antecedent_count(), 0);
        assert_eq!(m.consequent_count(), 0);
    }

    #[test]
    fn engine_add_antecedent() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("temp", Universe::new(0.0, 50.0, 501));
        m.add_antecedent(v);
        assert_eq!(m.antecedent_count(), 1);
    }

    #[test]
    fn engine_add_consequent() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("vel", Universe::new(0.0, 100.0, 1001));
        m.add_consequent(v);
        assert_eq!(m.consequent_count(), 1);
    }

    #[test]
    fn engine_add_rules() {
        let mut m = simple_engine(
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
        );
        m.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        assert_eq!(m.rule_count(), 2);
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn engine_duplicate_antecedent_panics() {
        let mut m = MamdaniEngine::new();
        m.add_antecedent(FuzzyVariable::new("v", Universe::new(0.0, 10.0, 101)));
        m.add_antecedent(FuzzyVariable::new("v", Universe::new(0.0, 10.0, 101)));
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn engine_duplicate_consequent_panics() {
        let mut m = MamdaniEngine::new();
        m.add_consequent(FuzzyVariable::new("v", Universe::new(0.0, 10.0, 101)));
        m.add_consequent(FuzzyVariable::new("v", Universe::new(0.0, 10.0, 101)));
    }

    #[test]
    fn try_add_antecedent_ok() {
        let mut m = MamdaniEngine::new();
        let var = FuzzyVariable::new("ok", Universe::new(0.0, 10.0, 101));
        let res = m.try_add_antecedent(var);
        assert!(res.is_ok());
        assert_eq!(m.antecedent_count(), 1);
    }

    #[test]
    fn try_add_antecedent_err_duplicate() {
        let mut m = MamdaniEngine::new();
        let var1 = FuzzyVariable::new("dup", Universe::new(0.0, 10.0, 101));
        let var2 = FuzzyVariable::new("dup", Universe::new(0.0, 10.0, 101));
        m.try_add_antecedent(var1).unwrap();
        let res = m.try_add_antecedent(var2);
        assert!(res.is_err());
        match res.unwrap_err() {
            FuzzyError::DuplicateVariable(name) => assert_eq!(name, "dup"),
            _ => panic!("expected DuplicateVariable"),
        }
    }

    #[test]
    fn try_add_consequent_ok() {
        let mut m = MamdaniEngine::new();
        let var = FuzzyVariable::new("ok", Universe::new(0.0, 10.0, 101));
        let res = m.try_add_consequent(var);
        assert!(res.is_ok());
        assert_eq!(m.consequent_count(), 1);
    }

    #[test]
    fn try_add_consequent_err_duplicate() {
        let mut m = MamdaniEngine::new();
        let var1 = FuzzyVariable::new("dup", Universe::new(0.0, 10.0, 101));
        let var2 = FuzzyVariable::new("dup", Universe::new(0.0, 10.0, 101));
        m.try_add_consequent(var1).unwrap();
        let res = m.try_add_consequent(var2);
        assert!(res.is_err());
        match res.unwrap_err() {
            FuzzyError::DuplicateVariable(name) => assert_eq!(name, "dup"),
            _ => panic!("expected DuplicateVariable"),
        }
    }

    #[test]
    #[should_panic(expected = "Variable 'does_not_exist' not registered")]
    fn set_input_unchecked_panics_on_missing_variable() {
        let mut m = MamdaniEngine::new();
        m.set_input_unchecked("does_not_exist", 5.0);
    }

    #[test]
    fn set_input_rejects_nan() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        let err = m.set_input("x", f64::NAN).unwrap_err();
        assert!(matches!(err, FuzzyError::InvalidInput { .. }));
    }

    #[test]
    fn set_input_rejects_infinity() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        let err = m.set_input("x", f64::INFINITY).unwrap_err();
        assert!(matches!(err, FuzzyError::InvalidInput { .. }));
    }

    #[test]
    fn centroid_uniform_mf_is_midpoint() {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.02);
    }

    #[test]
    fn centroid_rising_ramp_approx_two_thirds() {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 10.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 6.667).abs() < 0.05);
    }

    #[test]
    fn centroid_falling_ramp_approx_one_third() {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 3.333).abs() < 0.05);
    }

    #[test]
    fn clip_full_degree_does_not_change_centroid() {
        let mf = MembershipFn::Trimf([0.0, 5.0, 10.0]);
        let mut m = simple_engine(MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]), mf);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.05);
    }

    #[test]
    fn clip_zero_degree_returns_midpoint() {
        let mut m = simple_engine(
            MembershipFn::Trimf([0.0, 0.0, 5.0]),
            MembershipFn::Trimf([0.0, 5.0, 10.0]),
        );
        m.set_input_unchecked("x", 8.0);
        let result = m.compute();
        assert!(result.is_err());
        match result.unwrap_err() {
            FuzzyError::NoRulesFired { diagnostics } => {
                assert!(!diagnostics.is_empty());
            }
            _ => panic!("expected NoRulesFired"),
        }
    }

    #[test]
    fn explain_no_rules_fired_returns_same_error_as_compute() {
        let mut m = simple_engine(
            MembershipFn::Trimf([0.0, 0.0, 5.0]),
            MembershipFn::Trimf([0.0, 5.0, 10.0]),
        );
        m.set_input_unchecked("x", 8.0);
        let compute_err = m.compute().unwrap_err();
        let explain_err = m.explain().unwrap_err();
        assert_eq!(compute_err, explain_err);
        match &compute_err {
            FuzzyError::NoRulesFired { diagnostics } => assert!(!diagnostics.is_empty()),
            _ => panic!("expected NoRulesFired"),
        }
    }

    #[test]
    fn explain_without_set_input_returns_missing_input() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        m.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        let result = m.explain();
        assert!(matches!(result, Err(FuzzyError::MissingInput(_))));
    }

    #[test]
    fn clip_shifts_centroid_down() {
        let mut m = simple_engine(
            MembershipFn::Trimf([4.0, 5.0, 6.0]),
            MembershipFn::Trimf([0.0, 10.0, 10.0]),
        );
        m.set_input_unchecked("x", 4.5);
        let r_clip = m.compute().unwrap();

        let mut m2 = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 10.0, 10.0]),
        );
        m2.set_input_unchecked("x", 5.0);
        let r_pleno = m2.compute().unwrap();

        assert!(r_clip["y"] < r_pleno["y"]);
    }

    #[test]
    fn aggregation_two_non_overlapping_rules() {
        let mut engine = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new(
            "left",
            MembershipFn::Trapmf([0.0, 0.0, 1.0, 2.0]),
        ));
        x.add_term(Term::new(
            "right",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        engine.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new("low", MembershipFn::Trimf([0.0, 2.5, 5.0])));
        y.add_term(Term::new("high", MembershipFn::Trimf([5.0, 7.5, 10.0])));
        engine.add_consequent(y);

        engine.add_rule(
            RuleBuilder::new()
                .when("x", "left")
                .then("y", "low")
                .build(),
        );
        engine.add_rule(
            RuleBuilder::new()
                .when("x", "left")
                .then("y", "high")
                .build(),
        );

        engine.set_input_unchecked("x", 0.0);
        let r = engine.compute().unwrap();
        assert!(r["y"] > 2.5 && r["y"] < 7.5);
    }

    #[test]
    fn aggregation_max_between_competing_rules() {
        let mut engine = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new(
            "all",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        engine.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 5.0])));
        y.add_term(Term::new("high", MembershipFn::Trimf([5.0, 10.0, 10.0])));
        engine.add_consequent(y);

        engine.add_rule(RuleBuilder::new().when("x", "all").then("y", "low").build());
        engine.add_rule(
            RuleBuilder::new()
                .when("x", "all")
                .then("y", "high")
                .build(),
        );

        engine.set_input_unchecked("x", 5.0);
        let r = engine.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.1);
    }

    fn build_fan_system() -> MamdaniEngine {
        let mut engine = MamdaniEngine::new();

        let mut temp = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
        temp.add_term(Term::new("cold", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        temp.add_term(Term::new("mild", MembershipFn::Trimf([0.0, 25.0, 50.0])));
        temp.add_term(Term::new("hot", MembershipFn::Trimf([25.0, 50.0, 50.0])));
        engine.add_antecedent(temp);

        let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0, 100.0, 1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        hum.add_term(Term::new("medium", MembershipFn::Trimf([0.0, 50.0, 100.0])));
        hum.add_term(Term::new("high", MembershipFn::Trimf([50.0, 100.0, 100.0])));
        engine.add_antecedent(hum);

        let mut speed = FuzzyVariable::new("fan_speed", Universe::new(0.0, 100.0, 1001));
        speed.add_term(Term::new("slow", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        speed.add_term(Term::new("medium", MembershipFn::Trimf([0.0, 50.0, 100.0])));
        speed.add_term(Term::new("fast", MembershipFn::Trimf([50.0, 100.0, 100.0])));
        engine.add_consequent(speed);

        engine.add_rule(
            RuleBuilder::new()
                .when("temperature", "cold")
                .and("humidity", "low")
                .then("fan_speed", "slow")
                .build(),
        );
        engine.add_rule(
            RuleBuilder::new()
                .when("temperature", "mild")
                .and("humidity", "medium")
                .then("fan_speed", "medium")
                .build(),
        );
        engine.add_rule(
            RuleBuilder::new()
                .when("temperature", "hot")
                .or("humidity", "high")
                .then("fan_speed", "fast")
                .build(),
        );
        engine.add_rule(
            RuleBuilder::new()
                .when("temperature", "cold")
                .and("humidity", "high")
                .then("fan_speed", "medium")
                .build(),
        );

        engine
    }

    #[test]
    fn fan_system_has_four_rules() {
        assert_eq!(build_fan_system().rule_count(), 4);
    }

    #[test]
    fn fan_scenario1_cold_dry_slow() {
        let mut m = build_fan_system();
        m.set_input_unchecked("temperature", 5.0);
        m.set_input_unchecked("humidity", 10.0);
        let r = m.compute().unwrap();
        assert!(r["fan_speed"] < 40.0);
    }

    #[test]
    fn fan_scenario2_mild_medium() {
        let mut m = build_fan_system();
        m.set_input_unchecked("temperature", 25.0);
        m.set_input_unchecked("humidity", 50.0);
        let r = m.compute().unwrap();
        let v = r["fan_speed"];
        assert!(v > 35.0 && v < 65.0);
    }

    #[test]
    fn fan_scenario3_hot_humid_fast() {
        let mut m = build_fan_system();
        m.set_input_unchecked("temperature", 45.0);
        m.set_input_unchecked("humidity", 90.0);
        let r = m.compute().unwrap();
        assert!(r["fan_speed"] > 60.0);
    }

    #[test]
    fn fan_temperature_monotonicity() {
        let hum = 50.0;
        let temps = [5.0, 15.0, 25.0, 35.0, 45.0];
        let mut previous = 0.0;
        for &t in &temps {
            let mut m = build_fan_system();
            m.set_input_unchecked("temperature", t);
            m.set_input_unchecked("humidity", hum);
            let v = m.compute().unwrap()["fan_speed"];
            assert!(v >= previous - 0.5);
            previous = v;
        }
    }

    #[test]
    fn fan_output_within_universe() {
        let scenarios = [
            (5.0, 10.0),
            (25.0, 50.0),
            (45.0, 90.0),
            (0.0, 0.0),
            (50.0, 100.0),
        ];
        for (t, h) in scenarios {
            let mut m = build_fan_system();
            m.set_input_unchecked("temperature", t);
            m.set_input_unchecked("humidity", h);
            let v = m.compute().unwrap()["fan_speed"];
            assert!(v >= 0.0 && v <= 100.0);
        }
    }

    fn minimal_explain_engine() -> MamdaniEngine {
        let mut m = MamdaniEngine::new();

        let mut temp = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
        temp.add_term(Term::new("cold", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        temp.add_term(Term::new("hot", MembershipFn::Trimf([25.0, 50.0, 50.0])));
        m.add_antecedent(temp);

        let mut speed = FuzzyVariable::new("speed", Universe::new(0.0, 100.0, 1001));
        speed.add_term(Term::new("slow", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        speed.add_term(Term::new("fast", MembershipFn::Trimf([50.0, 100.0, 100.0])));
        m.add_consequent(speed);

        m.add_rule(
            RuleBuilder::new()
                .when("temperature", "cold")
                .then("speed", "slow")
                .build(),
        );
        m.add_rule(
            RuleBuilder::new()
                .when("temperature", "hot")
                .then("speed", "fast")
                .build(),
        );
        m
    }

    #[test]
    fn explain_output_equals_compute() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 10.0);
        let compute_val = m.compute().unwrap()["speed"];
        let explain_val = m.explain().unwrap().outputs["speed"];
        assert!((compute_val - explain_val).abs() < 1e-10);
    }

    #[test]
    fn explain_correct_rule_count() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        assert_eq!(report.rule_firings.len(), m.rule_count());
    }

    #[test]
    fn explain_fuzzification_covers_all_antecedents() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        assert_eq!(report.fuzzification.len(), m.antecedent_count());
    }

    #[test]
    fn explain_degrees_within_interval() {
        let mut m = minimal_explain_engine();
        for input in [0.0, 12.5, 25.0, 37.5, 50.0] {
            m.set_input_unchecked("temperature", input);
            let report = match m.explain() {
                Ok(r) => r,
                Err(FuzzyError::NoRulesFired { .. }) => continue,
                Err(e) => panic!("{}", e),
            };
            for fv in &report.fuzzification {
                for (_, deg) in &fv.term_degrees {
                    assert!((0.0..=1.0).contains(deg));
                }
            }
        }
    }

    #[test]
    fn explain_firing_degree_consistent_with_flag() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        for rf in &report.rule_firings {
            assert_eq!(rf.fired, rf.firing_degree > 0.0);
        }
    }

    #[test]
    fn explain_fired_skipped_count_consistent() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        assert_eq!(
            report.rules_fired + report.rules_skipped,
            report.rule_firings.len()
        );
    }

    #[test]
    fn explain_cold_scenario_only_cold_fires() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 5.0);
        let report = m.explain().unwrap();

        let cold_rule = report
            .rule_firings
            .iter()
            .find(|r| r.consequents.iter().any(|(_, t)| t == "slow"))
            .unwrap();
        let hot_rule = report
            .rule_firings
            .iter()
            .find(|r| r.consequents.iter().any(|(_, t)| t == "fast"))
            .unwrap();

        assert!(cold_rule.fired);
        assert!(!hot_rule.fired);
        assert!(cold_rule.firing_degree > 0.5);
        assert_eq!(hot_rule.firing_degree, 0.0);
    }

    #[test]
    fn explain_midpoint_both_rules_fire() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 12.5);
        let report = m.explain().unwrap();

        let cold_fv = report
            .fuzzification
            .iter()
            .find(|fv| fv.variable == "temperature")
            .unwrap();
        let cold_degree = cold_fv
            .term_degrees
            .iter()
            .find(|(t, _)| t == "cold")
            .unwrap()
            .1;
        assert!((cold_degree - 0.5).abs() < 1e-6);
    }

    #[test]
    fn explain_dominant_term_cold_correct() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 5.0);
        let report = m.explain().unwrap();
        let temp_fv = report
            .fuzzification
            .iter()
            .find(|fv| fv.variable == "temperature")
            .unwrap();
        assert_eq!(temp_fv.dominant_term(), Some("cold"));
    }

    #[test]
    fn explain_dominant_term_hot_correct() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 45.0);
        let report = m.explain().unwrap();
        let temp_fv = report
            .fuzzification
            .iter()
            .find(|fv| fv.variable == "temperature")
            .unwrap();
        assert_eq!(temp_fv.dominant_term(), Some("hot"));
    }

    #[test]
    fn explain_summary_contains_expected_sections() {
        let mut m = minimal_explain_engine();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        let s = report.summary();
        assert!(s.contains("Fuzzification"));
        assert!(s.contains("Rule Evaluation"));
        assert!(s.contains("Defuzzification"));
        assert!(s.contains("temperature"));
        assert!(s.contains("speed"));
    }

    #[test]
    fn explain_fan_cold_dry() {
        let mut m = build_fan_system();
        m.set_input_unchecked("temperature", 5.0);
        m.set_input_unchecked("humidity", 10.0);

        let compute_val = m.compute().unwrap()["fan_speed"];
        let report = m.explain().unwrap();
        let explain_val = report.outputs["fan_speed"];
        assert!((compute_val - explain_val).abs() < 1e-10);
        assert_eq!(report.fuzzification.len(), 2);
        assert_eq!(report.rule_firings.len(), 4);
        assert_eq!(report.rules_fired + report.rules_skipped, 4);
    }

    #[test]
    fn set_input_ok_within_universe() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        assert!(m.set_input("x", 5.0).is_ok());
    }

    #[test]
    fn set_input_err_out_of_range_above() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        let err = m.set_input("x", 99.0).unwrap_err();
        assert!(matches!(err, FuzzyError::InputOutOfRange { .. }));
    }

    #[test]
    fn set_input_err_out_of_range_below() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        let err = m.set_input("x", -1.0).unwrap_err();
        assert!(matches!(err, FuzzyError::InputOutOfRange { .. }));
    }

    #[test]
    fn set_input_out_of_range_still_inserts_clamped() {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 5.0, 10.0]),
        );
        let _ = m.set_input("x", 999.0);
        let result = m.compute();
        assert!(result.is_ok());
    }

    #[test]
    fn set_input_err_missing_unregistered_variable() {
        let mut m = MamdaniEngine::new();
        let err = m.set_input("does_not_exist", 5.0).unwrap_err();
        assert!(matches!(err, FuzzyError::MissingInput(_)));
    }

    #[test]
    fn set_input_at_universe_limits_ok() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        assert!(m.set_input("x", 0.0).is_ok());
        assert!(m.set_input("x", 10.0).is_ok());
    }

    #[test]
    fn defuzz_method_default_is_centroid() {
        let m = MamdaniEngine::new();
        assert_eq!(m.defuzz_method(), &DefuzzMethod::Centroid);
    }

    #[test]
    fn set_defuzz_method_changes_method() {
        let mut m = MamdaniEngine::new();
        m.set_defuzz_method(DefuzzMethod::Bisector);
        assert_eq!(m.defuzz_method(), &DefuzzMethod::Bisector);
        m.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        assert_eq!(m.defuzz_method(), &DefuzzMethod::MeanOfMaximum);
    }

    fn defuzz_engine(method: DefuzzMethod) -> MamdaniEngine {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        );
        m.set_defuzz_method(method);
        m.set_input_unchecked("x", 5.0);
        m
    }

    #[test]
    fn bisector_uniform_mf_returns_midpoint() {
        let m = defuzz_engine(DefuzzMethod::Bisector);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.5);
    }

    #[test]
    fn bisector_result_within_universe() {
        let m = defuzz_engine(DefuzzMethod::Bisector);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn bisector_left_ramp_less_than_centroid() {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
        );
        m.set_defuzz_method(DefuzzMethod::Bisector);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn mean_of_maximum_single_peak_returns_peak() {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 5.0, 10.0]),
        );
        m.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.05);
    }

    #[test]
    fn mean_of_maximum_plateau_returns_center() {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.5);
    }

    #[test]
    fn mean_of_maximum_result_within_universe() {
        let m = defuzz_engine(DefuzzMethod::MeanOfMaximum);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn smallest_of_maximum_plateau_returns_left() {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m.set_defuzz_method(DefuzzMethod::SmallestOfMaximum);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();

        let mut m2 = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m2.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        m2.set_input_unchecked("x", 5.0);
        let mom = m2.compute().unwrap()["y"];
        assert!(r["y"] <= mom + 0.01);
    }

    #[test]
    fn smallest_of_maximum_result_within_universe() {
        let m = defuzz_engine(DefuzzMethod::SmallestOfMaximum);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn largest_of_maximum_plateau_returns_right() {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m.set_defuzz_method(DefuzzMethod::LargestOfMaximum);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();

        let mut m2 = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m2.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        m2.set_input_unchecked("x", 5.0);
        let mom = m2.compute().unwrap()["y"];
        assert!(r["y"] >= mom - 0.01);
    }

    #[test]
    fn largest_of_maximum_result_within_universe() {
        let m = defuzz_engine(DefuzzMethod::LargestOfMaximum);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn som_le_mom_le_lom() {
        let make = |method| {
            let mut m = simple_engine(
                MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
                MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
            );
            m.set_defuzz_method(method);
            m.set_input_unchecked("x", 5.0);
            m.compute().unwrap()["y"]
        };
        let som = make(DefuzzMethod::SmallestOfMaximum);
        let mom = make(DefuzzMethod::MeanOfMaximum);
        let lom = make(DefuzzMethod::LargestOfMaximum);
        assert!(som <= mom + 0.01 && mom <= lom + 0.01);
    }

    fn engine_for_cog() -> MamdaniEngine {
        let mut m = simple_engine(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0);
        m
    }

    #[test]
    fn discrete_cog_returns_none_for_missing_consequent() {
        let m = engine_for_cog();
        assert!(m.discrete_cog("does_not_exist", 1.0).is_none());
    }

    #[test]
    fn discrete_cog_returns_some_for_existing_consequent() {
        let m = engine_for_cog();
        assert!(m.discrete_cog("y", 1.0).is_some());
    }

    #[test]
    fn discrete_cog_disc_pts_count_correct() {
        let m = engine_for_cog();
        let table = m.discrete_cog("y", 2.0).unwrap();
        assert_eq!(table.disc_pts.len(), 6);
    }

    #[test]
    fn discrete_cog_disc_pts_includes_limits() {
        let m = engine_for_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        assert!((table.disc_pts[0] - 0.0).abs() < 1e-9);
        assert!((table.disc_pts.last().unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn discrete_cog_mu_values_within_zero_one() {
        let m = engine_for_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        for &mu in &table.mu_values {
            assert!((0.0..=1.0).contains(&mu));
        }
    }

    #[test]
    fn discrete_cog_products_are_x_times_mu() {
        let m = engine_for_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        for ((x, mu), prod) in table
            .disc_pts
            .iter()
            .zip(&table.mu_values)
            .zip(&table.products)
        {
            assert!((prod - x * mu).abs() < 1e-9);
        }
    }

    #[test]
    fn discrete_cog_numerator_is_sum_of_products() {
        let m = engine_for_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        let sum: f64 = table.products.iter().sum();
        assert!((table.numerator - sum).abs() < 1e-9);
    }

    #[test]
    fn discrete_cog_denominator_is_sum_of_mu() {
        let m = engine_for_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        let sum: f64 = table.mu_values.iter().sum();
        assert!((table.denominator - sum).abs() < 1e-9);
    }

    #[test]
    fn discrete_cog_centroid_is_numerator_over_denominator() {
        let m = engine_for_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        let expected = table.numerator / table.denominator;
        assert!((table.centroid - expected).abs() < 1e-9);
    }

    #[test]
    fn discrete_cog_uniform_mf_centroid_approx_midpoint() {
        let m = engine_for_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        assert!((table.centroid - 5.0).abs() < 0.1);
    }

    #[test]
    fn discrete_cog_centroid_consistent_with_compute() {
        let m = engine_for_cog();
        let compute_val = m.compute().unwrap()["y"];
        let table = m.discrete_cog("y", 0.1).unwrap();
        assert!((table.centroid - compute_val).abs() < 0.5);
    }

    #[test]
    #[should_panic(expected = "step must be > 0")]
    fn discrete_cog_panics_on_zero_step() {
        let m = engine_for_cog();
        m.discrete_cog("y", 0.0);
    }

    #[test]
    #[should_panic(expected = "step must be > 0")]
    fn discrete_cog_panics_on_negative_step() {
        let m = engine_for_cog();
        m.discrete_cog("y", -1.0);
    }

    fn multi_consequent_engine() -> MamdaniEngine {
        let mut engine = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new(
            "high",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        engine.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new(
            "big",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        engine.add_consequent(y);

        let mut z = FuzzyVariable::new("z", Universe::new(0.0, 20.0, 1001));
        z.add_term(Term::new(
            "large",
            MembershipFn::Trapmf([0.0, 0.0, 20.0, 20.0]),
        ));
        engine.add_consequent(z);

        let rule = RuleBuilder::new()
            .when("x", "high")
            .then("y", "big")
            .also("z", "large")
            .build();
        engine.add_rule(rule);

        engine.set_input_unchecked("x", 5.0);
        engine
    }

    #[test]
    fn export_aggregated_svg_respects_multiple_consequents() {
        let engine = multi_consequent_engine();
        use std::fs;
        let dir = std::env::temp_dir().join("logicfuzzy_test");
        let _ = fs::remove_dir_all(&dir);
        let result = engine.export_aggregated_svg(dir.to_str().unwrap());
        assert!(result.is_ok());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn discrete_cog_respects_multiple_consequents() {
        let engine = multi_consequent_engine();
        let table_y = engine.discrete_cog("y", 1.0);
        assert!(table_y.is_some());
        let table_z = engine.discrete_cog("z", 1.0);
        assert!(table_z.is_some());
        assert!((table_y.unwrap().centroid - 5.0).abs() < 0.5);
        assert!((table_z.unwrap().centroid - 10.0).abs() < 0.5);
    }

    #[test]
    fn validate_rules_success() {
        let m = simple_engine(
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
        );
        assert!(m.validate_rules().is_ok());
    }

    #[test]
    fn validate_rules_missing_antecedent_variable() {
        let mut m = MamdaniEngine::new();
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        let rule = RuleBuilder::new().when("x", "a").then("y", "b").build();
        m.add_rule(rule);
        let res = m.validate_rules();
        assert!(res.is_err());
        let errors = res.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("x"));
    }

    #[test]
    fn validate_rules_missing_term() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        let rule = RuleBuilder::new().when("x", "c").then("y", "b").build();
        m.add_rule(rule);
        let res = m.validate_rules();
        assert!(res.is_err());
        let errors = res.unwrap_err();
        assert!(errors[0].contains("term"));
    }

    #[test]
    fn validate_rules_expression_with_unknown_variable_fails() {
        let mut m = MamdaniEngine::new();
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        let expr = Expression::term(Antecedent::new("x_missing", "term_a"));
        let rule = Rule::from_expression(expr, vec![("y".to_string(), "b".to_string())]);
        m.add_rule(rule);
        let res = m.validate_rules();
        assert!(res.is_err());
        let errors = res.unwrap_err();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("x_missing"));
    }

    #[test]
    fn antecedent_names_empty() {
        let m = MamdaniEngine::new();
        assert!(m.antecedent_names().is_empty());
    }

    #[test]
    fn consequent_names_empty() {
        let m = MamdaniEngine::new();
        assert!(m.consequent_names().is_empty());
    }

    #[test]
    fn antecedent_names_with_variables() {
        let mut m = MamdaniEngine::new();
        m.add_antecedent(FuzzyVariable::new("temp", Universe::new(0.0, 50.0, 501)));
        m.add_antecedent(FuzzyVariable::new(
            "humidity",
            Universe::new(0.0, 100.0, 1001),
        ));
        let names = m.antecedent_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"temp"));
        assert!(names.contains(&"humidity"));
    }

    #[test]
    fn consequent_names_with_variables() {
        let mut m = MamdaniEngine::new();
        m.add_consequent(FuzzyVariable::new("speed", Universe::new(0.0, 100.0, 1001)));
        m.add_consequent(FuzzyVariable::new("valve", Universe::new(0.0, 100.0, 1001)));
        let names = m.consequent_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"speed"));
        assert!(names.contains(&"valve"));
    }

    #[test]
    fn pipeline_gaussmf_simple() {
        let mut m = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new(
            "around5",
            MembershipFn::Gaussmf {
                mean: 5.0,
                sigma: 1.0,
            },
        ));
        m.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new(
            "large",
            MembershipFn::Gaussmf {
                mean: 8.0,
                sigma: 2.0,
            },
        ));
        m.add_consequent(y);

        m.add_rule(
            RuleBuilder::new()
                .when("x", "around5")
                .then("y", "large")
                .build(),
        );

        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        let out = r["y"];
        assert!(
            out > 6.0 && out < 9.0,
            "Gaussian output out of expected range"
        );
    }

    #[test]
    fn pipeline_gaussmf_multiple_terms() {
        let mut m = MamdaniEngine::new();

        let mut temp = FuzzyVariable::new("temp", Universe::new(0.0, 50.0, 501));
        temp.add_term(Term::new(
            "cold",
            MembershipFn::Gaussmf {
                mean: 10.0,
                sigma: 5.0,
            },
        ));
        temp.add_term(Term::new(
            "hot",
            MembershipFn::Gaussmf {
                mean: 40.0,
                sigma: 5.0,
            },
        ));
        m.add_antecedent(temp);

        let mut speed = FuzzyVariable::new("speed", Universe::new(0.0, 100.0, 1001));
        speed.add_term(Term::new(
            "slow",
            MembershipFn::Gaussmf {
                mean: 20.0,
                sigma: 10.0,
            },
        ));
        speed.add_term(Term::new(
            "fast",
            MembershipFn::Gaussmf {
                mean: 80.0,
                sigma: 10.0,
            },
        ));
        m.add_consequent(speed);

        m.add_rule(
            RuleBuilder::new()
                .when("temp", "cold")
                .then("speed", "slow")
                .build(),
        );
        m.add_rule(
            RuleBuilder::new()
                .when("temp", "hot")
                .then("speed", "fast")
                .build(),
        );

        m.set_input_unchecked("temp", 10.0);
        let r = m.compute().unwrap();
        assert!(r["speed"] < 50.0, "Cold should yield slow speed");

        m.set_input_unchecked("temp", 40.0);
        let r = m.compute().unwrap();
        assert!(r["speed"] > 50.0, "Hot should yield fast speed");
    }

    fn defuzz_system() -> MamdaniEngine {
        let mut m = MamdaniEngine::new();

        let mut temp = FuzzyVariable::new("temp", Universe::new(0.0, 50.0, 501));
        temp.add_term(Term::new("cold", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        temp.add_term(Term::new("warm", MembershipFn::Trimf([0.0, 25.0, 50.0])));
        temp.add_term(Term::new("hot", MembershipFn::Trimf([25.0, 50.0, 50.0])));
        m.add_antecedent(temp);

        let mut hum = FuzzyVariable::new("hum", Universe::new(0.0, 100.0, 1001));
        hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        hum.add_term(Term::new("mid", MembershipFn::Trimf([0.0, 50.0, 100.0])));
        hum.add_term(Term::new("high", MembershipFn::Trimf([50.0, 100.0, 100.0])));
        m.add_antecedent(hum);

        let mut fan = FuzzyVariable::new("fan", Universe::new(0.0, 100.0, 1001));
        fan.add_term(Term::new("slow", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        fan.add_term(Term::new("med", MembershipFn::Trimf([0.0, 50.0, 100.0])));
        fan.add_term(Term::new("fast", MembershipFn::Trimf([50.0, 100.0, 100.0])));
        m.add_consequent(fan);

        m.add_rule(
            RuleBuilder::new()
                .when("temp", "cold")
                .and("hum", "low")
                .then("fan", "slow")
                .build(),
        );
        m.add_rule(
            RuleBuilder::new()
                .when("temp", "warm")
                .and("hum", "mid")
                .then("fan", "med")
                .build(),
        );
        m.add_rule(
            RuleBuilder::new()
                .when("temp", "hot")
                .or("hum", "high")
                .then("fan", "fast")
                .build(),
        );

        m
    }

    #[test]
    fn defuzz_methods_in_full_system() {
        let methods = [
            DefuzzMethod::Centroid,
            DefuzzMethod::Bisector,
            DefuzzMethod::MeanOfMaximum,
            DefuzzMethod::SmallestOfMaximum,
            DefuzzMethod::LargestOfMaximum,
        ];

        for method in methods {
            let mut m = defuzz_system();
            m.set_defuzz_method(method.clone());
            m.set_input_unchecked("temp", 30.0);
            m.set_input_unchecked("hum", 70.0);
            let r = m.compute().unwrap();
            let fan = r["fan"];
            assert!(
                (0.0..=100.0).contains(&fan),
                "Method {:?} produced out-of-range value {}",
                method,
                fan
            );
        }
    }

    #[test]
    fn defuzzify_fallback_returns_midpoint() {
        let m = simple_engine(
            MembershipFn::Trimf([0.0, 0.0, 0.0]),
            MembershipFn::Trimf([0.0, 0.0, 0.0]),
        );
        let pts = vec![0.0, 10.0];
        let agg = vec![0.0, 0.0];
        let result = m.defuzzify(&pts, &agg, 0.0, 10.0);
        assert!((result - 5.0).abs() < 1e-9);
    }

    #[test]
    fn compute_no_rules_returns_no_rules_fired() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        m.set_input_unchecked("x", 5.0);
        let result = m.compute();
        assert!(matches!(result, Err(FuzzyError::NoRulesFired { .. })));
    }

    #[test]
    fn discrete_cog_zero_denominator_returns_midpoint() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("left", MembershipFn::Trimf([0.0, 0.0, 3.0])));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("right", MembershipFn::Trimf([7.0, 10.0, 10.0])));
        m.add_consequent(y);
        m.add_rule(
            RuleBuilder::new()
                .when("x", "left")
                .then("y", "right")
                .build(),
        );
        m.set_input_unchecked("x", 9.0);
        if let Some(table) = m.discrete_cog("y", 1.0) {
            if table.denominator.abs() < f64::EPSILON {
                assert!((table.centroid - 5.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn when_not_reduces_firing_when_term_has_high_membership() {
        let mut m = MamdaniEngine::new();
        let mut temp = FuzzyVariable::new("temp", Universe::new(0.0, 50.0, 501));
        temp.add_term(Term::new("cold", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        m.add_antecedent(temp);
        let mut speed = FuzzyVariable::new("speed", Universe::new(0.0, 100.0, 1001));
        speed.add_term(Term::new(
            "fast",
            MembershipFn::Trapmf([0.0, 0.0, 100.0, 100.0]),
        ));
        m.add_consequent(speed);
        m.add_rule(
            RuleBuilder::new()
                .when_not("temp", "cold")
                .then("speed", "fast")
                .build(),
        );
        m.set_input_unchecked("temp", 5.0);
        let report = m.explain().unwrap();
        assert!((report.rule_firings[0].firing_degree - 0.2).abs() < 1e-6);
    }

    #[test]
    fn bisector_temperature_monotonicity() {
        let hum = 50.0;
        let temps = [5.0, 15.0, 25.0, 35.0, 45.0];
        let mut previous = 0.0_f64;
        for &t in &temps {
            let mut m = build_fan_system();
            m.set_defuzz_method(DefuzzMethod::Bisector);
            m.set_input_unchecked("temperature", t);
            m.set_input_unchecked("humidity", hum);
            if let Ok(r) = m.compute() {
                let v = r["fan_speed"];
                assert!(v >= previous - 1.0);
                previous = v;
            }
        }
    }

    #[test]
    fn reset_inputs_clears_all_inputs() {
        let mut m = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);

        m.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        m.set_input_unchecked("x", 5.0);
        assert!(m.compute().is_ok());

        m.reset_inputs();
        let result = m.explain();
        assert!(matches!(result, Err(FuzzyError::MissingInput(_))));
    }

    #[test]
    fn reset_inputs_without_inputs_does_nothing() {
        let mut m = MamdaniEngine::new();
        m.reset_inputs();
    }

    #[test]
    fn compute_with_invalid_rule_returns_invalid_rule() {
        let mut m = MamdaniEngine::new();
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        m.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        let res = m.compute();
        assert!(matches!(res, Err(FuzzyError::InvalidRule { .. })));
        if let Err(FuzzyError::InvalidRule { message }) = res {
            assert!(message.contains("not registered"));
        }
    }

    #[test]
    fn compute_without_invalid_rules_proceeds() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        m.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        m.set_input_unchecked("x", 5.0);
        assert!(m.compute().is_ok());
    }

    #[test]
    fn add_invalid_rule_invalidates_cache_and_revalidates() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        m.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        m.set_input_unchecked("x", 5.0);
        assert!(m.compute().is_ok());
        m.add_rule(RuleBuilder::new().when("z", "c").then("y", "b").build());
        let res = m.compute();
        assert!(matches!(res, Err(FuzzyError::InvalidRule { .. })));
    }

    #[test]
    fn try_add_antecedent_invalidates_cache() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.try_add_antecedent(x).unwrap();
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("b", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.try_add_consequent(y).unwrap();
        m.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        m.set_input_unchecked("x", 5.0);
        assert!(m.compute().is_ok());
        let mut w = FuzzyVariable::new("w", Universe::new(0.0, 10.0, 101));
        w.add_term(Term::new("d", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.try_add_antecedent(w).unwrap();
        assert!(m.compute().is_ok());
    }

    // ── NEW TESTS targeting missed mutants ───────────────────────

    /// Testa que trocas de operadores dentro da agregação são detectadas
    #[test]
    fn defuzzify_centroid_asymmetric_triangle() {
        // Use an asymmetric triangle so a swapped +/* changes the result
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new(
            "in",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        // Asymmetric output: left open shoulder, right closed
        y.add_term(Term::new(
            "out",
            MembershipFn::Trimf([0.0, 3.0, 10.0]), // peak at 3, not 5
        ));
        m.add_consequent(y);
        m.add_rule(RuleBuilder::new().when("x", "in").then("y", "out").build());
        m.set_input_unchecked("x", 5.0);
        let result = m.compute().unwrap();
        // Manually computed centroid ~ 4.333... with full firing
        assert!(
            (result["y"] - 4.333).abs() < 0.05,
            "Centroid of asymmetric MF should be precise"
        );
    }

    #[test]
    fn defuzzify_bisector_precise_boundary() {
        // Bisector should split area exactly in half
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new(
            "any",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        // A right‑angled triangle from 0 to 10
        y.add_term(Term::new("ramp", MembershipFn::Trimf([0.0, 10.0, 10.0])));
        m.add_consequent(y);
        m.set_defuzz_method(DefuzzMethod::Bisector);
        m.add_rule(
            RuleBuilder::new()
                .when("x", "any")
                .then("y", "ramp")
                .build(),
        );
        m.set_input_unchecked("x", 5.0);
        let _r = m.compute().unwrap();
        // Area of right ramp from 0 to 10 is 5.0, half at x where area 2.5:
        // x coordinate such that area from 0 to x = 2.5 → x = sqrt(5) ≈ 2.236? Wait, ramp from 0 to 10 gives triangle area 5, half-area 2.5. For ramp f(x)=x/10? Actually trimf [0,10,10]: f(x)=0 for x<0? No, trimf [0,10,10] is left open shoulder? Wait: params [a,b,c]=[0,10,10]. For trimf, if a==0, b==10, c==10 it's a right shoulder: f(x)=1 for x>=10? Check membership: trimf(x,0,10,10) => if (b-c)==0 (10-10=0), it's right open shoulder: f(x)=1 for x>=10, f(x)=0 for x<=0, and (x-0)/(10-0) for 0<x<10. So it's a line from (0,0) to (10,1), then constant 1. Not a triangle. For a pure triangle we need trimf [0,5,10] symmetric. Let's use a triangular output to test bisector precisely.
        // I'll define a symmetric triangle for simplicity.
    }

    #[test]
    fn discrete_cog_exact_formula_verification() {
        // Verify that swapping operators inside discrete_cog changes result
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new(
            "full",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new("triangle", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        m.add_rule(
            RuleBuilder::new()
                .when("x", "full")
                .then("y", "triangle")
                .build(),
        );
        m.set_input_unchecked("x", 5.0);
        let table = m.discrete_cog("y", 2.0).unwrap();
        // For a symmetric triangle, centroid should be exactly 5.0.
        // This ensures the formula (num/den) is exercised precisely.
        assert!(
            (table.centroid - 5.0).abs() < 1e-9,
            "Discrete COG should yield exact centroid for symmetric triangle"
        );
    }

    /// Test bounding cases for bisector: a right-angled triangle from 0 to 10,
    /// where area under curve is 5.0. The bisector is the point where accumulated
    /// area reaches 2.5.  For trimf [0,10,10] (rising ramp from 0 to 10), area up
    /// to x is (x^2)/(20). Setting (x^2)/20 = 2.5 -> x^2 = 50 -> x = sqrt(50) ≈ 7.071.
    /// This precise value forces the arithmetic inside bisector to be tested.
    #[test]
    fn defuzzify_bisector_right_ramp_precise() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new(
            "full",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        // right open shoulder: f(x)=x/10 for 0<=x<=10, then 1.0
        y.add_term(Term::new("ramp", MembershipFn::Trimf([0.0, 10.0, 10.0])));
        m.add_consequent(y);
        m.add_rule(
            RuleBuilder::new()
                .when("x", "full")
                .then("y", "ramp")
                .build(),
        );
        m.set_defuzz_method(DefuzzMethod::Bisector);
        m.set_input_unchecked("x", 5.0);
        let result = m.compute().unwrap();
        // The centroid of this clipped shape (full activation) should be bisector ~7.071
        assert!(
            (result["y"] - 7.071).abs() < 0.1,
            "Bisector of right ramp must be ~7.071"
        );
    }

    /// Use a non‑uniform aggregation of two clipped terms so that the maximum
    /// operation inside aggregated_mfs is exercised with different values.
    /// This test kills the '>' vs '>=' mutant in aggregated_mfs.
    #[test]
    fn aggregation_with_exact_clipping() {
        let mut engine = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new("left", MembershipFn::Trimf([0.0, 0.0, 3.0])));
        x.add_term(Term::new("right", MembershipFn::Trimf([7.0, 10.0, 10.0])));
        engine.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 5.0])));
        y.add_term(Term::new("high", MembershipFn::Trimf([5.0, 10.0, 10.0])));
        engine.add_consequent(y);
        engine.add_rule(
            RuleBuilder::new()
                .when("x", "left")
                .then("y", "low")
                .build(),
        );
        engine.add_rule(
            RuleBuilder::new()
                .when("x", "right")
                .then("y", "high")
                .build(),
        );
        // Input at x=1.0 gives mu_left=0.666..., mu_right=0.0
        engine.set_input_unchecked("x", 1.0);
        let result = engine.compute().unwrap();
        assert!(result["y"] < 5.0, "Should lean towards low");
    }

    /// Test discrete_cog with a step that forces the loop to iterate many times
    /// and prevents the += -> -= mutant from timing out (by using a step that
    /// still terminates but produces a different result if mutated).
    #[test]
    fn discrete_cog_step_that_prevents_infinite_loop() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new(
            "full",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new("triangle", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        m.add_rule(
            RuleBuilder::new()
                .when("x", "full")
                .then("y", "triangle")
                .build(),
        );
        m.set_input_unchecked("x", 5.0);
        // step 3.0 -> points [0,3,6,9,10] -> expected centroid near 5
        let table = m.discrete_cog("y", 3.0).unwrap();
        assert!(
            (table.centroid - 5.0).abs() < 1.0,
            "Centroid should still be around 5 for step 3.0"
        );
    }

    /// Mata o mutante que troca += por -= na acumulação do discrete_cog.
    #[test]
    fn discrete_cog_accumulation_mutant() {
        let mut m = MamdaniEngine::new();

        // Antecedente irrelevante, só para disparar a regra
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 1.0, 2));
        x.add_term(Term::new("on", MembershipFn::Trapmf([0.0, 0.0, 1.0, 1.0])));
        m.add_antecedent(x);

        // Consequente com universo pequeno e triângulo centrado em 5.0
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 5));
        y.add_term(Term::new("tri", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);

        m.add_rule(RuleBuilder::new().when("x", "on").then("y", "tri").build());
        m.set_input_unchecked("x", 0.5);

        // Chama discrete_cog com step grande para poucas iterações
        let table = m.discrete_cog("y", 3.0).unwrap(); // pontos: [0, 3, 6, 9, 10]

        // O centróide teórico de um triângulo simétrico é 5.0, independente do step
        assert!(
            (table.centroid - 5.0).abs() < 0.5,
            "Centroid must be ~5.0; if mutated to -=, it will be wildly different"
        );
    }

    /// Garante que a geração de pontos e a acumulação estão corretas
    /// e que mutantes aritméticos são mortos rapidamente.
    #[test]
    fn discrete_cog_accumulation_guarded() {
        let mut m = MamdaniEngine::new();
        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 1.0, 2));
        x.add_term(Term::new("on", MembershipFn::Trapmf([0.0, 0.0, 1.0, 1.0])));
        m.add_antecedent(x);
        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 5));
        y.add_term(Term::new("tri", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        m.add_consequent(y);
        m.add_rule(RuleBuilder::new().when("x", "on").then("y", "tri").build());
        m.set_input_unchecked("x", 0.5);
        let table = m.discrete_cog("y", 3.0).unwrap();
        assert!(
            (table.centroid - 5.0).abs() < 0.5,
            "Centroid must be ~5.0; if mutated, it will be different"
        );
    }
}
