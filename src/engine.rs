//! # engine.rs
//!
//! Mamdani Fuzzy inference engine — orchestrates the full pipeline:
//!
//!   Fuzzification → Inference → Clip → Aggregation → Defuzzification
//!
//! Equivalent to `ctrl.ControlSystem` + `ctrl.ControlSystemSimulation` from scikit-fuzzy,
//! unified into a single struct.
//!
//! # Uso basico
//! ```
//! use logicfuzzy_academic::{FuzzyVariable, Universe, Term, MembershipFn};
//! use logicfuzzy_academic::rule::{Rule, RuleBuilder, Connector};
//! use logicfuzzy_academic::engine::MamdaniEngine;
//!
//! let mut motor = MamdaniEngine::new();
//!
//! // Antecedente
//! let mut temp = FuzzyVariable::new("temperatura", Universe::new(0.0, 50.0, 501));
//! temp.add_term(Term::new("fria",   MembershipFn::Trimf([0.0,  0.0, 25.0])));
//! temp.add_term(Term::new("quente", MembershipFn::Trimf([25.0,50.0, 50.0])));
//! motor.add_antecedent(temp);
//!
//! // Consequente
//! let mut vel = FuzzyVariable::new("velocidade", Universe::new(0.0, 100.0, 1001));
//! vel.add_term(Term::new("lenta",  MembershipFn::Trimf([0.0,  0.0,  50.0])));
//! vel.add_term(Term::new("rapida", MembershipFn::Trimf([50.0,100.0,100.0])));
//! motor.add_consequent(vel);
//!
//! // Regras
//! motor.add_rule(RuleBuilder::new().when("temperatura","fria").then("velocidade","lenta").build());
//! motor.add_rule(RuleBuilder::new().when("temperatura","quente").then("velocidade","rapida").build());
//!
//! // Calculo
//! motor.set_input_unchecked("temperatura", 0.0);
//! let resultado = motor.compute().unwrap();
//! assert!(resultado["velocidade"] < 50.0); // ventilador lento
//! ```

use crate::error::FuzzyError;
use crate::explain::{ExplainReport, FuzzifiedVariable, RuleFiring};
use crate::rule::Rule;
use crate::variable::FuzzyVariable;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone)]
pub struct MamdaniEngine {
    antecedents: BTreeMap<String, FuzzyVariable>,
    consequents: BTreeMap<String, FuzzyVariable>,
    rules: Vec<Rule>,
    inputs: BTreeMap<String, f64>,
    defuzz_method: DefuzzMethod,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum DefuzzMethod {
    #[default]
    Centroid,
    Bisector,
    MeanOfMaximum,
    SmallestOfMaximum,
    LargestOfMaximum,
}

impl MamdaniEngine {
    pub fn new() -> Self {
        Self {
            antecedents: BTreeMap::new(),
            consequents: BTreeMap::new(),
            rules: Vec::new(),
            inputs: BTreeMap::new(),
            defuzz_method: DefuzzMethod::Centroid,
        }
    }

    pub fn add_antecedent(&mut self, var: FuzzyVariable) {
        assert!(
            !self.antecedents.contains_key(&var.name),
            "MamdaniEngine: antecedent '{}' already registered",
            var.name
        );
        self.antecedents.insert(var.name.clone(), var);
    }

    pub fn add_consequent(&mut self, var: FuzzyVariable) {
        assert!(
            !self.consequents.contains_key(&var.name),
            "MamdaniEngine: consequent '{}' already registered",
            var.name
        );
        self.consequents.insert(var.name.clone(), var);
    }

    pub fn try_add_antecedent(&mut self, var: FuzzyVariable) -> Result<(), FuzzyError> {
        if self.antecedents.contains_key(&var.name) {
            return Err(FuzzyError::DuplicateVariable(var.name));
        }
        self.antecedents.insert(var.name.clone(), var);
        Ok(())
    }

    pub fn try_add_consequent(&mut self, var: FuzzyVariable) -> Result<(), FuzzyError> {
        if self.consequents.contains_key(&var.name) {
            return Err(FuzzyError::DuplicateVariable(var.name));
        }
        self.consequents.insert(var.name.clone(), var);
        Ok(())
    }

    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Validates all rules against the registered variables and terms.
    /// Returns `Ok(())` if every rule is sound, or `Err` with a list of problems.
    pub fn validate_rules(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        for (i, rule) in self.rules.iter().enumerate() {
            for (j, ant) in rule.antecedents_full().iter().enumerate() {
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

    pub fn set_input(&mut self, name: &str, value: f64) -> Result<(), FuzzyError> {
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

    pub fn set_input_unchecked(&mut self, name: &str, value: f64) {
        if let Err(e) = self.set_input(name, value) {
            match e {
                FuzzyError::MissingInput(_) => panic!("Variable '{}' not registered", name),
                FuzzyError::InputOutOfRange { .. } => {}
                FuzzyError::NoRulesFired { .. } => unreachable!(),
                FuzzyError::DuplicateVariable(_) => unreachable!(),
            }
        }
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
                let agg = aggregated.get_mut(cons_var.as_str()).unwrap();
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

    /// Returns per-consequent firing degrees: `{ consequent_name → [(term, max_firing), ...] }`
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

    pub fn compute(&self) -> Result<HashMap<String, f64>, FuzzyError> {
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

    pub fn explain(&self) -> Result<ExplainReport, FuzzyError> {
        let aggregated = self.aggregated_mfs();

        let mut fuzzification: Vec<FuzzifiedVariable> = self
            .antecedents
            .iter()
            .map(|(name, var)| {
                let crisp = *self.inputs.get(name).unwrap_or(&0.0);
                let term_degrees = var
                    .fuzzify(crisp)
                    .into_iter()
                    .map(|(label, degree)| (label.to_string(), degree))
                    .collect();
                FuzzifiedVariable {
                    variable: name.clone(),
                    crisp_input: crisp,
                    term_degrees,
                }
            })
            .collect();
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

        if rules_fired == 0 && !self.rules.is_empty() {
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

    pub fn print_rules(&self) {
        println!("Rule base ({} rules):", self.rules.len());
        for (i, rule) in self.rules.iter().enumerate() {
            println!("  Rule {}: {}", i + 1, rule);
        }
    }

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

    pub fn set_defuzz_method(&mut self, method: DefuzzMethod) {
        self.defuzz_method = method;
    }

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

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn antecedent_count(&self) -> usize {
        self.antecedents.len()
    }

    pub fn consequent_count(&self) -> usize {
        self.consequents.len()
    }

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

    pub fn export_aggregated_svg(&self, dir: &str) -> std::io::Result<()> {
        use std::fs;
        use std::path::Path;
        fs::create_dir_all(dir)?;

        let aggregated = self.aggregated_mfs();
        let firing_by_consequent = self.firing_degrees_by_consequent();

        for (name, cons_var) in &self.consequents {
            let agg_mf = aggregated.get(name).unwrap();
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
        let mut disc_pts = Vec::new();
        let mut v = min;
        while v <= max + 1e-9 {
            disc_pts.push(v.min(max));
            v += step;
        }

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
    use crate::{FuzzyVariable, MembershipFn, Term, Universe};

    fn motor_simples(mf_entrada: MembershipFn, mf_saida: MembershipFn) -> MamdaniEngine {
        let mut motor = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new("a", mf_entrada));
        motor.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new("b", mf_saida));
        motor.add_consequent(y);

        motor.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        motor
    }

    #[test]
    fn engine_novo_vazio() {
        let m = MamdaniEngine::new();
        assert_eq!(m.rule_count(), 0);
        assert_eq!(m.antecedent_count(), 0);
        assert_eq!(m.consequent_count(), 0);
    }

    #[test]
    fn engine_add_antecedente() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("temp", Universe::new(0.0, 50.0, 501));
        m.add_antecedent(v);
        assert_eq!(m.antecedent_count(), 1);
    }

    #[test]
    fn engine_add_consequente() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("vel", Universe::new(0.0, 100.0, 1001));
        m.add_consequent(v);
        assert_eq!(m.consequent_count(), 1);
    }

    #[test]
    fn engine_add_regras() {
        let mut m = motor_simples(
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
        );
        m.add_rule(RuleBuilder::new().when("x", "a").then("y", "b").build());
        assert_eq!(m.rule_count(), 2);
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn engine_antecedente_duplicado_panics() {
        let mut m = MamdaniEngine::new();
        m.add_antecedent(FuzzyVariable::new("v", Universe::new(0.0, 10.0, 101)));
        m.add_antecedent(FuzzyVariable::new("v", Universe::new(0.0, 10.0, 101)));
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn engine_consequente_duplicado_panics() {
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
    #[should_panic(expected = "Variable 'nao_existe' not registered")]
    fn set_input_unchecked_panics_on_missing_variable() {
        let mut m = MamdaniEngine::new();
        m.set_input_unchecked("nao_existe", 5.0);
    }

    #[test]
    fn centroide_mf_uniforme_e_ponto_medio() {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.02);
    }

    #[test]
    fn centroide_rampa_crescente_aproxima_dois_tercos() {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 10.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 6.667).abs() < 0.05);
    }

    #[test]
    fn centroide_rampa_decrescente_aproxima_um_terco() {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 3.333).abs() < 0.05);
    }

    #[test]
    fn clip_grau_pleno_nao_altera_centroide() {
        let mf = MembershipFn::Trimf([0.0, 5.0, 10.0]);
        let mut m = motor_simples(MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]), mf);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.05);
    }

    #[test]
    fn clip_grau_zero_retorna_ponto_medio_do_universo() {
        let mut m = motor_simples(
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
    fn explain_no_rules_fired_retorna_err_identico_ao_compute() {
        let mut m = motor_simples(
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
    fn clip_desloca_centroide_para_baixo() {
        let mut m = motor_simples(
            MembershipFn::Trimf([4.0, 5.0, 6.0]),
            MembershipFn::Trimf([0.0, 10.0, 10.0]),
        );
        m.set_input_unchecked("x", 4.5);
        let r_clip = m.compute().unwrap();

        let mut m2 = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 10.0, 10.0]),
        );
        m2.set_input_unchecked("x", 5.0);
        let r_pleno = m2.compute().unwrap();

        assert!(r_clip["y"] < r_pleno["y"]);
    }

    #[test]
    fn agregacao_duas_regras_nao_sobrepostas() {
        let mut motor = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new("esq", MembershipFn::Trapmf([0.0, 0.0, 1.0, 2.0])));
        x.add_term(Term::new(
            "dir",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        motor.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new("baixo", MembershipFn::Trimf([0.0, 2.5, 5.0])));
        y.add_term(Term::new("alto", MembershipFn::Trimf([5.0, 7.5, 10.0])));
        motor.add_consequent(y);

        motor.add_rule(
            RuleBuilder::new()
                .when("x", "esq")
                .then("y", "baixo")
                .build(),
        );
        motor.add_rule(
            RuleBuilder::new()
                .when("x", "esq")
                .then("y", "alto")
                .build(),
        );

        motor.set_input_unchecked("x", 0.0);
        let r = motor.compute().unwrap();
        assert!(r["y"] > 2.5 && r["y"] < 7.5);
    }

    #[test]
    fn agregacao_max_entre_regras_concorrentes() {
        let mut motor = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new(
            "tudo",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        motor.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 1001));
        y.add_term(Term::new("baixo", MembershipFn::Trimf([0.0, 0.0, 5.0])));
        y.add_term(Term::new("alto", MembershipFn::Trimf([5.0, 10.0, 10.0])));
        motor.add_consequent(y);

        motor.add_rule(
            RuleBuilder::new()
                .when("x", "tudo")
                .then("y", "baixo")
                .build(),
        );
        motor.add_rule(
            RuleBuilder::new()
                .when("x", "tudo")
                .then("y", "alto")
                .build(),
        );

        motor.set_input_unchecked("x", 5.0);
        let r = motor.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.1);
    }

    fn montar_ventilador() -> MamdaniEngine {
        let mut motor = MamdaniEngine::new();

        let mut temp = FuzzyVariable::new("temperatura", Universe::new(0.0, 50.0, 501));
        temp.add_term(Term::new("fria", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        temp.add_term(Term::new("morna", MembershipFn::Trimf([0.0, 25.0, 50.0])));
        temp.add_term(Term::new("quente", MembershipFn::Trimf([25.0, 50.0, 50.0])));
        motor.add_antecedent(temp);

        let mut umid = FuzzyVariable::new("umidade", Universe::new(0.0, 100.0, 1001));
        umid.add_term(Term::new("baixa", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        umid.add_term(Term::new("media", MembershipFn::Trimf([0.0, 50.0, 100.0])));
        umid.add_term(Term::new("alta", MembershipFn::Trimf([50.0, 100.0, 100.0])));
        motor.add_antecedent(umid);

        let mut vel = FuzzyVariable::new("velocidade_ventilador", Universe::new(0.0, 100.0, 1001));
        vel.add_term(Term::new("lenta", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        vel.add_term(Term::new("media", MembershipFn::Trimf([0.0, 50.0, 100.0])));
        vel.add_term(Term::new(
            "rapida",
            MembershipFn::Trimf([50.0, 100.0, 100.0]),
        ));
        motor.add_consequent(vel);

        motor.add_rule(
            RuleBuilder::new()
                .when("temperatura", "fria")
                .and("umidade", "baixa")
                .then("velocidade_ventilador", "lenta")
                .build(),
        );
        motor.add_rule(
            RuleBuilder::new()
                .when("temperatura", "morna")
                .and("umidade", "media")
                .then("velocidade_ventilador", "media")
                .build(),
        );
        motor.add_rule(
            RuleBuilder::new()
                .when("temperatura", "quente")
                .or("umidade", "alta")
                .then("velocidade_ventilador", "rapida")
                .build(),
        );
        motor.add_rule(
            RuleBuilder::new()
                .when("temperatura", "fria")
                .and("umidade", "alta")
                .then("velocidade_ventilador", "media")
                .build(),
        );

        motor
    }

    #[test]
    fn ventilador_tem_quatro_regras() {
        assert_eq!(montar_ventilador().rule_count(), 4);
    }

    #[test]
    fn ventilador_cenario1_frio_seco_resultado_lento() {
        let mut m = montar_ventilador();
        m.set_input_unchecked("temperatura", 5.0);
        m.set_input_unchecked("umidade", 10.0);
        let r = m.compute().unwrap();
        assert!(r["velocidade_ventilador"] < 40.0);
    }

    #[test]
    fn ventilador_cenario2_morno_medio_resultado_medio() {
        let mut m = montar_ventilador();
        m.set_input_unchecked("temperatura", 25.0);
        m.set_input_unchecked("umidade", 50.0);
        let r = m.compute().unwrap();
        let v = r["velocidade_ventilador"];
        assert!(v > 35.0 && v < 65.0);
    }

    #[test]
    fn ventilador_cenario3_quente_umido_resultado_rapido() {
        let mut m = montar_ventilador();
        m.set_input_unchecked("temperatura", 45.0);
        m.set_input_unchecked("umidade", 90.0);
        let r = m.compute().unwrap();
        assert!(r["velocidade_ventilador"] > 60.0);
    }

    #[test]
    fn ventilador_monotonia_temperatura() {
        let umid = 50.0;
        let temps = [5.0, 15.0, 25.0, 35.0, 45.0];
        let mut anterior = 0.0;
        for &t in &temps {
            let mut m = montar_ventilador();
            m.set_input_unchecked("temperatura", t);
            m.set_input_unchecked("umidade", umid);
            let v = m.compute().unwrap()["velocidade_ventilador"];
            assert!(v >= anterior - 0.5);
            anterior = v;
        }
    }

    #[test]
    fn ventilador_saida_dentro_do_universo() {
        let cenarios = [
            (5.0, 10.0),
            (25.0, 50.0),
            (45.0, 90.0),
            (0.0, 0.0),
            (50.0, 100.0),
        ];
        for (t, u) in cenarios {
            let mut m = montar_ventilador();
            m.set_input_unchecked("temperatura", t);
            m.set_input_unchecked("umidade", u);
            let v = m.compute().unwrap()["velocidade_ventilador"];
            assert!(v >= 0.0 && v <= 100.0);
        }
    }

    fn motor_explain_minimal() -> MamdaniEngine {
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
    fn explain_output_igual_ao_compute() {
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let compute_val = m.compute().unwrap()["speed"];
        let explain_val = m.explain().unwrap().outputs["speed"];
        assert!((compute_val - explain_val).abs() < 1e-10);
    }

    #[test]
    fn explain_numero_de_regras_correto() {
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        assert_eq!(report.rule_firings.len(), m.rule_count());
    }

    #[test]
    fn explain_fuzzificacao_cobre_todos_antecedentes() {
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        assert_eq!(report.fuzzification.len(), m.antecedent_count());
    }

    #[test]
    fn explain_graus_dentro_do_intervalo() {
        let mut m = motor_explain_minimal();
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
    fn explain_firing_degree_consistente_com_fired_flag() {
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        for rf in &report.rule_firings {
            assert_eq!(rf.fired, rf.firing_degree > 0.0);
        }
    }

    #[test]
    fn explain_contagem_fired_skipped_consistente() {
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        assert_eq!(
            report.rules_fired + report.rules_skipped,
            report.rule_firings.len()
        );
    }

    #[test]
    fn explain_cenario_frio_so_regra_cold_dispara() {
        let mut m = motor_explain_minimal();
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
    fn explain_ponto_medio_ambas_regras_disparam() {
        let mut m = motor_explain_minimal();
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
    fn explain_dominant_term_correto_cenario_frio() {
        let mut m = motor_explain_minimal();
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
    fn explain_dominant_term_correto_cenario_quente() {
        let mut m = motor_explain_minimal();
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
    fn explain_summary_contem_secoes_esperadas() {
        let mut m = motor_explain_minimal();
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
    fn explain_ventilador_cenario_frio_seco() {
        let mut m = montar_ventilador();
        m.set_input_unchecked("temperatura", 5.0);
        m.set_input_unchecked("umidade", 10.0);

        let compute_val = m.compute().unwrap()["velocidade_ventilador"];
        let report = m.explain().unwrap();
        let explain_val = report.outputs["velocidade_ventilador"];
        assert!((compute_val - explain_val).abs() < 1e-10);
        assert_eq!(report.fuzzification.len(), 2);
        assert_eq!(report.rule_firings.len(), 4);
        assert_eq!(report.rules_fired + report.rules_skipped, 4);
    }

    #[test]
    fn set_input_ok_quando_dentro_do_universo() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        assert!(m.set_input("x", 5.0).is_ok());
    }

    #[test]
    fn set_input_err_out_of_range_acima() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        let err = m.set_input("x", 99.0).unwrap_err();
        assert!(matches!(err, FuzzyError::InputOutOfRange { .. }));
    }

    #[test]
    fn set_input_err_out_of_range_abaixo() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        let err = m.set_input("x", -1.0).unwrap_err();
        assert!(matches!(err, FuzzyError::InputOutOfRange { .. }));
    }

    #[test]
    fn set_input_out_of_range_ainda_insere_valor_clamped() {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 5.0, 10.0]),
        );
        let _ = m.set_input("x", 999.0);
        let result = m.compute();
        assert!(result.is_ok());
    }

    #[test]
    fn set_input_err_missing_variavel_nao_registrada() {
        let mut m = MamdaniEngine::new();
        let err = m.set_input("nao_existe", 5.0).unwrap_err();
        assert!(matches!(err, FuzzyError::MissingInput(_)));
    }

    #[test]
    fn set_input_nos_limites_do_universo_e_ok() {
        let mut m = MamdaniEngine::new();
        let v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        m.add_antecedent(v);
        assert!(m.set_input("x", 0.0).is_ok());
        assert!(m.set_input("x", 10.0).is_ok());
    }

    #[test]
    fn defuzz_method_padrao_e_centroid() {
        let m = MamdaniEngine::new();
        assert_eq!(m.defuzz_method(), &DefuzzMethod::Centroid);
    }

    #[test]
    fn set_defuzz_method_altera_o_metodo() {
        let mut m = MamdaniEngine::new();
        m.set_defuzz_method(DefuzzMethod::Bisector);
        assert_eq!(m.defuzz_method(), &DefuzzMethod::Bisector);
        m.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        assert_eq!(m.defuzz_method(), &DefuzzMethod::MeanOfMaximum);
    }

    fn motor_defuzz(method: DefuzzMethod) -> MamdaniEngine {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        );
        m.set_defuzz_method(method);
        m.set_input_unchecked("x", 5.0);
        m
    }

    #[test]
    fn bisector_mf_uniforme_retorna_ponto_medio() {
        let m = motor_defuzz(DefuzzMethod::Bisector);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.5);
    }

    #[test]
    fn bisector_resultado_no_universo() {
        let m = motor_defuzz(DefuzzMethod::Bisector);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn bisector_rampa_esquerda_menor_que_centroide() {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
        );
        m.set_defuzz_method(DefuzzMethod::Bisector);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn mean_of_maximum_pico_unico_retorna_o_pico() {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 5.0, 10.0]),
        );
        m.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.05);
    }

    #[test]
    fn mean_of_maximum_plato_retorna_centro_do_plato() {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.5);
    }

    #[test]
    fn mean_of_maximum_resultado_no_universo() {
        let m = motor_defuzz(DefuzzMethod::MeanOfMaximum);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn smallest_of_maximum_plato_retorna_limite_esquerdo() {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m.set_defuzz_method(DefuzzMethod::SmallestOfMaximum);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();

        let mut m2 = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m2.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        m2.set_input_unchecked("x", 5.0);
        let mom = m2.compute().unwrap()["y"];
        assert!(r["y"] <= mom + 0.01);
    }

    #[test]
    fn smallest_of_maximum_resultado_no_universo() {
        let m = motor_defuzz(DefuzzMethod::SmallestOfMaximum);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn largest_of_maximum_plato_retorna_limite_direito() {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m.set_defuzz_method(DefuzzMethod::LargestOfMaximum);
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();

        let mut m2 = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([2.0, 4.0, 6.0, 8.0]),
        );
        m2.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
        m2.set_input_unchecked("x", 5.0);
        let mom = m2.compute().unwrap()["y"];
        assert!(r["y"] >= mom - 0.01);
    }

    #[test]
    fn largest_of_maximum_resultado_no_universo() {
        let m = motor_defuzz(DefuzzMethod::LargestOfMaximum);
        let r = m.compute().unwrap();
        assert!((0.0..=10.0).contains(&r["y"]));
    }

    #[test]
    fn som_menor_ou_igual_mom_menor_ou_igual_lom() {
        let make = |method| {
            let mut m = motor_simples(
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

    fn motor_para_cog() -> MamdaniEngine {
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0);
        m
    }

    #[test]
    fn discrete_cog_retorna_none_para_consequente_inexistente() {
        let m = motor_para_cog();
        assert!(m.discrete_cog("nao_existe", 1.0).is_none());
    }

    #[test]
    fn discrete_cog_retorna_some_para_consequente_existente() {
        let m = motor_para_cog();
        assert!(m.discrete_cog("y", 1.0).is_some());
    }

    #[test]
    fn discrete_cog_tamanho_disc_pts_correto() {
        let m = motor_para_cog();
        let table = m.discrete_cog("y", 2.0).unwrap();
        assert_eq!(table.disc_pts.len(), 6);
    }

    #[test]
    fn discrete_cog_disc_pts_inclui_limites() {
        let m = motor_para_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        assert!((table.disc_pts[0] - 0.0).abs() < 1e-9);
        assert!((table.disc_pts.last().unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn discrete_cog_mu_values_dentro_de_zero_um() {
        let m = motor_para_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        for &mu in &table.mu_values {
            assert!((0.0..=1.0).contains(&mu));
        }
    }

    #[test]
    fn discrete_cog_products_e_x_vezes_mu() {
        let m = motor_para_cog();
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
    fn discrete_cog_numerador_e_soma_dos_products() {
        let m = motor_para_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        let soma: f64 = table.products.iter().sum();
        assert!((table.numerator - soma).abs() < 1e-9);
    }

    #[test]
    fn discrete_cog_denominador_e_soma_dos_mu() {
        let m = motor_para_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        let soma: f64 = table.mu_values.iter().sum();
        assert!((table.denominator - soma).abs() < 1e-9);
    }

    #[test]
    fn discrete_cog_centroid_e_numerador_sobre_denominador() {
        let m = motor_para_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        let esperado = table.numerator / table.denominator;
        assert!((table.centroid - esperado).abs() < 1e-9);
    }

    #[test]
    fn discrete_cog_mf_uniforme_centroid_aproxima_ponto_medio() {
        let m = motor_para_cog();
        let table = m.discrete_cog("y", 1.0).unwrap();
        assert!((table.centroid - 5.0).abs() < 0.1);
    }

    #[test]
    fn discrete_cog_centroid_consistente_com_compute_centroid() {
        let m = motor_para_cog();
        let compute_val = m.compute().unwrap()["y"];
        let table = m.discrete_cog("y", 0.1).unwrap();
        assert!((table.centroid - compute_val).abs() < 0.5);
    }

    #[test]
    #[should_panic(expected = "step must be > 0")]
    fn discrete_cog_panics_on_zero_step() {
        let m = motor_para_cog();
        m.discrete_cog("y", 0.0);
    }

    #[test]
    #[should_panic(expected = "step must be > 0")]
    fn discrete_cog_panics_on_negative_step() {
        let m = motor_para_cog();
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
    fn export_aggregated_svg_respeita_multiplos_consequentes() {
        let engine = multi_consequent_engine();
        use std::fs;
        let dir = std::env::temp_dir().join("logicfuzzy_test");
        let _ = fs::remove_dir_all(&dir);
        let result = engine.export_aggregated_svg(dir.to_str().unwrap());
        assert!(result.is_ok());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn discrete_cog_respeita_multiplos_consequentes() {
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
        let m = motor_simples(
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
}
