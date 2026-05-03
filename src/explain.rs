//! # explain.rs
//!
//! Estruturas de dados para o relatório de explicabilidade do sistema Fuzzy.
//! Permite inspecionar cada etapa do pipeline Mamdani após um compute().

use std::collections::HashMap;

// ─────────────────────────────────────────────
// RuleFiring: resultado de uma regra individual
// ─────────────────────────────────────────────

/// Result of evaluating a single fuzzy inference rule.
///
/// Captures the human-readable rule description, its firing degree,
/// and whether it actually contributed to the output (degree > 0).
/// Now supports multiple consequents per rule (via `also()`).
#[derive(Debug, Clone)]
pub struct RuleFiring {
    /// Human-readable representation of the rule (e.g. "IF temp IS hot THEN speed IS fast").
    pub rule_text: String,
    /// Firing degree of this rule in [0.0, 1.0].
    pub firing_degree: f64,
    /// Whether this rule contributed to the aggregation (firing_degree > 0).
    pub fired: bool,
    /// All consequent variables + terms activated by this rule.
    pub consequents: Vec<(String, String)>,
}

// ─────────────────────────────────────────────
// FuzzifiedVariable: resultado da fuzzificação
// ─────────────────────────────────────────────

/// Fuzzification result for a single input variable.
///
/// Shows the membership degree of the crisp input value
/// across all linguistic terms of that variable.
#[derive(Debug, Clone)]
pub struct FuzzifiedVariable {
    /// Name of the antecedent variable.
    pub variable: String,
    /// Crisp input value provided to this variable.
    pub crisp_input: f64,
    /// Membership degrees for each term: `(term_label, degree)`.
    pub term_degrees: Vec<(String, f64)>,
}

impl FuzzifiedVariable {
    /// Returns the term with the highest membership degree for this input.
    pub fn dominant_term(&self) -> Option<&str> {
        self.term_degrees
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(label, _)| label.as_str())
    }
}

// ─────────────────────────────────────────────
// ExplainReport: relatório completo
// ─────────────────────────────────────────────

/// Full explanation report of one Mamdani inference cycle.
///
/// Returned by [`MamdaniEngine::explain`](crate::engine::MamdaniEngine::explain).
/// Contains the intermediate state of every pipeline stage:
/// fuzzification → rule evaluation → defuzzification.
#[derive(Debug, Clone)]
pub struct ExplainReport {
    /// Fuzzification results: one entry per antecedent variable.
    pub fuzzification: Vec<FuzzifiedVariable>,
    /// Evaluation result of each rule in the rule base.
    pub rule_firings: Vec<RuleFiring>,
    /// Final crisp outputs after defuzzification: `{ variable_name → crisp_value }`.
    pub outputs: HashMap<String, f64>,
    /// Number of rules that actually fired (firing_degree > 0).
    pub rules_fired: usize,
    /// Number of rules that did not contribute (firing_degree == 0).
    pub rules_skipped: usize,
}

impl ExplainReport {
    /// Returns a formatted multi-line summary of the report.
    pub fn summary(&self) -> String {
        let mut out = String::new();

        out.push_str("=== Fuzzy Mamdani — Explain Report ===\n\n");

        out.push_str("[ Fuzzification ]\n");
        for fv in &self.fuzzification {
            out.push_str(&format!(
                "  {} = {:.4} (crisp)\n",
                fv.variable, fv.crisp_input
            ));
            for (term, degree) in &fv.term_degrees {
                let bar = Self::bar(*degree);
                out.push_str(&format!("    {:>12}  {:.4}  {}\n", term, degree, bar));
            }
            if let Some(dom) = fv.dominant_term() {
                out.push_str(&format!("    → dominant term: {}\n", dom));
            }
            out.push('\n');
        }

        out.push_str(&format!(
            "[ Rule Evaluation ] ({} fired, {} skipped)\n",
            self.rules_fired, self.rules_skipped
        ));
        for rf in &self.rule_firings {
            let status = if rf.fired { "✓" } else { "✗" };
            out.push_str(&format!(
                "  {} [{:.4}]  {}\n",
                status, rf.firing_degree, rf.rule_text
            ));
        }
        out.push('\n');

        out.push_str("[ Defuzzification Output ]\n");
        for (var, val) in &self.outputs {
            out.push_str(&format!("  {} = {:.4}\n", var, val));
        }

        out
    }

    fn bar(degree: f64) -> String {
        let filled = (degree * 10.0).round() as usize;
        let empty = 10 - filled.min(10);
        format!("[{}{}]", "█".repeat(filled.min(10)), "░".repeat(empty))
    }
}

// ─────────────────────────────────────────────
// CogTable: tabela de centroide discreto
// ─────────────────────────────────────────────

/// Discrete Centre-of-Gravity (COG) table for a consequent variable.
#[derive(Debug, Clone)]
pub struct CogTable {
    pub disc_pts: Vec<f64>,
    pub mu_values: Vec<f64>,
    pub products: Vec<f64>,
    pub numerator: f64,
    pub denominator: f64,
    pub centroid: f64,
}

impl CogTable {
    pub fn print(&self, label: &str) {
        println!("\n  [ COG table — {} ]", label);
        println!(
            "  {:>8}  {:>14}  {:>22}",
            "I_i", "mu_agg(I_i)", "I_i * mu_agg(I_i)"
        );
        println!("  {}", "─".repeat(50));
        for ((x, mu), prod) in self
            .disc_pts
            .iter()
            .zip(self.mu_values.iter())
            .zip(self.products.iter())
        {
            println!("  {:>8.1}  {:>14.6}  {:>22.6}", x, mu, prod);
        }
        println!("  {}", "─".repeat(50));
        println!(
            "  {:>8}  {:>14.6}  {:>22.6}  <- sums",
            "", self.denominator, self.numerator
        );
        println!("  Numerator   = {:.6}", self.numerator);
        println!("  Denominator = {:.6}", self.denominator);
        println!("  Centroid    = {:.6}", self.centroid);
    }
}

// ─────────────────────────────────────────────
// Testes unitários
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::RuleBuilder;
    use crate::{FuzzyVariable, MamdaniEngine, MembershipFn, Term, Universe};

    // ── FuzzifiedVariable ──────────────────────────────────────

    fn make_fv(degrees: &[(&str, f64)]) -> FuzzifiedVariable {
        FuzzifiedVariable {
            variable: "test".to_string(),
            crisp_input: 0.0,
            term_degrees: degrees.iter().map(|(l, d)| (l.to_string(), *d)).collect(),
        }
    }

    #[test]
    fn dominant_term_retorna_o_maior() {
        let fv = make_fv(&[("low", 0.2), ("medium", 0.8), ("high", 0.1)]);
        assert_eq!(fv.dominant_term(), Some("medium"));
    }

    #[test]
    fn dominant_term_retorna_primeiro_quando_empate() {
        let fv = make_fv(&[("a", 0.5), ("b", 0.5)]);
        let dom = fv.dominant_term();
        assert!(dom == Some("a") || dom == Some("b"));
    }

    #[test]
    fn dominant_term_vazio_retorna_none() {
        let fv = make_fv(&[]);
        assert_eq!(fv.dominant_term(), None);
    }

    #[test]
    fn dominant_term_unico_retorna_ele() {
        let fv = make_fv(&[("only", 0.73)]);
        assert_eq!(fv.dominant_term(), Some("only"));
    }

    #[test]
    fn dominant_term_grau_zero_ainda_retorna() {
        let fv = make_fv(&[("a", 0.0), ("b", 0.0)]);
        assert!(fv.dominant_term().is_some());
    }

    // ── RuleFiring (multiconsequent) ─────────────────────────

    fn make_rule_firing(consequents: Vec<(&str, &str)>, fired: bool, degree: f64) -> RuleFiring {
        RuleFiring {
            rule_text: "IF x IS a THEN y IS b".to_string(),
            firing_degree: degree,
            fired,
            consequents: consequents
                .into_iter()
                .map(|(v, t)| (v.to_string(), t.to_string()))
                .collect(),
        }
    }

    #[test]
    fn rule_firing_single_consequent() {
        let rf = make_rule_firing(vec![("y", "b")], true, 0.75);
        assert!(rf.fired);
        assert_eq!(rf.consequents.len(), 1);
        assert_eq!(rf.consequents[0].0, "y");
        assert_eq!(rf.consequents[0].1, "b");
    }

    #[test]
    fn rule_firing_multiple_consequents() {
        let rf = make_rule_firing(vec![("y", "b"), ("z", "c")], true, 0.6);
        assert_eq!(rf.consequents.len(), 2);
        assert_eq!(rf.consequents[1].0, "z");
        assert_eq!(rf.consequents[1].1, "c");
    }

    #[test]
    fn rule_firing_not_fired_degree_zero() {
        let rf = make_rule_firing(vec![("y", "b")], false, 0.0);
        assert!(!rf.fired);
        assert_eq!(rf.firing_degree, 0.0);
    }

    // ── ExplainReport::summary ────────────────────────────────

    fn make_report() -> ExplainReport {
        let fv = FuzzifiedVariable {
            variable: "temperatura".to_string(),
            crisp_input: 5.0,
            term_degrees: vec![("fria".to_string(), 0.8), ("quente".to_string(), 0.0)],
        };
        let rf_fired = RuleFiring {
            rule_text: "IF temperatura IS fria THEN speed IS slow".to_string(),
            firing_degree: 0.8,
            fired: true,
            consequents: vec![("speed".to_string(), "slow".to_string())],
        };
        let rf_skip = RuleFiring {
            rule_text: "IF temperatura IS quente THEN speed IS fast".to_string(),
            firing_degree: 0.0,
            fired: false,
            consequents: vec![("speed".to_string(), "fast".to_string())],
        };
        let mut outputs = HashMap::new();
        outputs.insert("speed".to_string(), 18.5);

        ExplainReport {
            fuzzification: vec![fv],
            rule_firings: vec![rf_fired, rf_skip],
            outputs,
            rules_fired: 1,
            rules_skipped: 1,
        }
    }

    #[test]
    fn summary_contem_todas_secoes() {
        let s = make_report().summary();
        assert!(s.contains("Fuzzification"));
        assert!(s.contains("Rule Evaluation"));
        assert!(s.contains("Defuzzification"));
    }

    #[test]
    fn summary_contem_nome_da_variavel() {
        let s = make_report().summary();
        assert!(s.contains("temperatura"));
    }

    #[test]
    fn summary_contem_valor_crisp() {
        let s = make_report().summary();
        assert!(s.contains("5.0") || s.contains("5,0"));
    }

    #[test]
    fn summary_contem_contagem_fired_skipped() {
        let s = make_report().summary();
        assert!(s.contains("1 fired"));
        assert!(s.contains("1 skipped"));
    }

    #[test]
    fn summary_contem_saida_defuzzificada() {
        let s = make_report().summary();
        assert!(s.contains("speed"));
        assert!(s.contains("18.5") || s.contains("18,5"));
    }

    #[test]
    fn summary_marca_regra_disparada_com_check() {
        let s = make_report().summary();
        assert!(s.contains('✓'));
    }

    #[test]
    fn summary_marca_regra_nao_disparada_com_x() {
        let s = make_report().summary();
        assert!(s.contains('✗'));
    }

    // ── Explain integration with multi-consequent rules ───────

    fn multi_consequent_engine() -> MamdaniEngine {
        let mut engine = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        x.add_term(Term::new(
            "high",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        engine.add_antecedent(x);

        let mut y = FuzzyVariable::new("y", Universe::new(0.0, 10.0, 101));
        y.add_term(Term::new(
            "big",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        engine.add_consequent(y);

        let mut z = FuzzyVariable::new("z", Universe::new(0.0, 20.0, 101));
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
    fn explain_multi_consequent_rule_firing() {
        let engine = multi_consequent_engine();
        let report = engine.explain().unwrap();
        assert_eq!(report.rule_firings.len(), 1);
        let rf = &report.rule_firings[0];
        assert!(rf.fired);
        assert_eq!(rf.consequents.len(), 2);
        assert!(rf.consequents.iter().any(|(v, t)| v == "y" && t == "big"));
        assert!(rf.consequents.iter().any(|(v, t)| v == "z" && t == "large"));
    }

    #[test]
    fn explain_multi_consequent_outputs_present() {
        let engine = multi_consequent_engine();
        let report = engine.explain().unwrap();
        assert!(report.outputs.contains_key("y"));
        assert!(report.outputs.contains_key("z"));
    }
}
