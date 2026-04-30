//! # explain.rs
//!
//! Estruturas de dados para o relatorio de explicabilidade do sistema Fuzzy.
//! Permite inspecionar cada etapa do pipeline Mamdani apos um compute().

use std::collections::HashMap;

// ─────────────────────────────────────────────
// RuleFiring: resultado de uma regra individual
// ─────────────────────────────────────────────

/// Result of evaluating a single fuzzy inference rule.
///
/// Captures the human-readable rule description, its firing degree,
/// and whether it actually contributed to the output (degree > 0).
#[derive(Debug, Clone)]
pub struct RuleFiring {
    /// Human-readable representation of the rule (e.g. "IF temp IS hot THEN speed IS fast").
    pub rule_text: String,
    /// Firing degree of this rule in [0.0, 1.0].
    /// AND rules use min; OR rules use max across antecedents.
    pub firing_degree: f64,
    /// Whether this rule contributed to the aggregation (firing_degree > 0).
    pub fired: bool,
    /// Name of the consequent variable this rule affects.
    pub consequent_var: String,
    /// Name of the consequent term this rule activates.
    pub consequent_term: String,
}

// ─────────────────────────────────────────────
// FuzzifiedVariable: resultado da fuzzificacao
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
    /// Ordered as terms were added to the variable.
    pub term_degrees: Vec<(String, f64)>,
}

impl FuzzifiedVariable {
    /// Returns the term with the highest membership degree for this input.
    /// Useful for a quick "best match" summary.
    pub fn dominant_term(&self) -> Option<&str> {
        self.term_degrees
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(label, _)| label.as_str())
    }
}

// ─────────────────────────────────────────────
// ExplainReport: relatorio completo
// ─────────────────────────────────────────────

/// Full explanation report of one Mamdani inference cycle.
///
/// Returned by [`MamdaniEngine::explain`](crate::engine::MamdaniEngine::explain).
/// Contains the intermediate state of every pipeline stage:
/// fuzzification → rule evaluation → defuzzification.
///
/// # Example
/// ```
/// use logicfuzzy_academic::{FuzzyVariable, Universe, Term, MembershipFn};
/// use logicfuzzy_academic::rule::RuleBuilder;
/// use logicfuzzy_academic::engine::MamdaniEngine;
///
/// let mut engine = MamdaniEngine::new();
///
/// let mut temp = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
/// temp.add_term(Term::new("cold", MembershipFn::Trimf([0.0, 0.0, 25.0])));
/// temp.add_term(Term::new("hot",  MembershipFn::Trimf([25.0, 50.0, 50.0])));
/// engine.add_antecedent(temp);
///
/// let mut speed = FuzzyVariable::new("speed", Universe::new(0.0, 100.0, 1001));
/// speed.add_term(Term::new("slow", MembershipFn::Trimf([0.0, 0.0, 50.0])));
/// speed.add_term(Term::new("fast", MembershipFn::Trimf([50.0, 100.0, 100.0])));
/// engine.add_consequent(speed);
///
/// engine.add_rule(RuleBuilder::new().when("temperature","cold").then("speed","slow").build());
/// engine.add_rule(RuleBuilder::new().when("temperature","hot").then("speed","fast").build());
///
/// engine.set_input("temperature", 5.0).unwrap();
/// let report = engine.explain().unwrap();
/// assert!(report.outputs["speed"] < 50.0);
/// println!("{}", report.summary());
/// ```
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
    ///
    /// Useful for printing to stdout or logging during development.
    pub fn summary(&self) -> String {
        let mut out = String::new();

        // cabecalho
        out.push_str("=== Fuzzy Mamdani — Explain Report ===\n\n");

        // fuzzificacao
        out.push_str("[ Fuzzification ]\n");
        for fv in &self.fuzzification {
            out.push_str(&format!("  {} = {:.4} (crisp)\n", fv.variable, fv.crisp_input));
            for (term, degree) in &fv.term_degrees {
                let bar = Self::bar(*degree);
                out.push_str(&format!("    {:>12}  {:.4}  {}\n", term, degree, bar));
            }
            if let Some(dom) = fv.dominant_term() {
                out.push_str(&format!("    → dominant term: {}\n", dom));
            }
            out.push('\n');
        }

        // regras
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

        // saidas
        out.push_str("[ Defuzzification Output ]\n");
        for (var, val) in &self.outputs {
            out.push_str(&format!("  {} = {:.4}\n", var, val));
        }

        out
    }

    // barra visual de grau de pertinencia (0..10 chars)
    fn bar(degree: f64) -> String {
        let filled = (degree * 10.0).round() as usize;
        let empty  = 10 - filled.min(10);
        format!("[{}{}]", "█".repeat(filled.min(10)), "░".repeat(empty))
    }
}


// ─────────────────────────────────────────────
// CogTable: tabela de centroide discreto
// ─────────────────────────────────────────────

/// Discrete Centre-of-Gravity (COG) table for a consequent variable.
///
/// Returned by [`MamdaniEngine::discrete_cog`](crate::engine::MamdaniEngine::discrete_cog).
/// Replicates the step-by-step centroid calculation shown in fuzzy control textbooks
/// (equivalent to notebook cell 13).
///
/// # Fields
/// - `disc_pts`: the discrete sample points (e.g. {0, 10, 20, …, 100})
/// - `mu_values`: aggregated membership degree at each sample point
/// - `products`: `I_i × μ(I_i)` for each point
/// - `numerator`: `Σ(I_i × μ(I_i))`
/// - `denominator`: `Σ(μ(I_i))`
/// - `centroid`: `numerator / denominator`
#[derive(Debug, Clone)]
pub struct CogTable {
    /// Discrete sample points.
    pub disc_pts:    Vec<f64>,
    /// Aggregated μ at each sample point.
    pub mu_values:   Vec<f64>,
    /// `I_i × μ(I_i)` for each point.
    pub products:    Vec<f64>,
    /// `Σ(I_i × μ(I_i))`.
    pub numerator:   f64,
    /// `Σ(μ(I_i))`.
    pub denominator: f64,
    /// Defuzzified crisp value = `numerator / denominator`.
    pub centroid:    f64,
}

impl CogTable {
    /// Prints the table to stdout in a readable ASCII format.
    pub fn print(&self, label: &str) {
        println!("\n  [ COG table — {} ]", label);
        println!("  {:>8}  {:>14}  {:>22}", "I_i", "mu_agg(I_i)", "I_i * mu_agg(I_i)");
        println!("  {}", "─".repeat(50));
        for ((x, mu), prod) in self.disc_pts.iter()
            .zip(self.mu_values.iter())
            .zip(self.products.iter())
        {
            println!("  {:>8.1}  {:>14.6}  {:>22.6}", x, mu, prod);
        }
        println!("  {}", "─".repeat(50));
        println!("  {:>8}  {:>14.6}  {:>22.6}  <- sums", "", self.denominator, self.numerator);
        println!("  Numerator   = {:.6}", self.numerator);
        println!("  Denominator = {:.6}", self.denominator);
        println!("  Centroid    = {:.6}", self.centroid);
    }
}

// ─────────────────────────────────────────────
// Testes unitarios dos tipos de explain
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        // max_by retorna o ULTIMO elemento em caso de empate de f64 igual
        // este teste documenta o comportamento, nao exige ordem especifica
        let fv = make_fv(&[("a", 0.5), ("b", 0.5)]);
        let dom = fv.dominant_term();
        assert!(dom == Some("a") || dom == Some("b"),
            "dominant_term deveria ser 'a' ou 'b', obteve {:?}", dom);
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
        // Mesmo com todos os graus em 0.0, deve retornar algum (o primeiro/ultimo)
        let fv = make_fv(&[("a", 0.0), ("b", 0.0)]);
        assert!(fv.dominant_term().is_some());
    }

    // ── RuleFiring ────────────────────────────────────────────

    #[test]
    fn rule_firing_fired_true_quando_degree_positivo() {
        let rf = RuleFiring {
            rule_text:       "IF x IS a THEN y IS b".to_string(),
            firing_degree:   0.75,
            fired:           true,
            consequent_var:  "y".to_string(),
            consequent_term: "b".to_string(),
        };
        assert!(rf.fired);
        assert!(rf.firing_degree > 0.0);
    }

    #[test]
    fn rule_firing_fired_false_quando_degree_zero() {
        let rf = RuleFiring {
            rule_text:       "IF x IS a THEN y IS b".to_string(),
            firing_degree:   0.0,
            fired:           false,
            consequent_var:  "y".to_string(),
            consequent_term: "b".to_string(),
        };
        assert!(!rf.fired);
        assert_eq!(rf.firing_degree, 0.0);
    }

    // ── ExplainReport::bar ────────────────────────────────────

    #[test]
    fn bar_grau_zero_retorna_vazio() {
        let b = ExplainReport::bar(0.0);
        assert!(b.contains("░"), "grau 0.0 deve ser barra vazia: {}", b);
        assert!(!b.contains("█"), "grau 0.0 nao deve ter preenchimento: {}", b);
    }

    #[test]
    fn bar_grau_um_retorna_cheio() {
        let b = ExplainReport::bar(1.0);
        assert!(b.contains("█"), "grau 1.0 deve ter preenchimento: {}", b);
        assert!(!b.contains("░"), "grau 1.0 nao deve ter vazio: {}", b);
    }

    #[test]
    fn bar_grau_meio_tem_ambos() {
        let b = ExplainReport::bar(0.5);
        assert!(b.contains("█"), "grau 0.5 deve ter algum preenchimento: {}", b);
        assert!(b.contains("░"), "grau 0.5 deve ter algum vazio: {}", b);
    }

    // ── ExplainReport::summary ────────────────────────────────

    fn make_report() -> ExplainReport {
        let fv = FuzzifiedVariable {
            variable:     "temperatura".to_string(),
            crisp_input:  5.0,
            term_degrees: vec![
                ("fria".to_string(),   0.8),
                ("quente".to_string(), 0.0),
            ],
        };
        let rf_fired = RuleFiring {
            rule_text:       "IF temperatura IS fria THEN speed IS slow".to_string(),
            firing_degree:   0.8,
            fired:           true,
            consequent_var:  "speed".to_string(),
            consequent_term: "slow".to_string(),
        };
        let rf_skip = RuleFiring {
            rule_text:       "IF temperatura IS quente THEN speed IS fast".to_string(),
            firing_degree:   0.0,
            fired:           false,
            consequent_var:  "speed".to_string(),
            consequent_term: "fast".to_string(),
        };
        let mut outputs = HashMap::new();
        outputs.insert("speed".to_string(), 18.5);

        ExplainReport {
            fuzzification: vec![fv],
            rule_firings:  vec![rf_fired, rf_skip],
            outputs,
            rules_fired:   1,
            rules_skipped: 1,
        }
    }

    #[test]
    fn summary_contem_todas_secoes() {
        let s = make_report().summary();
        assert!(s.contains("Fuzzification"),  "faltou secao Fuzzification");
        assert!(s.contains("Rule Evaluation"),"faltou secao Rule Evaluation");
        assert!(s.contains("Defuzzification"),"faltou secao Defuzzification");
    }

    #[test]
    fn summary_contem_nome_da_variavel() {
        let s = make_report().summary();
        assert!(s.contains("temperatura"), "faltou nome da variavel: {}", s);
    }

    #[test]
    fn summary_contem_valor_crisp() {
        let s = make_report().summary();
        assert!(s.contains("5.0") || s.contains("5,0"),
            "faltou valor crisp 5.0 no summary: {}", s);
    }

    #[test]
    fn summary_contem_contagem_fired_skipped() {
        let s = make_report().summary();
        // "1 fired, 1 skipped" ou similar
        assert!(s.contains("1 fired"), "faltou contagem fired: {}", s);
        assert!(s.contains("1 skipped"), "faltou contagem skipped: {}", s);
    }

    #[test]
    fn summary_contem_saida_defuzzificada() {
        let s = make_report().summary();
        assert!(s.contains("speed"), "faltou nome da saida: {}", s);
        assert!(s.contains("18.5") || s.contains("18,5"),
            "faltou valor de saida 18.5: {}", s);
    }

    #[test]
    fn summary_marca_regra_disparada_com_check() {
        let s = make_report().summary();
        assert!(s.contains('✓'), "faltou marcador ✓ de regra disparada");
    }

    #[test]
    fn summary_marca_regra_nao_disparada_com_x() {
        let s = make_report().summary();
        assert!(s.contains('✗'), "faltou marcador ✗ de regra nao disparada");
    }
}
