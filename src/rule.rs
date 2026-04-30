//! # rule.rs
//!
//! Regras de inferencia Fuzzy do tipo SE–ENTAO.
//! Equivalente a ctrl.Rule do scikit-fuzzy.
//!
//! Cada Rule conecta um ou mais antecedentes (entradas) a um consequente
//! (saida) usando AND (min) ou OR (max).
//!
//! Pipeline de uso:
//!   1. Crie Rules com antecedentes e consequente
//!   2. O MamdaniEngine chamara firing_strength() na etapa de inferencia
//!   3. O grau retornado e usado para recortar (clip) a MF do consequente

use crate::variable::FuzzyVariable;
use std::collections::HashMap;
use std::fmt;

// ─────────────────────────────────────────────────────────────────
// Connector
// ─────────────────────────────────────────────────────────────────

/// Conector logico entre os antecedentes de uma regra.
///
/// - `And` → grau de disparo = MIN dos graus de pertinencia (t-norma)
/// - `Or`  → grau de disparo = MAX dos graus de pertinencia (s-norma)
///
/// Equivalente ao operador implicito `&` e `|` nas Rules do scikit-fuzzy:
///   ctrl.Rule(temperatura['alta'] & umidade['baixa'], velocidade['media'])
#[derive(Debug, Clone, PartialEq)]
pub enum Connector {
    /// Conjuncao fuzzy: min(μ₁, μ₂, ..., μₙ)
    And,
    /// Disjuncao fuzzy: max(μ₁, μ₂, ..., μₙ)
    Or,
}

// ─────────────────────────────────────────────────────────────────
// Rule
// ─────────────────────────────────────────────────────────────────

/// Regra de inferencia Fuzzy do tipo SE–ENTAO (Mamdani).
///
/// Uma Rule conecta N antecedentes a um unico consequente:
///   SE (var1 eh term1) AND/OR (var2 eh term2) ... ENTAO (saida eh term_saida)
///
/// O grau de disparo (firing strength) determina o corte na MF do consequente
/// durante a etapa de implicacao.
///
/// # Exemplo
/// ```
/// use fuzzy_mamdani::rule::{Rule, Connector};
///
/// let regra = Rule::new(
///     vec![
///         ("temperatura".to_string(), "quente".to_string()),
///         ("umidade".to_string(),     "alta".to_string()),
///     ],
///     Connector::And,
///     "velocidade_ventilador".to_string(),
///     "rapida".to_string(),
/// );
/// assert_eq!(regra.consequent_var(), "velocidade_ventilador");
/// assert_eq!(regra.consequent_term(), "rapida");
/// assert_eq!(regra.antecedent_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct Rule {
    /// Lista de (nome_da_variavel, label_do_termo) que formam os antecedentes.
    antecedents: Vec<(String, String)>,
    /// Conector logico entre os antecedentes (And ou Or).
    connector: Connector,
    /// Nome da variavel consequente (saida).
    consequent_var: String,
    /// Label do termo consequente.
    consequent_term: String,
}

impl Rule {
    /// Cria uma nova regra.
    ///
    /// # Argumentos
    /// * `antecedents` — lista de (nome_variavel, label_termo)
    /// * `connector`   — And ou Or
    /// * `consequent_var`  — nome da variavel de saida
    /// * `consequent_term` — label do termo de saida
    ///
    /// Panics se a lista de antecedentes estiver vazia.
    pub fn new(
        antecedents: Vec<(String, String)>,
        connector: Connector,
        consequent_var: String,
        consequent_term: String,
    ) -> Self {
        assert!(
            !antecedents.is_empty(),
            "Rule: antecedent list cannot be empty"
        );
        Self {
            antecedents,
            connector,
            consequent_var,
            consequent_term,
        }
    }

    // ── Getters ─────────────────────────────────────────────────

    /// Retorna referencia a lista de antecedentes (var_name, term_label).
    pub fn antecedents(&self) -> &[(String, String)] {
        &self.antecedents
    }

    /// Retorna o conector desta regra.
    pub fn connector(&self) -> &Connector {
        &self.connector
    }

    /// Returns the name of the consequent variable.
    pub fn consequent_var(&self) -> &str {
        &self.consequent_var
    }

    /// Retorna o label do termo consequente.
    pub fn consequent_term(&self) -> &str {
        &self.consequent_term
    }

    /// Retorna o numero de antecedentes desta regra.
    pub fn antecedent_count(&self) -> usize {
        self.antecedents.len()
    }

    // ── Avaliacao ───────────────────────────────────────────────

    /// Calcula o grau de disparo (firing strength) da regra dados os
    /// valores crisp de entrada e as variaveis antecedentes.
    ///
    /// Etapas:
    ///   1. Fuzzificacao: para cada antecedente, calcula μ(input_value)
    ///   2. Agregacao dos antecedentes via conector:
    ///      - And → min(μ₁, μ₂, ..., μₙ)  [t-norma de Mamdani]
    ///      - Or  → max(μ₁, μ₂, ..., μₙ)  [s-norma]
    ///
    /// Retorna 0.0 se alguma variavel ou termo nao for encontrado (gracioso).
    ///
    /// # Argumentos
    /// * `inputs`       — mapa { nome_variavel → valor_crisp }
    /// * `antecedents`  — mapa { nome_variavel → FuzzyVariable }
    ///
    /// # Exemplo
    /// ```
    /// use std::collections::HashMap;
    /// use fuzzy_mamdani::{FuzzyVariable, Universe, Term, MembershipFn};
    /// use fuzzy_mamdani::rule::{Rule, Connector};
    ///
    /// let mut temp = FuzzyVariable::new("temperatura", Universe::new(0.0, 50.0, 501));
    /// temp.add_term(Term::new("quente", MembershipFn::Trimf([25.0, 50.0, 50.0])));
    ///
    /// let mut antecedents = HashMap::new();
    /// antecedents.insert("temperatura".to_string(), temp);
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("temperatura".to_string(), 50.0_f64);
    ///
    /// let regra = Rule::new(
    ///     vec![("temperatura".to_string(), "quente".to_string())],
    ///     Connector::And,
    ///     "velocidade".to_string(),
    ///     "rapida".to_string(),
    /// );
    ///
    /// let grau = regra.firing_strength(&inputs, &antecedents);
    /// assert!((grau - 1.0).abs() < 1e-10);
    /// ```
    pub fn firing_strength(
        &self,
        inputs: &HashMap<String, f64>,
        antecedents: &HashMap<String, FuzzyVariable>,
    ) -> f64 {
        // Coleta os graus de pertinencia de cada antecedente
        let graus: Vec<f64> = self
            .antecedents
            .iter()
            .filter_map(|(var_name, term_label)| {
                let valor = inputs.get(var_name.as_str())?;
                let variavel = antecedents.get(var_name.as_str())?;
                Some(variavel.membership_at(term_label, *valor))
            })
            .collect();

        // Regra sem nenhum grau calculado → disparo zero
        if graus.is_empty() {
            return 0.0;
        }

        // Aplica o conector entre os antecedentes
        match self.connector {
            // AND (conjuncao): t-norma de Mamdani — minimo dos graus
            Connector::And => graus.iter().cloned().fold(f64::INFINITY, f64::min),
            // OR (disjuncao): s-norma — maximo dos graus
            Connector::Or => graus.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

// Implementa Display para Rule, satisfazendo o lint do clippy (inherent_to_string).
// Permite usar format!("{}", regra) e println!("{}", regra) normalmente.
impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let conector_str = match self.connector {
            Connector::And => "AND",
            Connector::Or => "OR",
        };

        let antecedentes_str: Vec<String> = self
            .antecedents
            .iter()
            .map(|(var, term)| format!("({} is {})", var, term))
            .collect();

        write!(
            f,
            "IF {} THEN {} is {}",
            antecedentes_str.join(&format!(" {} ", conector_str)),
            self.consequent_var,
            self.consequent_term,
        )
    }
}

// ─────────────────────────────────────────────────────────────────
// Builder (padrao fluente para facilitar a criacao de regras)
// ─────────────────────────────────────────────────────────────────

/// Builder fluente para construcao de Rules de forma legivel.
///
/// # Exemplo
/// ```
/// use fuzzy_mamdani::rule::{RuleBuilder, Connector};
///
/// let regra = RuleBuilder::new()
///     .when("temperatura", "quente")
///     .and("umidade", "alta")
///     .then("velocidade_ventilador", "rapida");
///
/// assert_eq!(regra.antecedent_count(), 2);
/// assert_eq!(regra.connector(), &Connector::And);
/// assert_eq!(regra.consequent_term(), "rapida");
/// ```
pub struct RuleBuilder {
    antecedents: Vec<(String, String)>,
    connector: Connector,
}

impl RuleBuilder {
    /// Inicia um novo builder.
    pub fn new() -> Self {
        Self {
            antecedents: Vec::new(),
            connector: Connector::And,
        }
    }

    /// Adiciona o primeiro antecedente (o SE).
    pub fn when(mut self, var: &str, term: &str) -> Self {
        self.antecedents.push((var.to_string(), term.to_string()));
        self
    }

    /// Adiciona antecedente com conector AND.
    pub fn and(mut self, var: &str, term: &str) -> Self {
        self.connector = Connector::And;
        self.antecedents.push((var.to_string(), term.to_string()));
        self
    }

    /// Adiciona antecedente com conector OR.
    pub fn or(mut self, var: &str, term: &str) -> Self {
        self.connector = Connector::Or;
        self.antecedents.push((var.to_string(), term.to_string()));
        self
    }

    /// Finaliza a regra com o consequente (o ENTAO). Consome o builder.
    pub fn then(self, var: &str, term: &str) -> Rule {
        Rule::new(
            self.antecedents,
            self.connector,
            var.to_string(),
            term.to_string(),
        )
    }
}

impl Default for RuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// Testes unitarios
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FuzzyVariable, MembershipFn, Term, Universe};

    // ── Helpers ────────────────────────────────────────────────

    fn make_antecedents() -> HashMap<String, FuzzyVariable> {
        let mut temperatura = FuzzyVariable::new("temperatura", Universe::new(0.0, 50.0, 501));
        temperatura.add_term(Term::new("fria", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        temperatura.add_term(Term::new("morna", MembershipFn::Trimf([0.0, 25.0, 50.0])));
        temperatura.add_term(Term::new("quente", MembershipFn::Trimf([25.0, 50.0, 50.0])));

        let mut umidade = FuzzyVariable::new("umidade", Universe::new(0.0, 100.0, 1001));
        umidade.add_term(Term::new("baixa", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        umidade.add_term(Term::new("media", MembershipFn::Trimf([0.0, 50.0, 100.0])));
        umidade.add_term(Term::new("alta", MembershipFn::Trimf([50.0, 100.0, 100.0])));

        let mut map = HashMap::new();
        map.insert("temperatura".to_string(), temperatura);
        map.insert("umidade".to_string(), umidade);
        map
    }

    fn make_inputs(temp: f64, umid: f64) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("temperatura".to_string(), temp);
        m.insert("umidade".to_string(), umid);
        m
    }

    fn regra_and() -> Rule {
        RuleBuilder::new()
            .when("temperatura", "quente")
            .and("umidade", "alta")
            .then("velocidade_ventilador", "rapida")
    }

    fn regra_or() -> Rule {
        RuleBuilder::new()
            .when("temperatura", "quente")
            .or("umidade", "alta")
            .then("velocidade_ventilador", "rapida")
    }

    // ── Connector ──────────────────────────────────────────────

    #[test]
    fn connector_and_eq() {
        assert_eq!(Connector::And, Connector::And);
        assert_ne!(Connector::And, Connector::Or);
    }

    // ── Rule: construcao ───────────────────────────────────────

    #[test]
    fn rule_antecedent_count() {
        assert_eq!(regra_and().antecedent_count(), 2);
    }

    #[test]
    fn rule_consequent_var() {
        assert_eq!(regra_and().consequent_var(), "velocidade_ventilador");
    }

    #[test]
    fn rule_consequent_term() {
        assert_eq!(regra_and().consequent_term(), "rapida");
    }

    #[test]
    fn rule_connector_and() {
        assert_eq!(regra_and().connector(), &Connector::And);
    }

    #[test]
    fn rule_connector_or() {
        assert_eq!(regra_or().connector(), &Connector::Or);
    }

    #[test]
    #[should_panic(expected = "antecedent list cannot be empty")]
    fn rule_sem_antecedentes_panics() {
        Rule::new(
            vec![],
            Connector::And,
            "saida".to_string(),
            "alto".to_string(),
        );
    }

    // ── firing_strength: AND ───────────────────────────────────

    #[test]
    fn and_ambos_pleno_disparo() {
        // temp=50 (quente=1.0), umid=100 (alta=1.0) → min(1.0,1.0) = 1.0
        let r = regra_and();
        let grau = r.firing_strength(&make_inputs(50.0, 100.0), &make_antecedents());
        assert!((grau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn and_um_zero_resulta_zero() {
        // temp=0 (quente=0.0), umid=100 (alta=1.0) → min(0.0,1.0) = 0.0
        let r = regra_and();
        let grau = r.firing_strength(&make_inputs(0.0, 100.0), &make_antecedents());
        assert_eq!(grau, 0.0);
    }

    #[test]
    fn and_usa_minimo() {
        // temp=50 (quente=1.0), umid=75 (alta=0.5) → min(1.0,0.5) = 0.5
        let r = regra_and();
        let grau = r.firing_strength(&make_inputs(50.0, 75.0), &make_antecedents());
        assert!((grau - 0.5).abs() < 1e-10);
    }

    #[test]
    fn and_grau_intermediario() {
        // temp=37.5 (quente=0.5), umid=75 (alta=0.5) → min(0.5,0.5) = 0.5
        let r = regra_and();
        let grau = r.firing_strength(&make_inputs(37.5, 75.0), &make_antecedents());
        assert!((grau - 0.5).abs() < 1e-10);
    }

    #[test]
    fn and_menor_grau_prevalece() {
        // temp=37.5 (quente=0.5), umid=100 (alta=1.0) → min = 0.5
        let r = regra_and();
        let grau = r.firing_strength(&make_inputs(37.5, 100.0), &make_antecedents());
        assert!((grau - 0.5).abs() < 1e-10);
    }

    // ── firing_strength: OR ────────────────────────────────────

    #[test]
    fn or_usa_maximo() {
        // temp=0 (quente=0.0), umid=100 (alta=1.0) → max(0.0,1.0) = 1.0
        let r = regra_or();
        let grau = r.firing_strength(&make_inputs(0.0, 100.0), &make_antecedents());
        assert!((grau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn or_ambos_zero_resulta_zero() {
        // temp=0 (quente=0.0), umid=0 (alta=0.0) → max(0.0,0.0) = 0.0
        let r = regra_or();
        let grau = r.firing_strength(&make_inputs(0.0, 0.0), &make_antecedents());
        assert_eq!(grau, 0.0);
    }

    #[test]
    fn or_grau_intermediario() {
        // temp=37.5 (quente=0.5), umid=75 (alta=0.5) → max(0.5,0.5) = 0.5
        let r = regra_or();
        let grau = r.firing_strength(&make_inputs(37.5, 75.0), &make_antecedents());
        assert!((grau - 0.5).abs() < 1e-10);
    }

    #[test]
    fn or_maior_grau_prevalece() {
        // temp=50 (quente=1.0), umid=75 (alta=0.5) → max = 1.0
        let r = regra_or();
        let grau = r.firing_strength(&make_inputs(50.0, 75.0), &make_antecedents());
        assert!((grau - 1.0).abs() < 1e-10);
    }

    // ── firing_strength: gracioso com inputs faltando ──────────

    #[test]
    fn variavel_ausente_retorna_zero() {
        // inputs nao tem "umidade" → filter_map descarta → graus=[quente_grau] so
        // Comportamento gracioso: regra dispara apenas com os antecedentes disponiveis
        let r = regra_and();
        let mut inputs = HashMap::new();
        inputs.insert("temperatura".to_string(), 50.0_f64);
        // umidade ausente → filter_map descarta → graus tem so 1 elemento
        // AND de um so grau = esse grau
        let grau = r.firing_strength(&inputs, &make_antecedents());
        assert!((grau - 1.0).abs() < 1e-10); // quente em 50 = 1.0
    }

    // ── Rule com 1 antecedente ─────────────────────────────────

    #[test]
    fn rule_um_antecedente_and() {
        let r = RuleBuilder::new()
            .when("temperatura", "fria")
            .then("velocidade_ventilador", "lenta");
        // temp=0 → fria=1.0
        let mut inputs = HashMap::new();
        inputs.insert("temperatura".to_string(), 0.0_f64);
        let grau = r.firing_strength(&inputs, &make_antecedents());
        assert!((grau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rule_um_antecedente_sem_disparo() {
        let r = RuleBuilder::new()
            .when("temperatura", "fria")
            .then("velocidade_ventilador", "lenta");
        // temp=50 → fria=0.0
        let mut inputs = HashMap::new();
        inputs.insert("temperatura".to_string(), 50.0_f64);
        let grau = r.firing_strength(&inputs, &make_antecedents());
        assert_eq!(grau, 0.0);
    }

    // ── to_string ──────────────────────────────────────────────

    #[test]
    fn to_string_contem_se_entao() {
        let s = regra_and().to_string();
        assert!(s.contains("IF"), "missing IF: {}", s);
        assert!(s.contains("THEN"), "missing THEN: {}", s);
    }

    #[test]
    fn to_string_contem_variaveis() {
        let s = regra_and().to_string();
        assert!(s.contains("temperatura"), "faltou temperatura: {}", s);
        assert!(s.contains("quente"), "faltou quente: {}", s);
        assert!(s.contains("velocidade"), "faltou velocidade: {}", s);
        assert!(s.contains("rapida"), "faltou rapida: {}", s);
    }

    #[test]
    fn to_string_and_contem_and() {
        assert!(regra_and().to_string().contains("AND"));
    }

    #[test]
    fn to_string_or_contem_or() {
        assert!(regra_or().to_string().contains("OR"));
    }

    // ── RuleBuilder ────────────────────────────────────────────

    #[test]
    fn builder_when_then() {
        let r = RuleBuilder::new()
            .when("temperatura", "quente")
            .then("saida", "alta");
        assert_eq!(r.antecedent_count(), 1);
        assert_eq!(r.consequent_var(), "saida");
        assert_eq!(r.consequent_term(), "alta");
    }

    #[test]
    fn builder_and_chain() {
        let r = RuleBuilder::new()
            .when("temperatura", "quente")
            .and("umidade", "alta")
            .then("saida", "alta");
        assert_eq!(r.antecedent_count(), 2);
        assert_eq!(r.connector(), &Connector::And);
    }

    #[test]
    fn builder_or_chain() {
        let r = RuleBuilder::new()
            .when("temperatura", "quente")
            .or("umidade", "alta")
            .then("saida", "alta");
        assert_eq!(r.connector(), &Connector::Or);
    }

    #[test]
    fn builder_tres_antecedentes() {
        let r = RuleBuilder::new()
            .when("a", "x")
            .and("b", "y")
            .and("c", "z")
            .then("saida", "alta");
        assert_eq!(r.antecedent_count(), 3);
    }

    // ── Cenario realista: ventilador ───────────────────────────

    #[test]
    fn cenario_ventilador_regras_basicas() {
        let ants = make_antecedents();

        // Regra 1: SE temp fria AND umid baixa ENTAO lenta
        let r1 = RuleBuilder::new()
            .when("temperatura", "fria")
            .and("umidade", "baixa")
            .then("velocidade_ventilador", "lenta");

        // Regra 2: SE temp quente OR umid alta ENTAO rapida
        let r2 = RuleBuilder::new()
            .when("temperatura", "quente")
            .or("umidade", "alta")
            .then("velocidade_ventilador", "rapida");

        // Cenario: temp=0 (fria=1.0), umid=0 (baixa=1.0)
        let inputs_frio = make_inputs(0.0, 0.0);
        assert!((r1.firing_strength(&inputs_frio, &ants) - 1.0).abs() < 1e-10);
        assert_eq!(r2.firing_strength(&inputs_frio, &ants), 0.0);

        // Cenario: temp=50 (quente=1.0), umid=100 (alta=1.0)
        let inputs_quente = make_inputs(50.0, 100.0);
        assert_eq!(r1.firing_strength(&inputs_quente, &ants), 0.0);
        assert!((r2.firing_strength(&inputs_quente, &ants) - 1.0).abs() < 1e-10);
    }
}
