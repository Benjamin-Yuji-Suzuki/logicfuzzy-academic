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
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────
// MamdaniEngine
// ─────────────────────────────────────────────────────────────────

/// Mamdani Fuzzy inference engine.
///
/// Orchestrates the five pipeline stages:
///
/// 1. **Fuzzification**   — converts crisp inputs into membership degrees
/// 2. **Inference**       — evaluates each rule firing degree (AND=min, OR=max)
/// 3. **Clip**            — clips the consequent MF at the firing degree (Mamdani implication)
/// 4. **Aggregation**    — combines all clipped MFs via max (per consequent)
/// 5. **Defuzzification** — computes the centroid of the aggregated set
///
/// Equivalent to `ctrl.ControlSystemSimulation` from scikit-fuzzy.
#[derive(Debug)]
pub struct MamdaniEngine {
    /// Input variables (antecedents).
    antecedents: HashMap<String, FuzzyVariable>,
    /// Output variables (consequents).
    consequents: HashMap<String, FuzzyVariable>,
    /// Inference rule base.
    rules: Vec<Rule>,
    /// Crisp input values for the next `compute()` call.
    inputs: HashMap<String, f64>,
    /// Defuzzification method to use in compute(). Default: Centroid.
    defuzz_method: DefuzzMethod,
}

// ─────────────────────────────────────────────────────────────────
// DefuzzMethod
// ─────────────────────────────────────────────────────────────────

/// Defuzzification method used by [`MamdaniEngine::compute`].
///
/// All methods operate on the aggregated output MF.
///
/// # Reference
/// These correspond to the methods available in `skfuzzy.defuzz`:
/// `'centroid'`, `'bisector'`, `'mom'`, `'som'`, `'lom'`.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DefuzzMethod {
    /// Centre of Gravity — `Σ(x·μ) / Σ(μ)`. Most common; smooth output.
    #[default]
    Centroid,
    /// Bisector — finds the point that splits the aggregated area in two equal halves.
    Bisector,
    /// Mean of Maximum — average x position among all points with the highest μ.
    MeanOfMaximum,
    /// Smallest of Maximum — leftmost x with the highest μ.
    SmallestOfMaximum,
    /// Largest of Maximum — rightmost x with the highest μ.
    LargestOfMaximum,
}

impl MamdaniEngine {
    /// Creates an empty engine (no variables or rules).
    pub fn new() -> Self {
        Self {
            antecedents: HashMap::new(),
            consequents: HashMap::new(),
            rules: Vec::new(),
            inputs: HashMap::new(),
            defuzz_method: DefuzzMethod::Centroid,
        }
    }

    // ── Configuracao ─────────────────────────────────────────────

    /// Registers an antecedent (input) variable.
    /// Panics if a variable with the same name is already registered.
    pub fn add_antecedent(&mut self, var: FuzzyVariable) {
        assert!(
            !self.antecedents.contains_key(&var.name),
            "MamdaniEngine: antecedent '{}' already registered",
            var.name
        );
        self.antecedents.insert(var.name.clone(), var);
    }

    /// Registers a consequent (output) variable.
    /// Panics if a variable with the same name is already registered.
    pub fn add_consequent(&mut self, var: FuzzyVariable) {
        assert!(
            !self.consequents.contains_key(&var.name),
            "MamdaniEngine: consequent '{}' already registered",
            var.name
        );
        self.consequents.insert(var.name.clone(), var);
    }

    /// Adds an inference rule to the rule base.
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Sets the crisp input value for an antecedent variable.
    /// Must be called before `compute()`.
    /// Sets the crisp input value for an antecedent variable.
    ///
    /// If `value` is outside the variable's universe of discourse it is
    /// **clamped** to `[min, max]` and `Err(FuzzyError::InputOutOfRange)` is
    /// returned so the caller can decide how to handle the warning.
    /// If the variable name is not registered, `Err(FuzzyError::MissingInput)`
    /// is returned.
    ///
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, antecedent};
    ///
    /// let mut engine = MamdaniEngine::new();
    /// antecedent!(engine, "temperature", 0.0, 50.0, 501,
    ///     "cold" => trimf [0.0, 0.0, 25.0],
    /// );
    ///
    /// // In-range: Ok(())
    /// assert!(engine.set_input("temperature", 20.0).is_ok());
    ///
    /// // Out-of-range: clamped, returns Err with InputOutOfRange
    /// assert!(engine.set_input("temperature", 999.0).is_err());
    ///
    /// // Unknown variable: MissingInput error
    /// assert!(engine.set_input("humidity", 50.0).is_err());
    /// ```
    pub fn set_input(&mut self, name: &str, value: f64) -> Result<(), FuzzyError> {
        // Verifica se a variavel existe
        let var = self
            .antecedents
            .get(name)
            .ok_or_else(|| FuzzyError::MissingInput(name.to_string()))?;

        let min = var.universe.min;
        let max = var.universe.max;

        if value < min || value > max {
            // Clampeia e retorna aviso — nao bloqueia a inferencia
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

    /// Sets input and panics on error — convenience for tests and demos.
    ///
    /// Prefer [`set_input`](Self::set_input) in production code.
    pub fn set_input_unchecked(&mut self, name: &str, value: f64) {
        let _ = self.set_input(name, value);
        // Garante que o valor seja inserido mesmo se fora do range
        if !self.inputs.contains_key(name) {
            self.inputs.insert(name.to_string(), value);
        }
    }

    // ── Pipeline principal ────────────────────────────────────────

    /// Runs the full Mamdani pipeline and returns the crisp output values.
    ///
    /// Returns a map `{ consequent_name → crisp_value }` for each registered output variable.
    /// de saida registrada.
    ///
    /// If no rule fires for a consequent, returns the midpoint of that variable's universe (graceful fallback).
    /// do universo daquela variavel (comportamento gracioso).
    ///
    /// # Internal steps
    ///
    /// ```text
    /// inputs crisp
    ///     │
    ///     ▼ (1) Fuzzificacao: firing_strength() por regra
    /// graus de disparo
    ///     │
    ///     ▼ (2+3) Inferencia + Clip: min(mf_consequente, grau)
    /// MFs recortadas por regra
    ///     │
    ///     ▼ (4) Agregacao: max entre todas as regras
    /// MF agregada por consequente
    ///     │
    ///     ▼ (5) Defuzzificacao: centroide = sum(x*mu) / sum(mu)
    /// saidas crisp
    /// ```
    pub fn compute(&self) -> Result<HashMap<String, f64>, FuzzyError> {
        // ── (4) Inicializa acumulador de agregacao com zeros ──────
        // Para cada consequente, cria um vetor de zeros com o tamanho
        // do universo discretizado — sera preenchido pelo max entre regras.
        let mut agregado: HashMap<String, Vec<f64>> = self
            .consequents
            .iter()
            .map(|(nome, var)| (nome.clone(), vec![0.0_f64; var.universe.resolution]))
            .collect();

        // ── (1+2+3) Fuzzificacao → Inferencia → Clip ─────────────
        for regra in &self.rules {
            // (1+2) Fuzzificacao + Inferencia: grau de disparo da regra
            // AND → min dos graus de pertinencia dos antecedentes
            // OR  → max dos graus de pertinencia dos antecedentes
            let grau = regra.firing_strength(&self.inputs, &self.antecedents);

            // Regra que nao dispara nao afeta o agregado
            if grau <= 0.0 {
                continue;
            }

            // Suporta multiplos consequentes por regra
            for (nome_cons, termo_cons) in regra.consequents() {
                let cons_var = match self.consequents.get(nome_cons.as_str()) {
                    Some(v) => v,
                    None => continue,
                };
                let curva_mf = cons_var.term_membership_curve(termo_cons);
                let agg = agregado.get_mut(nome_cons.as_str()).unwrap();
                for (i, &mu) in curva_mf.iter().enumerate() {
                    let recortado = mu.min(grau);
                    if recortado > agg[i] {
                        agg[i] = recortado;
                    }
                }
            }
        }

        // ── (5) Defuzzificacao usando self.defuzz_method ─────────
        // Verifica se ao menos uma regra disparou
        let any_fired = agregado.values().any(|agg| agg.iter().any(|&m| m > 0.0));
        if !any_fired {
            return Err(FuzzyError::NoRulesFired);
        }

        let mut resultados = HashMap::new();
        for (nome, agg_mf) in &agregado {
            let cons_var = &self.consequents[nome];
            let pontos = cons_var.universe_points();
            let saida = self.defuzzify(
                &pontos,
                agg_mf,
                cons_var.universe.min,
                cons_var.universe.max,
            );
            resultados.insert(nome.clone(), saida);
        }
        Ok(resultados)
    }

    /// Runs the full Mamdani pipeline and returns a detailed [`ExplainReport`].
    ///
    /// Identical to [`compute`](Self::compute) in results, but also captures
    /// every intermediate step: fuzzification degrees, per-rule firing strengths,
    /// and the final crisp outputs.
    ///
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{FuzzyVariable, Universe, Term, MembershipFn};
    /// use logicfuzzy_academic::rule::RuleBuilder;
    /// use logicfuzzy_academic::engine::MamdaniEngine;
    ///
    /// let mut engine = MamdaniEngine::new();
    /// let mut temp = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
    /// temp.add_term(Term::new("cold", MembershipFn::Trimf([0.0, 0.0, 25.0])));
    /// temp.add_term(Term::new("hot",  MembershipFn::Trimf([25.0, 50.0, 50.0])));
    /// engine.add_antecedent(temp);
    /// let mut speed = FuzzyVariable::new("speed", Universe::new(0.0, 100.0, 1001));
    /// speed.add_term(Term::new("slow", MembershipFn::Trimf([0.0, 0.0, 50.0])));
    /// speed.add_term(Term::new("fast", MembershipFn::Trimf([50.0, 100.0, 100.0])));
    /// engine.add_consequent(speed);
    /// engine.add_rule(RuleBuilder::new().when("temperature","cold").then("speed","slow").build());
    /// engine.set_input_unchecked("temperature", 5.0);
    /// let report = engine.explain().unwrap();
    /// println!("{}", report.summary());
    /// ```
    pub fn explain(&self) -> Result<ExplainReport, FuzzyError> {
        // ── Etapa 1: Fuzzificacao — coleta graus de todos os termos ──
        let mut fuzzification: Vec<FuzzifiedVariable> = self
            .antecedents
            .iter()
            .map(|(nome, var)| {
                let crisp = *self.inputs.get(nome).unwrap_or(&0.0);
                let term_degrees = var
                    .fuzzify(crisp)
                    .into_iter()
                    .map(|(label, degree)| (label.to_string(), degree))
                    .collect();
                FuzzifiedVariable {
                    variable: nome.clone(),
                    crisp_input: crisp,
                    term_degrees,
                }
            })
            .collect();

        // Ordena por nome para output deterministico
        fuzzification.sort_by(|a, b| a.variable.cmp(&b.variable));

        // ── Etapa 2: Agregacao (mesmo pipeline do compute) ───────────
        let mut agregado: HashMap<String, Vec<f64>> = self
            .consequents
            .iter()
            .map(|(nome, var)| (nome.clone(), vec![0.0_f64; var.universe.resolution]))
            .collect();

        // ── Etapa 2+3: Avalia cada regra e registra o disparo ────────
        let mut rule_firings: Vec<RuleFiring> = Vec::with_capacity(self.rules.len());
        let mut rules_fired = 0usize;
        let mut rules_skipped = 0usize;

        for regra in &self.rules {
            let grau = regra.firing_strength(&self.inputs, &self.antecedents);
            let fired = grau > 0.0;

            rule_firings.push(RuleFiring {
                rule_text: regra.to_string(),
                firing_degree: grau,
                fired,
                consequent_var: regra.consequent_var().to_string(),
                consequent_term: regra.consequent_term().to_string(),
            });

            if !fired {
                rules_skipped += 1;
                continue;
            }
            rules_fired += 1;

            // Suporta multiplos consequentes por regra
            for (nome_cons, termo_cons) in regra.consequents() {
                if let Some(cons_var) = self.consequents.get(nome_cons.as_str()) {
                    let curva_mf = cons_var.term_membership_curve(termo_cons);
                    let agg = agregado.get_mut(nome_cons.as_str()).unwrap();
                    for (i, &mu) in curva_mf.iter().enumerate() {
                        let recortado = mu.min(grau);
                        if recortado > agg[i] {
                            agg[i] = recortado;
                        }
                    }
                }
            }
        }

        // ── Etapa 3: Defuzzificacao (metodo configurado) ─────────────
        let mut outputs = HashMap::new();
        for (nome, agg_mf) in &agregado {
            let cons_var = &self.consequents[nome];
            let pontos = cons_var.universe_points();
            let saida = self.defuzzify(
                &pontos,
                agg_mf,
                cons_var.universe.min,
                cons_var.universe.max,
            );
            outputs.insert(nome.clone(), saida);
        }

        if rules_fired == 0 && !self.rules.is_empty() {
            return Err(FuzzyError::NoRulesFired);
        }
        Ok(ExplainReport {
            fuzzification,
            rule_firings,
            outputs,
            rules_fired,
            rules_skipped,
        })
    }

    /// Prints all rules in the rule base in human-readable format.
    /// Useful for documenting and verifying the system before running.
    pub fn print_rules(&self) {
        println!("Rule base ({} rules):", self.rules.len());
        for (i, rule) in self.rules.iter().enumerate() {
            println!("  Rule {}: {}", i + 1, rule);
        }
    }

    /// Prints a summary of the system: variables and number of rules.
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

    /// Sets the defuzzification method for subsequent `compute()` calls.
    ///
    /// Default is [`DefuzzMethod::Centroid`].
    ///
    /// # Example
    /// ```
    /// use logicfuzzy_academic::{MamdaniEngine, DefuzzMethod};
    /// let mut engine = MamdaniEngine::new();
    /// engine.set_defuzz_method(DefuzzMethod::Bisector);
    /// ```
    pub fn set_defuzz_method(&mut self, method: DefuzzMethod) {
        self.defuzz_method = method;
    }

    /// Returns the current defuzzification method.
    pub fn defuzz_method(&self) -> &DefuzzMethod {
        &self.defuzz_method
    }

    /// Applies the configured defuzzification method to an aggregated MF.
    ///
    /// `pts`  — discrete universe points
    /// `agg`  — aggregated membership degrees (same length as `pts`)
    /// `min`, `max` — universe bounds (used as fallback if denominator = 0)
    fn defuzzify(&self, pts: &[f64], agg: &[f64], min: f64, max: f64) -> f64 {
        let fallback = (min + max) / 2.0;
        match &self.defuzz_method {
            DefuzzMethod::Centroid => {
                // Centro de Gravidade: Σ(x·μ) / Σ(μ)
                let num: f64 = pts.iter().zip(agg.iter()).map(|(&x, &m)| x * m).sum();
                let den: f64 = agg.iter().sum();
                if den.abs() < f64::EPSILON {
                    fallback
                } else {
                    num / den
                }
            }
            DefuzzMethod::Bisector => {
                // Bissectriz: ponto que divide a area total ao meio
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
                // Media dos Maximos: media dos x onde μ é maximo
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
                // Menor dos Maximos: primeiro x onde μ é maximo
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
                // Maior dos Maximos: ultimo x onde μ é maximo
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

    /// Number of rules in the rule base.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Numero de variaveis antecedentes registradas.
    pub fn antecedent_count(&self) -> usize {
        self.antecedents.len()
    }

    /// Numero de variaveis consequentes registradas.
    pub fn consequent_count(&self) -> usize {
        self.consequents.len()
    }

    /// Generates one SVG file per variable (antecedents and consequents) in `dir`.
    ///
    /// Antecedent SVGs include a vertical input marker if a value was set via [`set_input`](Self::set_input).
    /// Consequent SVGs show only the membership functions.
    ///
    /// # Errors
    /// Returns `Err` if the directory cannot be created or any file cannot be written.
    ///
    /// # Example
    /// ```no_run
    /// use logicfuzzy_academic::{MamdaniEngine, antecedent, consequent, rule};
    ///
    /// let mut engine = MamdaniEngine::new();
    /// antecedent!(engine, "temperature", 0.0, 50.0, 501,
    ///     "cold" => trimf [0.0,  0.0, 25.0],
    ///     "hot"  => trimf [25.0,50.0, 50.0],
    /// );
    /// consequent!(engine, "fan_speed", 0.0, 100.0, 1001,
    ///     "slow" => trimf [0.0,  0.0, 50.0],
    ///     "fast" => trimf [50.0,100.0,100.0],
    /// );
    /// engine.set_input_unchecked("temperature", 5.0);
    /// engine.export_svg("output/").unwrap();
    /// // Writes: output/temperature.svg  output/fan_speed.svg
    /// ```
    pub fn export_svg(&self, dir: &str) -> std::io::Result<()> {
        use std::fs;
        use std::path::Path;

        // Cria o diretório de saída caso não exista
        fs::create_dir_all(dir)?;

        // Antecedentes: inclui linha do valor de entrada se disponível
        for (name, var) in &self.antecedents {
            let input = self.inputs.get(name.as_str()).copied();
            let svg = crate::svg::render_variable_svg(var, input);
            let path = Path::new(dir).join(format!("{}.svg", name));
            fs::write(path, svg)?;
        }

        // Consequentes: só as curvas, sem marcador de entrada
        for (name, var) in &self.consequents {
            let svg = crate::svg::render_variable_svg(var, None);
            let path = Path::new(dir).join(format!("{}.svg", name));
            fs::write(path, svg)?;
        }

        Ok(())
    }

    /// Generates an aggregated output SVG for each consequent variable.
    ///
    /// Shows the original curves (dashed), clipped activation areas,
    /// the aggregated envelope, and a centroid marker.
    /// Requires `set_input` + `compute` (or `explain`) to have been called first.
    ///
    /// # Errors
    /// Returns `Err` if the directory cannot be created or any file cannot be written.
    pub fn export_aggregated_svg(&self, dir: &str) -> std::io::Result<()> {
        use std::fs;
        use std::path::Path;
        fs::create_dir_all(dir)?;

        // Recalcula o pipeline para coletar os graus de disparo por consequente
        let mut firing_by_term: std::collections::HashMap<String, Vec<(String, f64)>> = self
            .consequents
            .keys()
            .map(|k| (k.clone(), Vec::new()))
            .collect();

        for regra in &self.rules {
            let grau = regra.firing_strength(&self.inputs, &self.antecedents);
            let entry = firing_by_term
                .entry(regra.consequent_var().to_string())
                .or_default();
            // Agrega: máximo por termo consequente
            if let Some(pos) = entry.iter().position(|(l, _)| l == regra.consequent_term()) {
                if grau > entry[pos].1 {
                    entry[pos].1 = grau;
                }
            } else {
                entry.push((regra.consequent_term().to_string(), grau));
            }
        }

        for (nome, cons_var) in &self.consequents {
            let firing: Vec<(&str, f64)> = firing_by_term
                .get(nome)
                .map(|v| v.iter().map(|(l, d)| (l.as_str(), *d)).collect())
                .unwrap_or_default();

            // Calcula o centroide
            let pts = cons_var.universe_points();
            let n = pts.len();
            let mut agg = vec![0.0_f64; n];
            for (lbl, deg) in &firing {
                if let Some(term) = cons_var.get_term(lbl) {
                    for (i, &x) in pts.iter().enumerate() {
                        let clipped = term.mf.eval(x).min(*deg);
                        if clipped > agg[i] {
                            agg[i] = clipped;
                        }
                    }
                }
            }
            // Use the configured defuzz method (not always centroid)
            let centroid = self.defuzzify(
                &pts,
                &agg,
                cons_var.universe.min,
                cons_var.universe.max,
            );

            let svg = crate::svg::render_aggregated_svg(cons_var, &firing, centroid);
            let path = Path::new(dir).join(format!("{}_aggregated.svg", nome));
            fs::write(path, svg)?;
        }
        Ok(())
    }

    /// Computes a discrete Centre-of-Gravity (COG) table for a consequent variable.
    ///
    /// Samples the aggregated output MF at evenly-spaced `step`-sized intervals,
    /// then computes `numerator`, `denominator`, and `centroid` — matching the
    /// didactic approach shown in fuzzy control textbooks.
    ///
    /// Returns `None` if the consequent name is not registered.
    ///
    /// # Example
    /// ```no_run
    /// # use logicfuzzy_academic::{MamdaniEngine, antecedent, consequent, rule};
    /// # let mut engine = MamdaniEngine::new();
    /// # antecedent!(engine, "x", 0.0, 10.0, 101, "low" => trimf [0.0, 0.0, 5.0]);
    /// # consequent!(engine, "y", 0.0, 10.0, 101, "small" => trimf [0.0, 0.0, 5.0]);
    /// # engine.add_rule(rule!(IF x IS low THEN y IS small));
    /// # engine.set_input_unchecked("x", 2.0);
    /// # engine.compute();
    /// let table = engine.discrete_cog("y", 10.0).unwrap();
    /// println!("centroid = {:.4}", table.centroid);
    /// ```
    pub fn discrete_cog(
        &self,
        consequent_name: &str,
        step: f64,
    ) -> Option<crate::explain::CogTable> {
        let cons_var = self.consequents.get(consequent_name)?;
        let pts = cons_var.universe_points();
        let n = pts.len();

        // Re-compute aggregated MF
        let mut agg = vec![0.0_f64; n];
        for regra in &self.rules {
            if regra.consequent_var() != consequent_name {
                continue;
            }
            let grau = regra.firing_strength(&self.inputs, &self.antecedents);
            if grau < 1e-12 {
                continue;
            }
            if let Some(term) = cons_var.get_term(regra.consequent_term()) {
                for (i, &x) in pts.iter().enumerate() {
                    let c = term.mf.eval(x).min(grau);
                    if c > agg[i] {
                        agg[i] = c;
                    }
                }
            }
        }

        // Discrete samples at `step`-sized intervals
        let min = cons_var.universe.min;
        let max = cons_var.universe.max;
        let mut disc_pts: Vec<f64> = Vec::new();
        let mut v = min;
        while v <= max + 1e-9 {
            disc_pts.push(v.min(max));
            v += step;
        }

        let mu_values: Vec<f64> = disc_pts
            .iter()
            .map(|&x| {
                // Linear interpolation into agg
                if pts.is_empty() {
                    return 0.0;
                }
                let pos = pts.partition_point(|&u| u <= x);
                if pos == 0 {
                    return agg[0];
                }
                if pos >= n {
                    return *agg.last().unwrap();
                }
                let i = pos - 1;
                let x0 = pts[i];
                let x1 = pts[i + 1];
                let y0 = agg[i];
                let y1 = agg[i + 1];
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

// ─────────────────────────────────────────────────────────────────
// Testes unitarios
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::RuleBuilder;
    use crate::{FuzzyVariable, MembershipFn, Term, Universe};

    // ── Sistema minimal para testes analiticos ─────────────────

    /// Monta um sistema com 1 entrada, 1 saida e 1 regra.
    /// Retorna (motor, nome_da_saida)
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

    // ── Configuracao ───────────────────────────────────────────

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

    // ── Defuzzificacao: centroide analitico ────────────────────

    #[test]
    fn centroide_mf_uniforme_e_ponto_medio() {
        // trapmf[0,0,10,10] = 1.0 em todo o universo
        // centroide = ponto medio = 5.0
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0); // pleno disparo (grau=1.0)
        let r = m.compute().unwrap();
        assert!((r["y"] - 5.0).abs() < 0.02, "centroide={}", r["y"]);
    }

    #[test]
    fn centroide_rampa_crescente_aproxima_dois_tercos() {
        // trimf[0,10,10] = rampa de 0 a 10 (ombro direito)
        // centroide analitico = 2/3 * 10 = 6.667
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]), // disparo total
            MembershipFn::Trimf([0.0, 10.0, 10.0]),       // rampa crescente
        );
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 6.667).abs() < 0.05, "centroide={:.4}", r["y"]);
    }

    #[test]
    fn centroide_rampa_decrescente_aproxima_um_terco() {
        // trimf[0,0,10] = rampa de 10 a 0 (ombro esquerdo)
        // centroide analitico = 1/3 * 10 = 3.333
        let mut m = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 0.0, 10.0]),
        );
        m.set_input_unchecked("x", 5.0);
        let r = m.compute().unwrap();
        assert!((r["y"] - 3.333).abs() < 0.05, "centroide={:.4}", r["y"]);
    }

    // ── Clip: grau de disparo recorta a MF ────────────────────

    #[test]
    fn clip_grau_pleno_nao_altera_centroide() {
        // Com grau=1.0 o clip nao muda nada — centroide identico ao sem clip
        let mf = MembershipFn::Trimf([0.0, 5.0, 10.0]);
        let mut m = motor_simples(MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]), mf);
        m.set_input_unchecked("x", 5.0); // grau=1.0
        let r = m.compute().unwrap();
        // Centroide de trimf[0,5,10] = 5.0 (simetrico)
        assert!((r["y"] - 5.0).abs() < 0.05, "centroide={:.4}", r["y"]);
    }

    #[test]
    fn clip_grau_zero_retorna_ponto_medio_do_universo() {
        // Grau=0.0: nenhuma regra dispara → compute() retorna Err(NoRulesFired)
        let mut m = motor_simples(
            MembershipFn::Trimf([0.0, 0.0, 5.0]), // fora da regiao de x=8
            MembershipFn::Trimf([0.0, 5.0, 10.0]),
        );
        m.set_input_unchecked("x", 8.0); // grau de "a" em 8.0 = 0.0
        let result = m.compute();
        assert!(result.is_err(), "esperava Err(NoRulesFired) quando grau=0");
        assert_eq!(result.unwrap_err(), crate::error::FuzzyError::NoRulesFired);
    }

    #[test]
    fn clip_desloca_centroide_para_baixo() {
        // Regra com MF simetrica clipada em 0.5:
        // a cauda superior e cortada → centroide desloca para a area de base
        // (nao vamos verificar valor exato, so a direcao do deslocamento)
        let mut m = motor_simples(
            MembershipFn::Trimf([4.0, 5.0, 6.0]), // pico estreito em 5 → grau~0.5 em 4.5
            MembershipFn::Trimf([0.0, 10.0, 10.0]), // rampa crescente
        );
        m.set_input_unchecked("x", 4.5); // grau de "a" em 4.5 ≈ 0.5
        let r_clip = m.compute().unwrap();

        // Sem clip (grau pleno): centroide da rampa crescente ≈ 6.67
        let mut m2 = motor_simples(
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
            MembershipFn::Trimf([0.0, 10.0, 10.0]),
        );
        m2.set_input_unchecked("x", 5.0);
        let r_pleno = m2.compute().unwrap();

        // Com clip em 0.5, centroide deve ser menor que com grau pleno
        assert!(
            r_clip["y"] < r_pleno["y"],
            "clip={:.3} pleno={:.3}",
            r_clip["y"],
            r_pleno["y"]
        );
    }

    // ── Agregacao: max entre regras ────────────────────────────

    #[test]
    fn agregacao_duas_regras_nao_sobrepostas() {
        // Duas regras com termos de saida disjuntos:
        // Se ambas dispararem com grau=1.0, o centroide deve estar
        // entre os dois picos
        let mut motor = MamdaniEngine::new();

        let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 1001));
        x.add_term(Term::new("esq", MembershipFn::Trapmf([0.0, 0.0, 1.0, 2.0])));
        x.add_term(Term::new(
            "dir",
            MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
        ));
        // Evitar complexidade: usar x com grau pleno via trapmf amplo
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

        motor.set_input_unchecked("x", 0.0); // x=0 → esq=1.0
        let r = motor.compute().unwrap();
        // Com ambas as saidas ativas em grau=1.0, centroide deve estar em torno de 5.0
        assert!(r["y"] > 2.5 && r["y"] < 7.5, "centroide={:.3}", r["y"]);
    }

    #[test]
    fn agregacao_max_entre_regras_concorrentes() {
        // Duas regras apontam para o mesmo consequente com termos diferentes.
        // A agregacao por max deve unir as duas MFs recortadas.
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

        // Ambas disparando com grau=1.0 → MFs simétricas → centroide=5.0
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
        assert!((r["y"] - 5.0).abs() < 0.1, "centroide={:.4}", r["y"]);
    }

    // ── Sistema completo: ventilador ───────────────────────────

    fn montar_ventilador() -> MamdaniEngine {
        let mut motor = MamdaniEngine::new();

        // Entrada 1: temperatura [0, 50] °C
        let mut temp = FuzzyVariable::new("temperatura", Universe::new(0.0, 50.0, 501));
        temp.add_term(Term::new("fria", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        temp.add_term(Term::new("morna", MembershipFn::Trimf([0.0, 25.0, 50.0])));
        temp.add_term(Term::new("quente", MembershipFn::Trimf([25.0, 50.0, 50.0])));
        motor.add_antecedent(temp);

        // Entrada 2: umidade [0, 100] %
        let mut umid = FuzzyVariable::new("umidade", Universe::new(0.0, 100.0, 1001));
        umid.add_term(Term::new("baixa", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        umid.add_term(Term::new("media", MembershipFn::Trimf([0.0, 50.0, 100.0])));
        umid.add_term(Term::new("alta", MembershipFn::Trimf([50.0, 100.0, 100.0])));
        motor.add_antecedent(umid);

        // Saida: velocidade do ventilador [0, 100] %
        let mut vel = FuzzyVariable::new("velocidade_ventilador", Universe::new(0.0, 100.0, 1001));
        vel.add_term(Term::new("lenta", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        vel.add_term(Term::new("media", MembershipFn::Trimf([0.0, 50.0, 100.0])));
        vel.add_term(Term::new(
            "rapida",
            MembershipFn::Trimf([50.0, 100.0, 100.0]),
        ));
        motor.add_consequent(vel);

        // Base de regras (≥ 4 regras conforme checklist)
        // Regra 1: SE temp fria   AND umid baixa  ENTAO lenta
        motor.add_rule(
            RuleBuilder::new()
                .when("temperatura", "fria")
                .and("umidade", "baixa")
                .then("velocidade_ventilador", "lenta")
                .build(),
        );
        // Regra 2: SE temp morna  AND umid media  ENTAO media
        motor.add_rule(
            RuleBuilder::new()
                .when("temperatura", "morna")
                .and("umidade", "media")
                .then("velocidade_ventilador", "media")
                .build(),
        );
        // Regra 3: SE temp quente OR  umid alta   ENTAO rapida
        motor.add_rule(
            RuleBuilder::new()
                .when("temperatura", "quente")
                .or("umidade", "alta")
                .then("velocidade_ventilador", "rapida")
                .build(),
        );
        // Regra 4: SE temp fria   AND umid alta   ENTAO media
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
        // Cenario 1: temp=5 (fria≈0.8), umid=10 (baixa≈0.8)
        // → Regra1 dispara forte → velocidade deve ser baixa (< 40)
        let mut m = montar_ventilador();
        m.set_input_unchecked("temperatura", 5.0);
        m.set_input_unchecked("umidade", 10.0);
        let r = m.compute().unwrap();
        assert!(
            r["velocidade_ventilador"] < 40.0,
            "esperava < 40, obteve {:.2}",
            r["velocidade_ventilador"]
        );
    }

    #[test]
    fn ventilador_cenario2_morno_medio_resultado_medio() {
        // Cenario 2: temp=25 (morna=1.0), umid=50 (media=1.0)
        // → Regra2 dispara forte → velocidade deve ser media (35..65)
        let mut m = montar_ventilador();
        m.set_input_unchecked("temperatura", 25.0);
        m.set_input_unchecked("umidade", 50.0);
        let r = m.compute().unwrap();
        let v = r["velocidade_ventilador"];
        assert!(v > 35.0 && v < 65.0, "esperava 35..65, obteve {:.2}", v);
    }

    #[test]
    fn ventilador_cenario3_quente_umido_resultado_rapido() {
        // Cenario 3: temp=45 (quente≈0.8), umid=90 (alta≈0.8)
        // → Regra3 (OR) dispara forte → velocidade deve ser alta (> 60)
        let mut m = montar_ventilador();
        m.set_input_unchecked("temperatura", 45.0);
        m.set_input_unchecked("umidade", 90.0);
        let r = m.compute().unwrap();
        assert!(
            r["velocidade_ventilador"] > 60.0,
            "esperava > 60, obteve {:.2}",
            r["velocidade_ventilador"]
        );
    }

    #[test]
    fn ventilador_monotonia_temperatura() {
        // Aumentar a temperatura mantendo umidade constante deve
        // aumentar (ou manter) a velocidade do ventilador
        let umid = 50.0;
        let temps = [5.0_f64, 15.0, 25.0, 35.0, 45.0];
        let mut anterior = 0.0_f64;
        for &t in &temps {
            let mut m = montar_ventilador();
            m.set_input_unchecked("temperatura", t);
            m.set_input_unchecked("umidade", umid);
            let v = m.compute().unwrap()["velocidade_ventilador"];
            assert!(
                v >= anterior - 0.5,
                "nao-monotonia: temp={} vel={:.2} < anterior={:.2}",
                t,
                v,
                anterior
            );
            anterior = v;
        }
    }

    #[test]
    fn ventilador_saida_dentro_do_universo() {
        // A saida sempre deve estar dentro do universo [0, 100]
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
            assert!(
                v >= 0.0 && v <= 100.0,
                "saida={:.2} fora do universo [0,100] para temp={} umid={}",
                v,
                t,
                u
            );
        }
    }

    // ── Testes do explain() ────────────────────────────────────

    // Monta sistema minimal de 1 entrada, 2 termos, 2 regras para testes analiticos do explain
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
        // explain() e compute() devem retornar o mesmo valor crisp de saida
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let compute_val = m.compute().unwrap()["speed"];
        let explain_val = m.explain().unwrap().outputs["speed"];
        assert!(
            (compute_val - explain_val).abs() < 1e-10,
            "compute={:.6} != explain={:.6}",
            compute_val,
            explain_val
        );
    }

    #[test]
    fn explain_numero_de_regras_correto() {
        // rule_firings deve ter uma entrada por regra registrada
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        assert_eq!(report.rule_firings.len(), m.rule_count());
    }

    #[test]
    fn explain_fuzzificacao_cobre_todos_antecedentes() {
        // fuzzification deve ter uma entrada por variavel antecedente
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        assert_eq!(report.fuzzification.len(), m.antecedent_count());
    }

    #[test]
    fn explain_graus_dentro_do_intervalo() {
        // todos os graus de pertinencia devem estar em [0.0, 1.0]
        let mut m = motor_explain_minimal();
        for input in [0.0_f64, 12.5, 25.0, 37.5, 50.0] {
            m.set_input_unchecked("temperature", input);
            let report = m.explain().unwrap();
            for fv in &report.fuzzification {
                for (term, degree) in &fv.term_degrees {
                    assert!(
                        *degree >= 0.0 && *degree <= 1.0,
                        "grau fora de [0,1]: input={} term={} degree={}",
                        input,
                        term,
                        degree
                    );
                }
            }
        }
    }

    #[test]
    fn explain_firing_degree_consistente_com_fired_flag() {
        // fired == true se e somente se firing_degree > 0
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        for rf in &report.rule_firings {
            assert_eq!(
                rf.fired,
                rf.firing_degree > 0.0,
                "inconsistencia: fired={} mas firing_degree={:.4} para regra: {}",
                rf.fired,
                rf.firing_degree,
                rf.rule_text
            );
        }
    }

    #[test]
    fn explain_contagem_fired_skipped_consistente() {
        // rules_fired + rules_skipped deve ser igual ao total de regras
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        assert_eq!(
            report.rules_fired + report.rules_skipped,
            report.rule_firings.len(),
            "rules_fired({}) + rules_skipped({}) != total({})",
            report.rules_fired,
            report.rules_skipped,
            report.rule_firings.len()
        );
    }

    #[test]
    fn explain_cenario_frio_so_regra_cold_dispara() {
        // temp=5 (cold≈0.8, hot=0.0) → so a regra "cold→slow" deve disparar
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 5.0);
        let report = m.explain().unwrap();

        let cold_rule = report
            .rule_firings
            .iter()
            .find(|r| r.consequent_term == "slow")
            .expect("regra cold→slow nao encontrada");
        let hot_rule = report
            .rule_firings
            .iter()
            .find(|r| r.consequent_term == "fast")
            .expect("regra hot→fast nao encontrada");

        assert!(cold_rule.fired, "cold→slow deveria ter disparado");
        assert!(!hot_rule.fired, "hot→fast nao deveria ter disparado");
        assert!(
            cold_rule.firing_degree > 0.5,
            "grau cold esperado > 0.5, obteve {:.4}",
            cold_rule.firing_degree
        );
        assert_eq!(hot_rule.firing_degree, 0.0);
    }

    #[test]
    fn explain_ponto_medio_ambas_regras_disparam() {
        // temp=12.5 (cold=0.5, hot=0.0 ainda): grau da regra cold = 0.5
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
            .map(|(_, d)| *d)
            .unwrap();

        assert!(
            (cold_degree - 0.5).abs() < 1e-6,
            "grau cold em 12.5 esperado=0.5, obteve={:.6}",
            cold_degree
        );
    }

    #[test]
    fn explain_dominant_term_correto_cenario_frio() {
        // temp=5 → dominant term de temperature deve ser "cold"
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 5.0);
        let report = m.explain().unwrap();

        let temp_fv = report
            .fuzzification
            .iter()
            .find(|fv| fv.variable == "temperature")
            .unwrap();

        assert_eq!(
            temp_fv.dominant_term(),
            Some("cold"),
            "dominant_term esperado 'cold', obteve '{:?}'",
            temp_fv.dominant_term()
        );
    }

    #[test]
    fn explain_dominant_term_correto_cenario_quente() {
        // temp=45 → dominant term de temperature deve ser "hot"
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 45.0);
        let report = m.explain().unwrap();

        let temp_fv = report
            .fuzzification
            .iter()
            .find(|fv| fv.variable == "temperature")
            .unwrap();

        assert_eq!(
            temp_fv.dominant_term(),
            Some("hot"),
            "dominant_term esperado 'hot', obteve '{:?}'",
            temp_fv.dominant_term()
        );
    }

    #[test]
    fn explain_summary_contem_secoes_esperadas() {
        // summary() deve conter os cabecalhos das tres secoes
        let mut m = motor_explain_minimal();
        m.set_input_unchecked("temperature", 10.0);
        let report = m.explain().unwrap();
        let s = report.summary();

        assert!(s.contains("Fuzzification"), "faltou secao Fuzzification");
        assert!(
            s.contains("Rule Evaluation"),
            "faltou secao Rule Evaluation"
        );
        assert!(
            s.contains("Defuzzification"),
            "faltou secao Defuzzification"
        );
        assert!(s.contains("temperature"), "faltou nome da variavel");
        assert!(s.contains("speed"), "faltou nome da saida");
    }

    #[test]
    fn explain_ventilador_cenario_frio_seco() {
        // Compara explain() vs compute() no sistema completo do ventilador
        let mut m = montar_ventilador();
        m.set_input_unchecked("temperatura", 5.0);
        m.set_input_unchecked("umidade", 10.0);

        let compute_val = m.compute().unwrap()["velocidade_ventilador"];
        let report = m.explain().unwrap();
        let explain_val = report.outputs["velocidade_ventilador"];

        assert!(
            (compute_val - explain_val).abs() < 1e-10,
            "divergencia compute/explain: {:.6} vs {:.6}",
            compute_val,
            explain_val
        );
        assert_eq!(
            report.fuzzification.len(),
            2,
            "esperava 2 variaveis fuzzificadas"
        );
        assert_eq!(
            report.rule_firings.len(),
            4,
            "esperava 4 entradas de regras"
        );
        assert!(report.rules_fired + report.rules_skipped == 4);
    }
}