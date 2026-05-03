//! # macros.rs
//!
//! Macros declarativas para construir sistemas Fuzzy com menos verbosidade.
//!
//! ## Macros disponíveis
//!
//! | Macro | O que faz |
//! |---|---|
//! | `rule!` | Cria uma regra IF–THEN em linguagem natural |
//! | `fuzzy_var!` | Cria um `FuzzyVariable` com universo e termos |
//! | `antecedent!` | Cria e registra uma variável de entrada no engine |
//! | `consequent!` | Cria e registra uma variável de saída no engine |
//! | `var_svg!` | Gera SVG de uma `FuzzyVariable` (com ou sem marcador de entrada) |
//! | `export_svg!` | Exporta todos os SVGs do engine para um diretório |
//!
//! ## Funções de pertinência suportadas nas macros
//!
//! ```text
//! "label" => trimf   [a, b, c]
//! "label" => trapmf  [a, b, c, d]
//! "label" => gaussmf { mean: m, sigma: s }
//! ```

// ─────────────────────────────────────────────────────────────────
// __make_term! — helper interno, nao faz parte da API publica
// ─────────────────────────────────────────────────────────────────

/// Internal helper — converts (label, mf_type, params) into a `Term`.
/// Not part of the public API. Used by `fuzzy_var!`, `antecedent!`, `consequent!`.
#[doc(hidden)]
#[macro_export]
macro_rules! __make_term {
    // Triangular: [a, b, c]
    ($label:literal, trimf, [$a:expr, $b:expr, $c:expr]) => {
        $crate::Term::new(
            $label,
            $crate::MembershipFn::Trimf([$a as f64, $b as f64, $c as f64]),
        )
    };
    // Trapezoidal: [a, b, c, d]
    ($label:literal, trapmf, [$a:expr, $b:expr, $c:expr, $d:expr]) => {
        $crate::Term::new(
            $label,
            $crate::MembershipFn::Trapmf([$a as f64, $b as f64, $c as f64, $d as f64]),
        )
    };
    // Gaussiana: { mean: m, sigma: s }
    ($label:literal, gaussmf, { mean: $mean:expr, sigma: $sigma:expr }) => {
        $crate::Term::new(
            $label,
            $crate::MembershipFn::Gaussmf {
                mean: $mean as f64,
                sigma: $sigma as f64,
            },
        )
    };
}

// ─────────────────────────────────────────────────────────────────
// fuzzy_var! — cria FuzzyVariable com todos os termos
// ─────────────────────────────────────────────────────────────────

/// Creates a [`FuzzyVariable`](crate::FuzzyVariable) with its universe and all linguistic terms
/// in a single expression — without needing multiple `add_term` calls.
#[macro_export]
macro_rules! fuzzy_var {
    (
        $name:literal, $min:literal, $max:literal, $res:literal,
        $( $label:literal => $mf:ident $params:tt ),*
        $(,)?
    ) => {{
        let mut __var = $crate::FuzzyVariable::new(
            $name,
            $crate::Universe::new($min as f64, $max as f64, $res as usize),
        );
        $(
            __var.add_term($crate::__make_term!($label, $mf, $params));
        )*
        __var
    }};
}

// ─────────────────────────────────────────────────────────────────
// antecedent! — cria e registra variavel de entrada no engine
// ─────────────────────────────────────────────────────────────────

/// Creates a [`FuzzyVariable`](crate::FuzzyVariable) and registers it as an antecedent
/// (input variable) in the given [`MamdaniEngine`](crate::MamdaniEngine).
#[macro_export]
macro_rules! antecedent {
    (
        $engine:expr, $name:literal, $min:literal, $max:literal, $res:literal,
        $( $label:literal => $mf:ident $params:tt ),*
        $(,)?
    ) => {
        $engine.add_antecedent(
            $crate::fuzzy_var!(
                $name, $min, $max, $res,
                $( $label => $mf $params ),*
            )
        );
    };
}

// ─────────────────────────────────────────────────────────────────
// consequent! — cria e registra variavel de saida no engine
// ─────────────────────────────────────────────────────────────────

/// Creates a [`FuzzyVariable`](crate::FuzzyVariable) and registers it as a consequent
/// (output variable) in the given [`MamdaniEngine`](crate::MamdaniEngine).
#[macro_export]
macro_rules! consequent {
    (
        $engine:expr, $name:literal, $min:literal, $max:literal, $res:literal,
        $( $label:literal => $mf:ident $params:tt ),*
        $(,)?
    ) => {
        $engine.add_consequent(
            $crate::fuzzy_var!(
                $name, $min, $max, $res,
                $( $label => $mf $params ),*
            )
        );
    };
}

// ─────────────────────────────────────────────────────────────────
// rule! — DSL para regras IF–THEN
// ─────────────────────────────────────────────────────────────────

/// Declarative macro for building Fuzzy IF–THEN rules.
///
/// Expands into [`RuleBuilder`](crate::rule::RuleBuilder) calls at compile time —
/// zero runtime cost, no external dependencies.
///
/// Keywords: `IF`, `IS`, `AND`, `OR`, `THEN`.
/// Identifiers with underscores (e.g. `fan_speed`, `smoke_level`) work normally.
///
/// Supports up to 5 antecedents with uniform connectors (`AND` or `OR`),
/// plus selected mixed patterns (for 3 antecedents). For more complex
/// rules, use the [`RuleBuilder`](crate::rule::RuleBuilder) directly.
///
/// # Examples
/// ```
/// use logicfuzzy_academic::rule;
/// use logicfuzzy_academic::rule::Connector;
///
/// // 2 antecedents
/// let r = rule!(IF temperatura IS quente OR umidade IS alta THEN velocidade_ventilador IS rapida);
///
/// // 3 antecedents
/// let r = rule!(IF a IS x AND b IS y AND c IS z THEN out IS result);
///
/// // 4 antecedents (all AND)
/// let r = rule!(IF a IS x AND b IS y AND c IS z AND d IS w THEN out IS result);
///
/// // 5 antecedents (all OR)
/// let r = rule!(IF a IS x OR b IS y OR c IS z OR d IS w OR e IS v THEN out IS result);
/// ```
#[macro_export]
macro_rules! rule {
    // ── 1 antecedente NOT ─────────────────────────────────────────
    (IF $v1:ident IS NOT $t1:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when_not(stringify!($v1), stringify!($t1))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 1 antecedente ─────────────────────────────────────────────
    (IF $v1:ident IS $t1:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 2 antecedentes: AND ────────────────────────────────────────
    (IF $v1:ident IS $t1:ident AND $v2:ident IS $t2:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .and(stringify!($v2), stringify!($t2))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 2 antecedentes: AND NOT ────────────────────────────────────
    (IF $v1:ident IS $t1:ident AND NOT $v2:ident IS $t2:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .and_not(stringify!($v2), stringify!($t2))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 2 antecedentes: NOT AND ────────────────────────────────────
    (IF $v1:ident IS NOT $t1:ident AND $v2:ident IS $t2:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when_not(stringify!($v1), stringify!($t1))
            .and(stringify!($v2), stringify!($t2))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 2 antecedentes: OR ─────────────────────────────────────────
    (IF $v1:ident IS $t1:ident OR $v2:ident IS $t2:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .or(stringify!($v2), stringify!($t2))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 2 antecedentes: OR NOT ─────────────────────────────────────
    (IF $v1:ident IS $t1:ident OR NOT $v2:ident IS $t2:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .or_not(stringify!($v2), stringify!($t2))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 3 antecedentes: AND AND ────────────────────────────────────
    (IF $v1:ident IS $t1:ident AND $v2:ident IS $t2:ident AND $v3:ident IS $t3:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .and(stringify!($v2), stringify!($t2))
            .and(stringify!($v3), stringify!($t3))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 3 antecedentes: AND OR (mixed) ─────────────────────────────
    (IF $v1:ident IS $t1:ident AND $v2:ident IS $t2:ident OR $v3:ident IS $t3:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .and(stringify!($v2), stringify!($t2))
            .or(stringify!($v3), stringify!($t3))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 4 antecedentes: AND AND AND ────────────────────────────────
    (IF $v1:ident IS $t1:ident AND $v2:ident IS $t2:ident AND $v3:ident IS $t3:ident AND $v4:ident IS $t4:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .and(stringify!($v2), stringify!($t2))
            .and(stringify!($v3), stringify!($t3))
            .and(stringify!($v4), stringify!($t4))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 4 antecedentes: OR OR OR ───────────────────────────────────
    (IF $v1:ident IS $t1:ident OR $v2:ident IS $t2:ident OR $v3:ident IS $t3:ident OR $v4:ident IS $t4:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .or(stringify!($v2), stringify!($t2))
            .or(stringify!($v3), stringify!($t3))
            .or(stringify!($v4), stringify!($t4))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 5 antecedentes: AND AND AND AND ────────────────────────────
    (IF $v1:ident IS $t1:ident AND $v2:ident IS $t2:ident AND $v3:ident IS $t3:ident AND $v4:ident IS $t4:ident AND $v5:ident IS $t5:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .and(stringify!($v2), stringify!($t2))
            .and(stringify!($v3), stringify!($t3))
            .and(stringify!($v4), stringify!($t4))
            .and(stringify!($v5), stringify!($t5))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 5 antecedentes: OR OR OR OR ────────────────────────────────
    (IF $v1:ident IS $t1:ident OR $v2:ident IS $t2:ident OR $v3:ident IS $t3:ident OR $v4:ident IS $t4:ident OR $v5:ident IS $t5:ident THEN $cv:ident IS $ct:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .or(stringify!($v2), stringify!($t2))
            .or(stringify!($v3), stringify!($t3))
            .or(stringify!($v4), stringify!($t4))
            .or(stringify!($v5), stringify!($t5))
            .then(stringify!($cv), stringify!($ct))
            .build()
    };

    // ── 1 antecedente, 2 consequentes (multi-consequent) ──────────
    (IF $v1:ident IS $t1:ident THEN $cv1:ident IS $ct1:ident AND $cv2:ident IS $ct2:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .then(stringify!($cv1), stringify!($ct1))
            .also(stringify!($cv2), stringify!($ct2))
            .build()
    };

    // ── 2 antecedentes AND, 2 consequentes ────────────────────────
    (IF $v1:ident IS $t1:ident AND $v2:ident IS $t2:ident THEN $cv1:ident IS $ct1:ident AND $cv2:ident IS $ct2:ident) => {
        $crate::rule::RuleBuilder::new()
            .when(stringify!($v1), stringify!($t1))
            .and(stringify!($v2), stringify!($t2))
            .then(stringify!($cv1), stringify!($ct1))
            .also(stringify!($cv2), stringify!($ct2))
            .build()
    };
}

// ─────────────────────────────────────────────────────────────────
// var_svg! — gera SVG de uma FuzzyVariable
// ─────────────────────────────────────────────────────────────────

/// Generates an SVG string for a [`FuzzyVariable`](crate::FuzzyVariable).
///
/// Two forms:
/// - `var_svg!(var)` — membership function curves only
/// - `var_svg!(var, value)` — adds a vertical input marker + μ annotations
///
/// Returns a `String` containing self-contained SVG markup.
#[macro_export]
macro_rules! var_svg {
    // Sem marcador de entrada / without input marker
    ($var:expr) => {
        $var.to_svg()
    };
    // Com marcador de entrada / with input marker
    ($var:expr, $value:expr) => {
        $var.to_svg_with_input($value as f64)
    };
}

// ─────────────────────────────────────────────────────────────────
// export_svg! — exporta SVGs do engine para um diretório
// ─────────────────────────────────────────────────────────────────

/// Exports SVG files for all variables of a [`MamdaniEngine`](crate::MamdaniEngine)
/// to a directory, printing a status line for each outcome.
#[macro_export]
macro_rules! export_svg {
    // Somente MFs / membership functions only
    ($engine:expr, $dir:expr) => {{
        match $engine.export_svg($dir) {
            Ok(_) => println!("  ✓  SVGs → {}", $dir),
            Err(e) => println!("  ✗  Error writing SVGs to {}: {}", $dir, e),
        }
    }};
    // MFs + saída agregada / membership + aggregated output
    ($engine:expr, $dir:expr, aggregated) => {{
        match $engine.export_svg($dir) {
            Ok(_) => println!("  ✓  Membership SVGs → {}", $dir),
            Err(e) => println!("  ✗  Error writing membership SVGs to {}: {}", $dir, e),
        }
        match $engine.export_aggregated_svg($dir) {
            Ok(_) => println!("  ✓  Aggregated SVGs → {}", $dir),
            Err(e) => println!("  ✗  Error writing aggregated SVGs to {}: {}", $dir, e),
        }
    }};
}

// ─────────────────────────────────────────────────────────────────
// Testes unitarios
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::rule::Connector;

    #[test]
    fn rule_one_antecedent() {
        let r = rule!(IF temperature IS cold THEN fan_speed IS slow);
        assert_eq!(r.antecedents().len(), 1);
        assert_eq!(r.antecedents()[0].0, "temperature");
        assert_eq!(r.antecedents()[0].1, "cold");
        assert_eq!(r.consequent_var(), "fan_speed");
        assert_eq!(r.consequent_term(), "slow");
    }

    #[test]
    fn rule_two_antecedents_and() {
        let r = rule!(IF temperature IS hot AND humidity IS high THEN fan_speed IS fast);
        assert_eq!(r.antecedents().len(), 2);
        assert_eq!(r.connector(), &Connector::And);
        assert_eq!(r.antecedents()[0].0, "temperature");
        assert_eq!(r.antecedents()[1].0, "humidity");
        assert_eq!(r.consequent_term(), "fast");
    }

    #[test]
    fn rule_two_antecedents_or() {
        let r = rule!(IF temperature IS hot OR humidity IS high THEN fan_speed IS fast);
        assert_eq!(r.antecedents().len(), 2);
        assert_eq!(r.connector(), &Connector::Or);
    }

    #[test]
    fn rule_three_antecedents_all_and() {
        let r = rule!(IF a IS x AND b IS y AND c IS z THEN output IS result);
        assert_eq!(r.antecedents().len(), 3);
        assert_eq!(r.connector(), &Connector::And);
        assert_eq!(r.antecedents()[2].0, "c");
        assert_eq!(r.antecedents()[2].1, "z");
    }

    #[test]
    fn rule_three_antecedents_mixed() {
        let r = rule!(IF a IS x AND b IS y OR c IS z THEN output IS result);
        assert_eq!(r.antecedents().len(), 3);
        assert_eq!(r.connector(), &Connector::Or); // last connector used
        assert_eq!(r.antecedents()[0].0, "a");
        assert_eq!(r.antecedents()[1].0, "b");
        assert_eq!(r.antecedents()[2].0, "c");
    }

    #[test]
    fn rule_four_antecedents_all_and() {
        let r = rule!(IF a IS x AND b IS y AND c IS z AND d IS w THEN output IS result);
        assert_eq!(r.antecedents().len(), 4);
        assert_eq!(r.connector(), &Connector::And);
        assert_eq!(r.antecedents()[3].0, "d");
    }

    #[test]
    fn rule_five_antecedents_all_or() {
        let r = rule!(IF a IS x OR b IS y OR c IS z OR d IS w OR e IS v THEN output IS result);
        assert_eq!(r.antecedents().len(), 5);
        assert_eq!(r.connector(), &Connector::Or);
        assert_eq!(r.antecedents()[4].0, "e");
    }

    #[test]
    fn rule_underscore_identifiers() {
        let r =
            rule!(IF smoke_level IS high AND ambient_temp IS critical THEN alert_level IS maximum);
        assert_eq!(r.antecedents()[0].0, "smoke_level");
        assert_eq!(r.antecedents()[1].0, "ambient_temp");
        assert_eq!(r.consequent_var(), "alert_level");
        assert_eq!(r.consequent_term(), "maximum");
    }

    #[test]
    fn rule_equivalent_to_builder() {
        use crate::rule::RuleBuilder;
        let via_macro =
            rule!(IF temperatura IS quente AND umidade IS alta THEN velocidade IS rapida);
        let via_builder = RuleBuilder::new()
            .when("temperatura", "quente")
            .and("umidade", "alta")
            .then("velocidade", "rapida")
            .build();
        assert_eq!(
            via_macro.antecedents().len(),
            via_builder.antecedents().len()
        );
        assert_eq!(via_macro.connector(), via_builder.connector());
        assert_eq!(via_macro.consequent_var(), via_builder.consequent_var());
        assert_eq!(via_macro.consequent_term(), via_builder.consequent_term());
    }

    #[test]
    fn rule_to_string_readable() {
        let r =
            rule!(IF temperatura IS quente OR umidade IS alta THEN velocidade_ventilador IS rapida);
        let s = r.to_string();
        assert!(s.contains("temperatura"));
        assert!(s.contains("quente"));
        assert!(s.contains("OR"));
        assert!(s.contains("rapida"));
    }

    // ── fuzzy_var! ─────────────────────────────────────────────────

    #[test]
    fn fuzzy_var_cria_com_termos_corretos() {
        let var = fuzzy_var!("temperature", 0.0, 50.0, 501,
            "cold" => trimf [0.0,  0.0, 25.0],
            "warm" => trimf [0.0, 25.0, 50.0],
            "hot"  => trimf [25.0,50.0, 50.0],
        );
        assert_eq!(var.term_count(), 3);
        assert_eq!(var.term_labels(), vec!["cold", "warm", "hot"]);
    }

    #[test]
    fn fuzzy_var_trimf_grau_correto() {
        let var = fuzzy_var!("x", 0.0, 10.0, 101,
            "low" => trimf [0.0, 0.0, 5.0],
        );
        assert!((var.membership_at("low", 0.0) - 1.0).abs() < 1e-6);
        assert!((var.membership_at("low", 6.0)).abs() < 1e-6);
    }

    #[test]
    fn fuzzy_var_trapmf_grau_correto() {
        let var = fuzzy_var!("x", 0.0, 10.0, 101,
            "mid" => trapmf [2.0, 3.0, 7.0, 8.0],
        );
        assert!((var.membership_at("mid", 5.0) - 1.0).abs() < 1e-6);
        assert!((var.membership_at("mid", 9.0)).abs() < 1e-6);
    }

    #[test]
    fn fuzzy_var_gaussmf_pico_correto() {
        let var = fuzzy_var!("x", 0.0, 100.0, 1001,
            "centro" => gaussmf { mean: 50.0, sigma: 10.0 },
        );
        assert!((var.membership_at("centro", 50.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fuzzy_var_sem_virgula_final() {
        let var = fuzzy_var!("x", 0.0, 10.0, 101,
            "a" => trimf [0.0, 5.0, 10.0]
        );
        assert_eq!(var.term_count(), 1);
    }

    #[test]
    fn fuzzy_var_mf_mistas() {
        let var = fuzzy_var!("x", 0.0, 100.0, 1001,
            "low"    => trimf   [0.0,   0.0,  40.0],
            "medium" => trapmf  [30.0, 40.0,  60.0, 70.0],
            "peak"   => gaussmf { mean: 50.0, sigma: 8.0 },
            "high"   => trimf   [60.0,100.0, 100.0],
        );
        assert_eq!(var.term_count(), 4);
    }

    // ── antecedent! e consequent! ──────────────────────────────────

    #[test]
    fn antecedent_registra_no_engine() {
        use crate::MamdaniEngine;
        let mut engine = MamdaniEngine::new();
        antecedent!(engine, "temperature", 0.0, 50.0, 501,
            "cold" => trimf [0.0,  0.0, 25.0],
            "hot"  => trimf [25.0,50.0, 50.0],
        );
        assert_eq!(engine.antecedent_count(), 1);
    }

    #[test]
    fn consequent_registra_no_engine() {
        use crate::MamdaniEngine;
        let mut engine = MamdaniEngine::new();
        consequent!(engine, "fan_speed", 0.0, 100.0, 1001,
            "slow" => trimf [0.0,  0.0, 50.0],
            "fast" => trimf [50.0,100.0,100.0],
        );
        assert_eq!(engine.consequent_count(), 1);
    }

    #[test]
    fn antecedent_multiplos_acumulam() {
        use crate::MamdaniEngine;
        let mut engine = MamdaniEngine::new();
        antecedent!(engine, "temperature", 0.0, 50.0,  501,
            "cold" => trimf [0.0, 0.0, 25.0],
            "hot"  => trimf [25.0,50.0,50.0],
        );
        antecedent!(engine, "humidity", 0.0, 100.0, 1001,
            "low"  => trimf [0.0, 0.0,  50.0],
            "high" => trimf [50.0,100.0,100.0],
        );
        assert_eq!(engine.antecedent_count(), 2);
    }

    #[test]
    fn antecedent_consequent_pipeline_completo() {
        use crate::MamdaniEngine;
        let mut engine = MamdaniEngine::new();

        antecedent!(engine, "x", 0.0, 10.0, 101,
            "low"  => trimf [0.0, 0.0, 5.0],
            "high" => trimf [5.0,10.0,10.0],
        );
        consequent!(engine, "y", 0.0, 10.0, 101,
            "small" => trimf [0.0, 0.0, 5.0],
            "large" => trimf [5.0,10.0,10.0],
        );

        engine.add_rule(rule!(IF x IS low  THEN y IS small));
        engine.add_rule(rule!(IF x IS high THEN y IS large));

        engine.set_input_unchecked("x", 2.0);
        let out = engine.compute().unwrap()["y"];
        assert!(out < 5.0);

        engine.set_input_unchecked("x", 8.0);
        let out = engine.compute().unwrap()["y"];
        assert!(out > 5.0);
    }
}
