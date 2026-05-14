//! # PSO Demo — Optimizing TSK Coefficients
//!
//! Uses Particle Swarm Optimization to find the optimal polynomial
//! coefficients for a TSK fuzzy system that models a target function.
//!
//! The target function is:  y = sin(x) · cos(x)   (x ∈ [0, π])
//!
//! A TSK system with triangular antecedents learns to approximate it
//! by optimizing its consequent coefficients via PSO.
//!
//! Run:  cargo run --example pso_demo

use logicfuzzy_academic::{
    rule::Antecedent, rule::Connector, tsk_output, FuzzyVariable, MembershipFn, PsoConfig,
    PsoOptimizer, Term, TskConsequent, TskEngine, TskRule, Universe,
};

/// Target function to approximate
fn target(x: f64) -> f64 {
    (x * 0.5).sin() * (x * 0.3).cos() * 50.0 + 50.0
}

/// Build a TSK engine with tunable coefficients for rule consequents.
/// The `params` slice contains concatenated coefficients for all rules:
///
///   Rule format: IF x IS term_k THEN y = c0 + c1·x
///   Each rule contributes 2 params: [c0, c1]
fn build_tsk(params: &[f64]) -> TskEngine {
    let mut engine = TskEngine::new();

    let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 501));
    x.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 5.0])));
    x.add_term(Term::new("medium", MembershipFn::Trimf([0.0, 5.0, 10.0])));
    x.add_term(Term::new("high", MembershipFn::Trimf([5.0, 10.0, 10.0])));
    engine.add_antecedent(x);
    tsk_output!(engine, "y", 0.0, 100.0, 101);

    // Rule 1: IF x IS low THEN y = c0 + c1·x
    engine.add_rule(TskRule::new(
        vec![Antecedent::new("x", "low")],
        Connector::And,
        vec![TskConsequent::new("y", vec![params[0], params[1]])],
    ));

    // Rule 2: IF x IS medium THEN y = c2 + c3·x
    engine.add_rule(TskRule::new(
        vec![Antecedent::new("x", "medium")],
        Connector::And,
        vec![TskConsequent::new("y", vec![params[2], params[3]])],
    ));

    // Rule 3: IF x IS high THEN y = c4 + c5·x
    engine.add_rule(TskRule::new(
        vec![Antecedent::new("x", "high")],
        Connector::And,
        vec![TskConsequent::new("y", vec![params[4], params[5]])],
    ));

    engine
}

/// Evaluate the TSK system at a set of test points and return the
/// mean-squared error against the target function.
fn fitness(params: &[f64]) -> f64 {
    let mut engine = build_tsk(params);
    let test_points: Vec<f64> = (0..=50).map(|i| i as f64 * 0.2).collect();
    let mse: f64 = test_points
        .iter()
        .map(|&x| {
            engine.reset_inputs();
            engine.set_input_unchecked("x", x);
            let y_tsk = engine.compute().map(|r| r["y"]).unwrap_or(50.0);
            let y_target = target(x);
            (y_tsk - y_target).powi(2)
        })
        .sum();
    mse / test_points.len() as f64
}

fn divider() {
    println!("{}", "═".repeat(65));
}
fn section(t: &str) {
    println!(
        "\n  ── {} {}",
        t,
        "─".repeat(55_usize.saturating_sub(t.len() + 5))
    );
}

fn main() {
    println!();
    divider();
    println!("  PSO Demo — Optimizing TSK Coefficients");
    println!("  Target:  y = sin(0.5·x) · cos(0.3·x) · 50 + 50");
    println!("  3 TSK rules × 2 coefficients each = 6 parameters");
    divider();

    // ── PSO Configuration ────────────────────────────────────────
    //
    // Coefficients: [c0, c1] for low, [c2, c3] for medium, [c4, c5] for high
    // Bounds based on output range [0, 100]

    let config = PsoConfig {
        population_size: 60,
        max_iterations: 300,
        inertia_weight: 0.729,
        cognitive_coefficient: 1.494,
        social_coefficient: 1.494,
        bounds: vec![(-50.0, 100.0); 6],
        velocity_limit: Some(20.0),
        tolerance: 1e-7,
        patience: 40,
        seed: Some(42),
    };

    // ── Run optimization ─────────────────────────────────────────

    section("Optimizing");
    println!("  Population: {}", config.population_size);
    println!("  Max iterations: {}", config.max_iterations);
    println!("  Params: 6 (bias + coeff per rule)");

    let mut optimizer = PsoOptimizer::new(config);
    let (best_params, best_fit, state) = optimizer.optimize(fitness);

    println!();
    println!("  Best fitness (MSE): {:.8}", best_fit);
    println!(
        "  Converged: {}  Iterations: {}",
        state.converged,
        state.iteration + 1
    );

    // ── Results ──────────────────────────────────────────────────

    section("Optimized coefficients");
    let labels = [
        "low_bias",
        "low_coef",
        "med_bias",
        "med_coef",
        "high_bias",
        "high_coef",
    ];
    for (i, (label, val)) in labels.iter().zip(best_params.iter()).enumerate() {
        println!("  Rule {}: {:>10} = {:.6}", i / 2 + 1, label, val);
    }

    section("Approximation table");
    let final_engine = build_tsk(&best_params);
    println!(
        "  {:<8}  {:>10}  {:>10}  {:>10}",
        "x", "Target", "TSK", "Error"
    );
    println!("  {}", "─".repeat(45));
    for i in 0..=10 {
        let x = i as f64;
        let y_target = target(x);
        let mut eng = final_engine.clone();
        eng.set_input_unchecked("x", x);
        let y_tsk = eng.compute().map(|r| r["y"]).unwrap_or(50.0);
        println!(
            "  {:<8.1}  {:>10.4}  {:>10.4}  {:>10.4}",
            x,
            y_target,
            y_tsk,
            (y_tsk - y_target).abs()
        );
    }

    // ── Optimization progress ─────────────────────────────

    section("Optimization progress");
    println!("  Converged in {} iterations", state.iteration + 1);
    println!("  Final MSE: {:.6}", best_fit);
    println!("  Avg absolute error per point: {:.4}", best_fit.sqrt());

    divider();
    println!();
}
