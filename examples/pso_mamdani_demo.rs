//! # PSO + Mamdani Demo — Optimizing Membership Function Parameters
//!
//! Uses PSO to tune the peak position (b) of each membership function
//! in a Mamdani fuzzy system, minimizing error against a target curve.
//!
//! The target is:  yₜ(x) = 10 · sin(x · 0.3) + 50   (x ∈ [0, 10])
//!
//! A Mamdani system with 3 triangular MFs learns to approximate it
//! by moving the peak of each MF via PSO.
//!
//! Run:  cargo run --example pso_mamdani_demo

use logicfuzzy_academic::{
    DefuzzMethod, FuzzyVariable, MamdaniEngine, MembershipFn, PsoConfig, PsoOptimizer, RuleBuilder,
    Term, Universe,
};

/// Target function to approximate
fn target(x: f64) -> f64 {
    (x * 0.3).sin() * 10.0 + 50.0
}

/// Build a Mamdani engine where the three MF peaks are tunable parameters.
///
/// Parameters: [peak_low, peak_medium, peak_high] in [0, 10]
/// Each MF is a symmetric triangle: the peak is at param, with base extending ±3.
fn build_mamdani(params: &[f64]) -> MamdaniEngine {
    let mut engine = MamdaniEngine::new();

    let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 501));
    let p_low = params[0].clamp(0.0, 10.0);
    let p_med = params[1].clamp(0.0, 10.0);
    let p_high = params[2].clamp(0.0, 10.0);

    x.add_term(Term::new(
        "low",
        MembershipFn::Trimf([(p_low - 4.0).max(0.0), p_low, (p_low + 4.0).min(10.0)]),
    ));
    x.add_term(Term::new(
        "medium",
        MembershipFn::Trimf([(p_med - 4.0).max(0.0), p_med, (p_med + 4.0).min(10.0)]),
    ));
    x.add_term(Term::new(
        "high",
        MembershipFn::Trimf([(p_high - 4.0).max(0.0), p_high, (p_high + 4.0).min(10.0)]),
    ));
    engine.add_antecedent(x);

    let mut y = FuzzyVariable::new("y", Universe::new(0.0, 100.0, 1001));
    y.add_term(Term::new("low", MembershipFn::Trimf([30.0, 40.0, 50.0])));
    y.add_term(Term::new("medium", MembershipFn::Trimf([45.0, 55.0, 65.0])));
    y.add_term(Term::new("high", MembershipFn::Trimf([60.0, 70.0, 80.0])));
    engine.add_consequent(y);

    engine.add_rule(RuleBuilder::new().when("x", "low").then("y", "low").build());
    engine.add_rule(
        RuleBuilder::new()
            .when("x", "medium")
            .then("y", "medium")
            .build(),
    );
    engine.add_rule(
        RuleBuilder::new()
            .when("x", "high")
            .then("y", "high")
            .build(),
    );

    engine
}

/// Evaluate MSE between Mamdani output and target at 20 test points.
fn fitness(params: &[f64]) -> f64 {
    let mut engine = build_mamdani(params);
    engine.set_defuzz_method(DefuzzMethod::Centroid);
    let mse: f64 = (0..=20)
        .map(|i| {
            let x = i as f64 * 0.5;
            engine.reset_inputs();
            engine.set_input_unchecked("x", x);
            let y_fuzzy = engine.compute().map(|r| r["y"]).unwrap_or(50.0);
            let y_target = target(x);
            (y_fuzzy - y_target).powi(2)
        })
        .sum();
    mse / 21.0
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
    println!("  PSO + Mamdani Demo — Optimizing MF Peaks");
    println!("  Target:  y = 10·sin(0.3·x) + 50");
    println!("  3 parameters: peak positions of low/medium/high MFs");
    divider();

    let config = PsoConfig {
        population_size: 40,
        max_iterations: 200,
        inertia_weight: 0.729,
        cognitive_coefficient: 1.494,
        social_coefficient: 1.494,
        bounds: vec![(0.0, 10.0); 3],
        velocity_limit: Some(2.0),
        tolerance: 1e-6,
        patience: 30,
        seed: Some(42),
    };

    section("Optimizing");
    let mut optimizer = PsoOptimizer::new(config);
    let (best_params, best_fit, state) = optimizer.optimize(fitness);

    println!("  Best MSE:         {:.6}", best_fit);
    println!("  RMSE:             {:.4}", best_fit.sqrt());
    println!("  Converged:        {}", state.converged);
    println!("  Iterations:       {}", state.iteration + 1);

    section("Optimized MF peaks");
    println!("  Low MF peak:      {:.4}", best_params[0]);
    println!("  Medium MF peak:   {:.4}", best_params[1]);
    println!("  High MF peak:     {:.4}", best_params[2]);

    let final_engine = build_mamdani(&best_params);
    section("Approximation table");
    println!(
        "  {:<6}  {:>10}  {:>10}  {:>10}",
        "x", "Target", "Mamdani", "Error"
    );
    println!("  {}", "─".repeat(45));
    for i in 0..=10 {
        let x = i as f64;
        let y_target = target(x);
        let mut eng = final_engine.clone();
        eng.reset_inputs();
        eng.set_input_unchecked("x", x);
        let y_fuzzy = eng.compute().map(|r| r["y"]).unwrap_or(50.0);
        println!(
            "  {:<6.1}  {:>10.4}  {:>10.4}  {:>10.4}",
            x,
            y_target,
            y_fuzzy,
            (y_fuzzy - y_target).abs()
        );
    }

    divider();
    println!();
}
