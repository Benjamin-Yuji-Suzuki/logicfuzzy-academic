//! # TSK Demo — Room Climate Control
//!
//! Demonstrates the Takagi-Sugeno-Kang inference engine with a
//! dual-output climate control system.
//!
//!   Inputs:   temperature [0, 40]°C   — cold / warm / hot
//!             humidity    [0, 100]%   — low / medium / high
//!
//!   Outputs:  fan         [0, 100]%   — linear function of temp + humidity
//!             heater      [0, 100]%   — linear function of temp
//!
//! Run:  cargo run --example tsk_demo

use logicfuzzy_academic::{
    rule::Antecedent, rule::Connector, tsk_output, FuzzyVariable, MembershipFn, Term,
    TskConsequent, TskEngine, TskRule, Universe,
};

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
    println!("  TSK Demo — Room Climate Control");
    divider();

    let mut engine = TskEngine::new();

    // ── Antecedents ──────────────────────────────────────────────

    let mut temp = FuzzyVariable::new("temperature", Universe::new(0.0, 40.0, 501));
    temp.add_term(Term::new(
        "cold",
        MembershipFn::Trapmf([0.0, 0.0, 15.0, 22.0]),
    ));
    temp.add_term(Term::new("warm", MembershipFn::Trimf([18.0, 25.0, 32.0])));
    temp.add_term(Term::new(
        "hot",
        MembershipFn::Trapmf([28.0, 34.0, 40.0, 40.0]),
    ));
    engine.add_antecedent(temp);

    let mut hum = FuzzyVariable::new("humidity", Universe::new(0.0, 100.0, 1001));
    hum.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 40.0])));
    hum.add_term(Term::new("medium", MembershipFn::Trimf([30.0, 50.0, 70.0])));
    hum.add_term(Term::new("high", MembershipFn::Trimf([60.0, 100.0, 100.0])));
    engine.add_antecedent(hum);

    // ── Outputs ──────────────────────────────────────────────────

    tsk_output!(engine, "fan", 0.0, 100.0, 101);
    tsk_output!(engine, "heater", 0.0, 100.0, 101);

    // ── TSK Rules ────────────────────────────────────────────────
    //
    // Coefficients format: [bias, coeff_temp, coeff_humidity]
    // (antecedent variables are sorted alphabetically: humidity, temperature)
    // So position 0 = bias, 1 = humidity, 2 = temperature

    // Rule 1: IF temp IS cold THEN heater = 80 - 0.5·temp + 0·hum, fan = 10
    engine.add_rule(TskRule::new(
        vec![Antecedent::new("temperature", "cold")],
        Connector::And,
        vec![
            TskConsequent::new("heater", vec![80.0, 0.0, -0.5]),
            TskConsequent::new("fan", vec![10.0, 0.0, 0.0]),
        ],
    ));

    // Rule 2: IF temp IS warm AND humidity IS low THEN fan = 10 + 0.2·hum + 1.5·temp, heater = 30
    engine.add_rule(TskRule::new(
        vec![
            Antecedent::new("temperature", "warm"),
            Antecedent::new("humidity", "low"),
        ],
        Connector::And,
        vec![
            TskConsequent::new("fan", vec![10.0, 0.2, 1.5]),
            TskConsequent::new("heater", vec![30.0, 0.0, 0.0]),
        ],
    ));

    // Rule 3: IF temp IS warm AND humidity IS high THEN fan = 20 + 2.0·hum, heater = 20
    engine.add_rule(TskRule::new(
        vec![
            Antecedent::new("temperature", "warm"),
            Antecedent::new("humidity", "high"),
        ],
        Connector::And,
        vec![
            TskConsequent::new("fan", vec![20.0, 2.0, 0.0]),
            TskConsequent::new("heater", vec![20.0, 0.0, 0.0]),
        ],
    ));

    // Rule 4: IF temp IS hot AND humidity IS low THEN fan = 40 + 4·hum + 2·temp, heater = 0
    engine.add_rule(TskRule::new(
        vec![
            Antecedent::new("temperature", "hot"),
            Antecedent::new("humidity", "low"),
        ],
        Connector::And,
        vec![
            TskConsequent::new("fan", vec![40.0, 4.0, 2.0]),
            TskConsequent::new("heater", vec![0.0, 0.0, 0.0]),
        ],
    ));

    // Rule 5: IF temp IS hot AND humidity IS high THEN fan = 60 + 6·hum, heater = 0
    engine.add_rule(TskRule::new(
        vec![
            Antecedent::new("temperature", "hot"),
            Antecedent::new("humidity", "high"),
        ],
        Connector::And,
        vec![
            TskConsequent::new("fan", vec![60.0, 6.0, 0.0]),
            TskConsequent::new("heater", vec![0.0, 0.0, 0.0]),
        ],
    ));

    // Rule 6: IF temp IS warm AND humidity IS medium THEN fan = 20 + 0.5·hum + 1.0·temp, heater = 25
    engine.add_rule(TskRule::new(
        vec![
            Antecedent::new("temperature", "warm"),
            Antecedent::new("humidity", "medium"),
        ],
        Connector::And,
        vec![
            TskConsequent::new("fan", vec![20.0, 0.5, 1.0]),
            TskConsequent::new("heater", vec![25.0, 0.0, 0.0]),
        ],
    ));

    // Rule 7: IF humidity IS low AND temp IS NOT hot THEN heater = 50
    engine.add_rule(TskRule::new(
        vec![
            Antecedent::new("humidity", "low"),
            Antecedent::negated("temperature", "hot"),
        ],
        Connector::And,
        vec![TskConsequent::new("heater", vec![50.0, 0.0, 0.0])],
    ));

    // ── System Summary ───────────────────────────────────────────

    section("System summary");
    engine.print_summary();

    println!("\n  Rules:");
    // We can't iterate rules easily since TskRule Display shows them
    section("Scenario table");

    let scenarios: &[(&str, f64, f64)] = &[
        ("Cold, dry", 5.0, 20.0),
        ("Cool, humid", 12.0, 80.0),
        ("Mild, average", 22.0, 50.0),
        ("Warm, dry", 28.0, 25.0),
        ("Hot, humid", 35.0, 85.0),
        ("Hot, dry", 38.0, 15.0),
    ];

    println!(
        "  {:<20}  {:>8}  {:>8}  {:>8}  {:>10}",
        "Scenario", "Temp(°C)", "Hum(%)", "Fan(%)", "Heater(%)"
    );
    println!("  {}", "─".repeat(65));

    for (desc, t, h) in scenarios {
        engine.reset_inputs();
        engine.set_input_unchecked("temperature", *t);
        engine.set_input_unchecked("humidity", *h);
        let result = engine.compute().expect("compute failed");
        let fan_val = result["fan"];
        let heater_val = result["heater"];
        println!(
            "  {:<20}  {:>8.1}  {:>8.1}  {:>8.2}  {:>10.2}",
            desc, t, h, fan_val, heater_val
        );
    }

    // ── SVG Export ───────────────────────────────────────────────

    section("SVG export");
    engine.set_input_unchecked("temperature", 35.0);
    engine.set_input_unchecked("humidity", 85.0);
    let _ = engine.compute();
    match engine.export_svg("output/tsk_climate") {
        Ok(_) => println!("  ✓  SVGs → output/tsk_climate/"),
        Err(e) => println!("  ✗  Error: {}", e),
    }

    divider();
    println!();
}
