//! # logicfuzzy-academic — Demo
//!
//! Two complete Mamdani Fuzzy Inference Systems that demonstrate
//! every major feature of the library:
//!
//!   1. **Tip Control**        — food quality + service → tip percentage
//!   2. **Irrigation Control** — soil moisture + temperature → valve opening
//!
//! Run: `cargo run`  →  SVGs written to `output/gorjeta/` and `output/irrigacao/`

use logicfuzzy_academic::{
    MamdaniEngine,
    antecedent, consequent, rule, export_svg, var_svg, fuzzy_var,
};

// ─── Simple table printer ─────────────────────────────────────────

struct Table {
    widths: Vec<usize>,
    aligns: Vec<bool>,    // true = right-align
    rows:   Vec<Vec<String>>,
}

impl Table {
    fn new(headers: &[(&str, usize, bool)]) -> Self {
        let widths: Vec<usize>  = headers.iter().map(|h| h.1).collect();
        let aligns: Vec<bool>   = headers.iter().map(|h| h.2).collect();
        let header_row: Vec<String> = headers.iter().map(|h| h.0.to_string()).collect();
        Self { widths: widths.clone(), aligns, rows: vec![header_row] }
    }

    fn push(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }

    fn print(&self) {
        let sep: String = self.widths.iter().map(|&w| "─".repeat(w + 2)).collect::<Vec<_>>().join("─");
        for (i, row) in self.rows.iter().enumerate() {
            if i == 1 { println!("  {}", sep); }
            let line: String = row.iter().zip(self.widths.iter()).zip(self.aligns.iter())
                .map(|((cell, &w), &right)| {
                    if right { format!("{:>width$}", cell, width = w) }
                    else      { format!("{:<width$}", cell, width = w) }
                })
                .collect::<Vec<_>>()
                .join("  ");
            println!("  {}", line);
        }
    }
}

// ─── Display helpers ──────────────────────────────────────────────

fn bar(d: f64) -> String {
    let n = (d * 12.0).round() as usize;
    format!("[{}{}]", "█".repeat(n.min(12)), "░".repeat(12 - n.min(12)))
}
fn divider()        { println!("{}", "═".repeat(70)); }
fn section(t: &str) { println!("\n  ── {} {}", t, "─".repeat(62_usize.saturating_sub(t.len() + 5))); }

fn print_pipeline(engine: &MamdaniEngine, output_var: &str) {
    let report = engine.explain().expect("explain failed");

    section("Fuzzification");
    for fv in &report.fuzzification {
        println!("\n  {} = {:.4}  (crisp input)", fv.variable, fv.crisp_input);
        for (term, deg) in &fv.term_degrees {
            println!("    {:>14}  mu = {:.6}  {}", term, deg, bar(*deg));
        }
        if let Some(dom) = fv.dominant_term() {
            println!("    -> dominant term: {}", dom);
        }
    }

    section("Rule firing");
    for rf in &report.rule_firings {
        println!("  {}  [{:.6}]  {}",
            if rf.fired { "✓" } else { "✗" },
            rf.firing_degree,
            rf.rule_text
        );
    }

    section(&format!("Defuzzified output — {}", output_var));
    if let Some(&v) = report.outputs.get(output_var) {
        println!("  centroid = {:.6}", v);
    }
}

// ─── System 1: Tip Control ───────────────────────────────────────
//
// Inputs:
//   food_quality  [0, 10]  — poor / good / excellent
//   service       [0, 10]  — poor / acceptable / great
// Output:
//   tip           [0, 25]% — low / medium / high

fn sistema_gorjeta() {
    divider();
    println!("  SYSTEM 1 — Tip Control");
    divider();

    let mut engine = MamdaniEngine::new();

    antecedent!(engine, "food_quality", 0.0, 10.0, 1001,
        "poor"      => trimf [0.0,  0.0,  5.0],
        "good"      => trimf [0.0,  5.0, 10.0],
        "excellent" => trimf [5.0, 10.0, 10.0],
    );

    antecedent!(engine, "service", 0.0, 10.0, 1001,
        "poor"       => trimf [0.0,  0.0,  5.0],
        "acceptable" => trimf [0.0,  5.0, 10.0],
        "great"      => trimf [5.0, 10.0, 10.0],
    );

    consequent!(engine, "tip", 0.0, 25.0, 1001,
        "low"    => trimf [ 0.0,  0.0, 10.0],
        "medium" => trimf [ 0.0, 12.0, 25.0],
        "high"   => trimf [12.0, 25.0, 25.0],
    );

    engine.add_rule(rule!(IF food_quality IS poor       OR  service IS poor       THEN tip IS low));
    engine.add_rule(rule!(IF food_quality IS good        OR  service IS poor       THEN tip IS low));
    engine.add_rule(rule!(IF food_quality IS poor        OR  service IS acceptable THEN tip IS medium));
    engine.add_rule(rule!(IF service IS acceptable                                 THEN tip IS medium));
    engine.add_rule(rule!(IF service IS great            OR  food_quality IS excellent THEN tip IS high));

    section("System summary");
    engine.print_summary();
    println!();
    engine.print_rules();

    section("Main scenario  →  quality=6.5  service=5.0");
    engine.set_input_unchecked("food_quality", 6.5);
    engine.set_input_unchecked("service",      5.0);
    print_pipeline(&engine, "tip");

    if let Some(cog) = engine.discrete_cog("tip", 5.0) {
        cog.print("tip — step 5.0");
    }

    section("Scenario table");
    let scenarios: &[(&str, f64, f64)] = &[
        ("Poor food, poor service",    1.0,  1.0),
        ("Good food, acceptable svc",  6.5,  5.0),
        ("Excellent food, great svc",  9.0,  9.0),
        ("Poor food, great service",   2.0,  8.0),
        ("Average all",                5.0,  5.0),
    ];

    let mut t = Table::new(&[
        ("Scenario",       32, false),
        ("Quality", 7, true),
        ("Service", 7, true),
        ("Tip%",    9, true),
        ("Class",   6, false),
    ]);
    for (desc, q, sv) in scenarios {
        engine.set_input_unchecked("food_quality", *q);
        engine.set_input_unchecked("service",      *sv);
        let tip = engine.compute().expect("compute failed")["$1"];
        let cls = if tip < 8.0 { "LOW" } else if tip < 16.0 { "MEDIUM" } else { "HIGH" };
        t.push(vec![
            desc.to_string(),
            format!("{:.1}", q),
            format!("{:.1}", sv),
            format!("{:.4}%", tip),
            cls.to_string(),
        ]);
    }
    t.print();

    section("SVG export — all methods");

    // Method 1: export_svg! — all variables at once + aggregated
    engine.set_input_unchecked("food_quality", 6.5);
    engine.set_input_unchecked("service",      5.0);
    engine.compute();
    println!("  [1] export_svg!(engine, dir, aggregated)");
    export_svg!(engine, "output/gorjeta", aggregated);

    // Method 2: var_svg! without input marker
    println!("\n  [2] var_svg!(var)  — clean MF plot");
    let tip_var = fuzzy_var!("tip", 0.0, 25.0, 1001,
        "low"    => trimf [ 0.0,  0.0, 10.0],
        "medium" => trimf [ 0.0, 12.0, 25.0],
        "high"   => trimf [12.0, 25.0, 25.0],
    );
    std::fs::write("output/gorjeta/tip_clean.svg", var_svg!(tip_var)).ok();
    println!("  ✓  output/gorjeta/tip_clean.svg");

    // Method 3: var_svg! with input — one SVG per scenario
    println!("\n  [3] var_svg!(var, value)  — per-scenario quality plots");
    let quality_var = fuzzy_var!("food_quality", 0.0, 10.0, 1001,
        "poor"      => trimf [0.0,  0.0,  5.0],
        "good"      => trimf [0.0,  5.0, 10.0],
        "excellent" => trimf [5.0, 10.0, 10.0],
    );
    for (desc, q, _) in scenarios {
        let key  = desc.replace(|c: char| !c.is_alphanumeric(), "_").to_lowercase();
        let path = format!("output/gorjeta/quality_{}.svg", key);
        std::fs::write(&path, var_svg!(quality_var, *q)).ok();
        println!("  ✓  {} (quality={:.1})", path, q);
    }
}

// ─── System 2: Irrigation Control ───────────────────────────────
//
// Inputs:
//   moisture     [0, 100]%  — low (trapmf) / medium (trimf) / high (trapmf)
//   temperature  [0,  40]°C — cold (trapmf) / warm (trimf) / hot (trapmf)
// Output:
//   valve        [0, 100]%  — low / medium / high
//
// Main scenario: moisture=38%  temperature=31°C
//   Expected fuzzification:
//     mu_low(38)  = 0.600000   mu_medium(38) = 0.400000   mu_high(38)  = 0.0
//     mu_cold(31) = 0.000000   mu_warm(31)   = 0.142857   mu_hot(31)   = 0.5

fn sistema_irrigacao() {
    divider();
    println!("  SYSTEM 2 — Irrigation Control");
    divider();

    let mut engine = MamdaniEngine::new();

    antecedent!(engine, "moisture", 0.0, 100.0, 1001,
        "low"    => trapmf [ 0.0,  0.0, 30.0,  50.0],
        "medium" => trimf  [30.0, 50.0, 70.0],
        "high"   => trapmf [60.0, 80.0,100.0, 100.0],
    );

    antecedent!(engine, "temperature", 0.0, 40.0, 1001,
        "cold" => trapmf [ 0.0,  0.0, 15.0, 22.0],
        "warm" => trimf  [18.0, 25.0, 32.0],
        "hot"  => trapmf [28.0, 34.0, 40.0, 40.0],
    );

    consequent!(engine, "valve", 0.0, 100.0, 1001,
        "low"    => trapmf [ 0.0,  0.0, 20.0, 40.0],
        "medium" => trimf  [25.0, 50.0, 75.0],
        "high"   => trapmf [60.0, 80.0,100.0,100.0],
    );

    // R1: low moisture  AND hot  → high  valve
    // R2: low moisture  AND warm → medium valve
    // R3: med moisture  AND hot  → medium valve
    // R4: med moisture  AND warm → low   valve
    // R5: high moisture OR  cold → low   valve
    engine.add_rule(rule!(IF moisture IS low    AND temperature IS hot  THEN valve IS high));
    engine.add_rule(rule!(IF moisture IS low    AND temperature IS warm THEN valve IS medium));
    engine.add_rule(rule!(IF moisture IS medium AND temperature IS hot  THEN valve IS medium));
    engine.add_rule(rule!(IF moisture IS medium AND temperature IS warm THEN valve IS low));
    engine.add_rule(rule!(IF moisture IS high    OR temperature IS cold THEN valve IS low));

    section("System summary");
    engine.print_summary();
    println!();
    engine.print_rules();

    section("Main scenario  →  moisture=38%  temperature=31°C");
    engine.set_input_unchecked("moisture",     38.0);
    engine.set_input_unchecked("temperature",  31.0);
    print_pipeline(&engine, "valve");

    section("Audit table");
    {
        let report = engine.explain().expect("explain failed");

        let mut tf = Table::new(&[
            ("Variable",   12, false),
            ("Term",       10, false),
            ("mu",         12, true),
            ("Bar",        14, false),
        ]);
        for fv in &report.fuzzification {
            for (term, deg) in &fv.term_degrees {
                tf.push(vec![
                    fv.variable.clone(),
                    term.clone(),
                    format!("{:.6}", deg),
                    bar(*deg),
                ]);
            }
        }
        tf.print();

        println!();
        let ops = ["min/AND", "min/AND", "min/AND", "min/AND", "max/OR"];
        let mut tr = Table::new(&[
            ("Rule",  5, false),
            ("Op",   10, false),
            ("alpha", 12, true),
            ("Fired",  5, false),
        ]);
        for (i, rf) in report.rule_firings.iter().enumerate() {
            tr.push(vec![
                format!("R{}", i + 1),
                ops[i].to_string(),
                format!("{:.6}", rf.firing_degree),
                if rf.fired { "yes" } else { "no" }.to_string(),
            ]);
        }
        tr.print();
    }

    section("COG discrete table  (step = 10)");
    if let Some(cog) = engine.discrete_cog("valve", 10.0) {
        cog.print("valve");
        println!();
        let numer_terms: Vec<String> = cog.disc_pts.iter()
            .zip(cog.mu_values.iter())
            .filter(|(_, &mu)| mu > 1e-9)
            .map(|(&x, &mu)| format!("{:.0}({:.4})", x, mu))
            .collect();
        println!("  I* = [ {} ]", numer_terms.join(" + "));
        println!("       / {:.6}", cog.denominator);
        println!("  I* = {:.6} / {:.6}", cog.numerator, cog.denominator);
        println!("  I* = {:.6}%", cog.centroid);
    }

    section("Scenario table");
    let scenarios: &[(&str, f64, f64)] = &[
        ("Dry soil, hot temperature",    20.0, 35.0),
        ("Moist soil, mild temperature", 75.0, 20.0),
        ("Intermediate case",            50.0, 30.0),
        ("Very moist, cold",             90.0, 10.0),
        ("Main scenario",                38.0, 31.0),
    ];

    let mut t = Table::new(&[
        ("Scenario",          28, false),
        ("Moisture%",          9, true),
        ("Temp(C)",            7, true),
        ("Valve%",            10, true),
        ("Class",              6, false),
    ]);
    for (desc, moist, temp) in scenarios {
        engine.set_input_unchecked("moisture",    *moist);
        engine.set_input_unchecked("temperature", *temp);
        let v   = engine.compute().expect("compute failed")["$1"];
        let cls = if v < 33.0 { "LOW" } else if v < 66.0 { "MEDIUM" } else { "HIGH" };
        t.push(vec![
            desc.to_string(),
            format!("{:.1}", moist),
            format!("{:.1}", temp),
            format!("{:.4}%", v),
            cls.to_string(),
        ]);
    }
    t.print();

    section("SVG export — all methods");

    // Method 1: export_svg! macro
    engine.set_input_unchecked("moisture",     38.0);
    engine.set_input_unchecked("temperature",  31.0);
    engine.compute();
    println!("  [1] export_svg!(engine, dir, aggregated)");
    export_svg!(engine, "output/irrigacao", aggregated);

    // Method 2: var_svg! — clean output variable
    println!("\n  [2] var_svg!(var)  — output variable, no input marker");
    let valve_var = fuzzy_var!("valve", 0.0, 100.0, 1001,
        "low"    => trapmf [ 0.0,  0.0, 20.0, 40.0],
        "medium" => trimf  [25.0, 50.0, 75.0],
        "high"   => trapmf [60.0, 80.0,100.0,100.0],
    );
    std::fs::write("output/irrigacao/valve_clean.svg", var_svg!(valve_var)).ok();
    println!("  ✓  output/irrigacao/valve_clean.svg");

    // Method 3: var_svg! with input — moisture per scenario
    println!("\n  [3] var_svg!(var, value)  — moisture per scenario");
    let moisture_var = fuzzy_var!("moisture", 0.0, 100.0, 1001,
        "low"    => trapmf [ 0.0,  0.0, 30.0,  50.0],
        "medium" => trimf  [30.0, 50.0, 70.0],
        "high"   => trapmf [60.0, 80.0,100.0, 100.0],
    );
    for (desc, moist, _) in scenarios {
        let key  = desc.replace(|c: char| !c.is_alphanumeric(), "_").to_lowercase();
        let path = format!("output/irrigacao/moisture_{}.svg", key);
        std::fs::write(&path, var_svg!(moisture_var, *moist)).ok();
        println!("  ✓  {} ({:.0}%)", path, moist);
    }

    // Method 4: var_svg! with input — temperature per scenario
    println!("\n  [4] var_svg!(var, value)  — temperature per scenario");
    let temp_var = fuzzy_var!("temperature", 0.0, 40.0, 1001,
        "cold" => trapmf [ 0.0,  0.0, 15.0, 22.0],
        "warm" => trimf  [18.0, 25.0, 32.0],
        "hot"  => trapmf [28.0, 34.0, 40.0, 40.0],
    );
    for (desc, _, temp) in scenarios {
        let key  = desc.replace(|c: char| !c.is_alphanumeric(), "_").to_lowercase();
        let path = format!("output/irrigacao/temperature_{}.svg", key);
        std::fs::write(&path, var_svg!(temp_var, *temp)).ok();
        println!("  ✓  {} ({:.0}C)", path, temp);
    }
}

// ─── Main ─────────────────────────────────────────────────────────

fn main() {
    std::fs::create_dir_all("output/gorjeta").ok();
    std::fs::create_dir_all("output/irrigacao").ok();

    println!();
    divider();
    println!("  logicfuzzy-academic v0.1.2");
    println!("  Pure-Rust Mamdani Fuzzy Inference System");
    println!("  Disciplina: Inteligência Artificial e Computacional — CESUPA");
    divider();

    sistema_gorjeta();
    sistema_irrigacao();

    println!();
    divider();
    println!("  SVG output summary:\n");
    for path in [
        "output/gorjeta/food_quality.svg     — MF plot + input marker",
        "output/gorjeta/service.svg           — MF plot + input marker",
        "output/gorjeta/tip.svg               — MF plot",
        "output/gorjeta/tip_aggregated.svg    — aggregated set + centroid",
        "output/gorjeta/tip_clean.svg         — MF plot (no marker)",
        "output/gorjeta/quality_*.svg         — per-scenario quality (5 files)",
        "",
        "output/irrigacao/moisture.svg        — MF plot + input marker",
        "output/irrigacao/temperature.svg     — MF plot + input marker",
        "output/irrigacao/valve.svg           — MF plot",
        "output/irrigacao/valve_aggregated.svg — aggregated set + centroid",
        "output/irrigacao/valve_clean.svg      — MF plot (no marker)",
        "output/irrigacao/moisture_*.svg       — per-scenario moisture (5 files)",
        "output/irrigacao/temperature_*.svg    — per-scenario temperature (5 files)",
    ] { println!("    {}", path); }
    divider();
    println!();
}
