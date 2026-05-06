# logicfuzzy-academic

[![Crates.io](https://img.shields.io/crates/v/logicfuzzy_academic.svg)](https://crates.io/crates/logicfuzzy_academic)
[![Docs.rs](https://docs.rs/logicfuzzy_academic/badge.svg)](https://docs.rs/logicfuzzy_academic)
[![CI](https://github.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic/actions/workflows/ci.yml/badge.svg)](https://github.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/Benjamin-Yuji-Suzuki/logicfuzzy-academic/graph/badge.svg?token=EW9I24MLI4)](https://codecov.io/github/Benjamin-Yuji-Suzuki/logicfuzzy-academic)
![Mutation Testing](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic/mutation-badge/mutation.json)

A pure-Rust Mamdani Fuzzy Inference System built from scratch — no external fuzzy crates.  
Developed as an academic project for the course **Artificial Intelligence and Computation** at CESUPA.

```rust
antecedent!(engine, "temperature", 0.0, 50.0, 501,
    "cold" => trimf [0.0,  0.0, 25.0],
    "warm" => trimf [0.0, 25.0, 50.0],
    "hot"  => trimf [25.0,50.0, 50.0],
);
engine.add_rule(rule!(IF temperature IS hot OR humidity IS high THEN fan_speed IS fast));
export_svg!(engine, "output/", aggregated);
```

---

## 📝 AI Audit

* **AI Assistance:** The logic, architecture, and development of this library were built with the assistance of **Claude Pro (Anthropic)** and **DeepSeek**. It was used as an advanced pair-programming tool to ensure high-quality, safe, and idiomatic Rust code.

---

## Features

- **Complete Mamdani pipeline** — fuzzification → inference → aggregation → defuzzification
- **Membership functions** — `trimf`, `trapmf`, `gaussmf`, including open shoulders
- **`rule!` macro** — declarative DSL: `IF x IS NOT cold AND y IS high THEN z IS fast` (up to 5 antecedents)
- **`fuzzy_var!` / `antecedent!` / `consequent!` macros** — build variables in one block
- **`var_svg!` / `export_svg!` macros** — SVG export in one call
- **`RuleBuilder`** — fluent API with `when_not()`, `and_not()`, `or_not()`, `also()`, `weight()`
- **`Expression` AST** — arbitrary nested `AND`/`OR` trees via `Rule::from_expression()` and `RuleBuilder::when_expr()`; `Expression::antecedents()` collects all leaf antecedents
- **AND / OR / NOT connectors** — min (t-norm), max (s-norm), complement
- **Rule weights** — `rule.with_weight(0.8)` scales firing degree
- **Multiple consequents** — `THEN fan IS fast AND light IS bright`
- **`DefuzzMethod`** — `Centroid`, `Bisector`, `MeanOfMaximum`, `SmallestOfMaximum`, `LargestOfMaximum`
- **`FuzzyError`** — `Result`-based errors: `MissingInput`, `InputOutOfRange`, `NoRulesFired`, `DuplicateVariable`, `InvalidInput` (NaN/infinite values rejected)
- **`try_add_antecedent()` / `try_add_consequent()`** — fallible registration returning `Result`
- **`antecedent_names()` / `consequent_names()`** — introspect registered variable names
- **`validate_rules()`** — checks all rule variables and terms exist, including inside `Expression`-based rules
- **`explain()`** — full pipeline report with fuzzification degrees and rule firing strengths
- **`discrete_cog()`** — step-by-step Centre-of-Gravity table
- **SVG visualization** — colour legend, μ annotations, clipped activation areas, aggregated output
- **`MamdaniEngine: Clone`** — clone the engine to run multiple scenarios without rebuilding
- **CI with coverage** — separate `doc-test` job, `coverage` job with `cargo-llvm-cov` and Codecov upload
- **Zero fuzzy dependencies** — only Rust `std`
- **274 unit tests + 6 doctests** — covering the full pipeline

---

## Quick start

```rust
use logicfuzzy_academic::{MamdaniEngine, antecedent, consequent, rule, export_svg};

fn main() {
    let mut engine = MamdaniEngine::new();

    antecedent!(engine, "temperature", 0.0, 50.0, 501,
        "cold" => trimf [0.0,  0.0, 25.0],
        "warm" => trimf [0.0, 25.0, 50.0],
        "hot"  => trimf [25.0,50.0, 50.0],
    );
    antecedent!(engine, "humidity", 0.0, 100.0, 1001,
        "low"    => trimf [0.0,  0.0,  50.0],
        "medium" => trimf [0.0, 50.0, 100.0],
        "high"   => trimf [50.0,100.0,100.0],
    );
    consequent!(engine, "fan_speed", 0.0, 100.0, 1001,
        "slow"   => trimf [0.0,  0.0,  50.0],
        "medium" => trimf [0.0, 50.0, 100.0],
        "fast"   => trimf [50.0,100.0,100.0],
    );

    engine.add_rule(rule!(IF temperature IS cold AND humidity IS low    THEN fan_speed IS slow));
    engine.add_rule(rule!(IF temperature IS warm AND humidity IS medium THEN fan_speed IS medium));
    engine.add_rule(rule!(IF temperature IS hot  OR  humidity IS high   THEN fan_speed IS fast));
    engine.add_rule(rule!(IF temperature IS cold AND humidity IS high   THEN fan_speed IS medium));

    engine.set_input("temperature", 45.0).unwrap();
    engine.set_input("humidity",    90.0).unwrap();

    let result = engine.compute().unwrap();
    println!("fan_speed = {:.2}%", result["fan_speed"]); // ~67%

    export_svg!(engine, "output/", aggregated);
}
```

---

## Macros

### `fuzzy_var!` — create a variable

```rust
let temp = fuzzy_var!("temperature", 0.0, 50.0, 501,
    "cold"   => trimf   [0.0,  0.0, 25.0],
    "stable" => trapmf  [10.0,20.0, 30.0, 40.0],
    "peak"   => gaussmf { mean: 25.0, sigma: 5.0 },
);
```

### `antecedent!` / `consequent!` — register in the engine

```rust
antecedent!(engine, "humidity", 0.0, 100.0, 1001,
    "low"  => trapmf [0.0,  0.0, 30.0, 50.0],
    "mid"  => trimf  [30.0,50.0, 70.0],
    "high" => trapmf [60.0,80.0,100.0,100.0],
);
```

### `rule!` — natural language rules with NOT

```rust
rule!(IF temperature IS hot AND humidity IS high THEN fan_speed IS fast)
rule!(IF temperature IS NOT cold OR humidity IS high THEN fan_speed IS fast)

// Up to 5 antecedents
rule!(IF a IS x AND b IS y AND c IS z AND d IS w AND e IS v THEN out IS result)

// Multiple consequents
rule!(IF temperature IS hot THEN fan_speed IS fast AND light IS bright)

// Rule weight
rule!(IF smoke IS high THEN alert IS critical).with_weight(0.9)
```

Or with `RuleBuilder`:

```rust
RuleBuilder::new()
    .when_not("temperature", "cold")
    .and("humidity", "high")
    .then("fan_speed", "fast")
    .also("light", "bright")
    .weight(0.9)
    .build()
```

### `Expression` AST — arbitrary nested logic

For rules more complex than the `rule!` macro supports, build an expression tree directly:

```rust
use logicfuzzy_academic::{Expression, Antecedent, Rule};

let expr = Expression::and(vec![
    Expression::term(Antecedent::new("temperature", "hot")),
    Expression::or(vec![
        Expression::term(Antecedent::new("humidity", "high")),
        Expression::term(Antecedent::negated("pressure", "low")),
    ]),
]);

let rule = Rule::from_expression(expr, vec![("fan_speed".into(), "fast".into())]);
engine.add_rule(rule);

// Or via RuleBuilder
RuleBuilder::new()
    .when_expr(expr)
    .then("fan_speed", "fast")
    .build()
```

### `var_svg!` / `export_svg!`

```rust
// Single variable SVG
let svg = var_svg!(var);           // clean MF plot
let svg = var_svg!(var, 35.0);    // with input marker at x=35

// All engine variables at once
export_svg!(engine, "output/");               // MF SVGs only
export_svg!(engine, "output/", aggregated);   // + aggregated output SVGs
```

---

## Defuzzification methods

```rust
use logicfuzzy_academic::DefuzzMethod;

engine.set_defuzz_method(DefuzzMethod::Bisector);
engine.set_defuzz_method(DefuzzMethod::MeanOfMaximum);
engine.set_defuzz_method(DefuzzMethod::SmallestOfMaximum);
engine.set_defuzz_method(DefuzzMethod::LargestOfMaximum);
// Default: DefuzzMethod::Centroid
```

---

## Error handling

```rust
use logicfuzzy_academic::FuzzyError;

// set_input validates the universe range
match engine.set_input("temperature", 999.0) {
    Ok(()) => {}
    Err(FuzzyError::InputOutOfRange { variable, value, min, max }) => {
        eprintln!("{variable} = {value} clamped to [{min}, {max}]");
    }
    Err(FuzzyError::MissingInput(name)) => {
        eprintln!("variable '{name}' not registered");
    }
    Err(_) => {}
}

// compute() returns Err(NoRulesFired { .. }) when all firing degrees are zero
match engine.compute() {
    Ok(outputs) => println!("fan_speed = {:.2}", outputs["fan_speed"]),
    Err(FuzzyError::NoRulesFired { .. }) => eprintln!("no rule fired — check inputs"),
    Err(e) => eprintln!("error: {e}"),
}
```

### Fallible registration and rule validation

```rust
// try_add_antecedent / try_add_consequent return Err(DuplicateVariable) instead of panicking
engine.try_add_antecedent(var)?;
engine.try_add_consequent(out_var)?;

// validate_rules checks all variable and term names before running inference
if let Err(errors) = engine.validate_rules() {
    for e in errors {
        eprintln!("rule error: {e}");
    }
}
```

### Cloning the engine

```rust
// MamdaniEngine derives Clone — useful for running multiple scenarios
let base = engine.clone();

for value in [10.0, 30.0, 50.0] {
    let mut e = base.clone();
    e.set_input("temperature", value).unwrap();
    let out = e.compute().unwrap();
    println!("temp={value} → fan={:.2}", out["fan_speed"]);
}
```

---

## `explain()` — inspecting the pipeline

```rust
engine.set_input("temperature", 5.0).unwrap();
engine.set_input("humidity",    10.0).unwrap();

let report = engine.explain().unwrap();
println!("{}", report.summary());
```

```
=== Fuzzy Mamdani — Explain Report ===

[ Fuzzification ]
  humidity = 10.0000 (crisp)
          low  1.0000  [████████████]
       medium  0.2000  [██░░░░░░░░░░]
         high  0.0000  [░░░░░░░░░░░░]
    -> dominant term: low

[ Rule Evaluation ] (2 fired, 2 skipped)
  ✓ [0.8000]  IF (temperature IS cold) AND (humidity IS low) THEN fan_speed IS slow
  ✗ [0.0000]  IF (temperature IS warm) AND (humidity IS medium) THEN fan_speed IS medium

[ Defuzzification Output ]
  fan_speed = 18.4956
```

---

## `discrete_cog()` — step-by-step centroid table

```rust
engine.set_input("moisture",     38.0).unwrap();
engine.set_input("temperature",  31.0).unwrap();
engine.compute().unwrap();

let table = engine.discrete_cog("valve", 10.0).unwrap();
table.print("valve");
// Numerator   = 240.285714
// Denominator = 3.828571
// Centroid    = 62.761194
```

---

## SVG visualization

Every SVG is self-contained and opens in any browser.  
Features: colour legend, μ-value annotations, clipped activation areas, centroid marker.

```rust
// Per-variable
std::fs::write("temp.svg", var.to_svg()).unwrap();
std::fs::write("temp.svg", var.to_svg_with_input(35.0)).unwrap();

// All engine variables
engine.export_svg("output/").unwrap();
engine.export_aggregated_svg("output/").unwrap();
```

---

## Project structure

```
src/
├── error.rs      — FuzzyError (MissingInput, InputOutOfRange, NoRulesFired, DuplicateVariable)
├── membership.rs — trimf, trapmf, gaussmf, MembershipFn
├── variable.rs   — Universe, Term, FuzzyVariable, Universe::with_resolution()
├── rule.rs       — Antecedent (with NOT), Expression, Rule, RuleBuilder, Connector
├── engine.rs     — MamdaniEngine (Clone) — full pipeline, DefuzzMethod, discrete_cog(),
│                   try_add_antecedent(), try_add_consequent(), validate_rules()
├── explain.rs    — ExplainReport, RuleFiring, FuzzifiedVariable, CogTable
├── svg.rs        — pure-Rust SVG renderer (zero dependencies)
├── macros.rs     — rule!, fuzzy_var!, antecedent!, consequent!, var_svg!, export_svg!
└── lib.rs        — public re-exports

examples/
└── demo.rs       — two complete systems (tip control + irrigation) with SVG export
```

### Pipeline

```
crisp inputs  →  fuzzification  →  inference (AND=min, OR=max, NOT=1-μ)
             →  clip (Mamdani implication)  →  aggregation (max)
             →  defuzzification (Centroid / Bisector / MOM / SOM / LOM)
             →  crisp output
```

---

## Running

```bash
git clone https://github.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic
cd logicfuzzy-academic
cargo run --example demo   # two systems + SVG export to output/
cargo test                 # 274 tests (unit + doctests)
```

---

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for the full history.

---

## Acknowledgements

Designed as a functional equivalent of **[scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy)** for Python.

Theoretical foundation:
- **Lotfi A. Zadeh** — *Fuzzy Sets* (1965)
- **E. H. Mamdani & S. Assilian** — *An experiment in linguistic synthesis with a fuzzy logic controller* (1975)

---

## License

MIT © 2026 [Benjamin Yuji Suzuki](https://github.com/Benjamin-Yuji-Suzuki)

See [LICENSE](./LICENSE) for the full text.