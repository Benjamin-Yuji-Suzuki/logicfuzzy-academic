# logicfuzzy-academic
---
[![Crates.io](https://img.shields.io/crates/v/logicfuzzy_academic.svg)](https://crates.io/crates/logicfuzzy_academic)
[![Docs.rs](https://docs.rs/logicfuzzy_academic/badge.svg)](https://docs.rs/logicfuzzy_academic)
[![CI](https://github.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic/actions/workflows/ci.yml/badge.svg)](https://github.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/Benjamin-Yuji-Suzuki/logicfuzzy-academic/graph/badge.svg?token=EW9I24MLI4)](https://codecov.io/github/Benjamin-Yuji-Suzuki/logicfuzzy-academic)
![Mutation Testing](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic/mutation-badge/mutation.json)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Benjamin-Yuji-Suzuki_logicfuzzy-academic&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Benjamin-Yuji-Suzuki_logicfuzzy-academic)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Benjamin-Yuji-Suzuki_logicfuzzy-academic&metric=coverage)](https://sonarcloud.io/summary/new_code?id=Benjamin-Yuji-Suzuki_logicfuzzy-academic)

---
A pure-Rust Fuzzy Inference System built from scratch — zero external dependencies.  
Includes both **Mamdani** and **Takagi-Sugeno-Kang (TSK)** inference engines, plus a **Particle Swarm Optimizer** for parameter tuning.  
Developed as an academic project for the course **Artificial Intelligence and Computation** at CESUPA.

```rust
// Mamdani
antecedent!(engine, "temperature", 0.0, 50.0, 501,
    "cold" => trimf [0.0,  0.0, 25.0],
    "hot"  => trimf [25.0,50.0, 50.0],
);
engine.add_rule(rule!(IF temperature IS hot OR humidity IS high THEN fan_speed IS fast));

// TSK
let mut tsk = TskEngine::new();
tsk.add_antecedent(/* fuzzy variable */);
tsk.add_rule(TskRule::new(/* antecedents */, Connector::And, /* consequents */));
let result = tsk.compute().unwrap();
```

---

## 📝 AI Audit

* **AI Assistance:** The logic, architecture, and development of this library were built with the assistance of **Claude Pro (Anthropic)** and **DeepSeek**. Used as advanced pair-programming tools to ensure high-quality, safe, and idiomatic Rust code.

For the full declaration of AI usage, see the [AI Usage Declaration](#-ai-usage-declaration) section at the bottom of this document.

---

## Features

- **Complete Mamdani pipeline** — fuzzification → inference → aggregation → defuzzification
- **TSK (Takagi-Sugeno-Kang) engine** — crisp polynomial consequents, weighted-average output
- **Particle Swarm Optimizer** — tune MF parameters, rule weights, or TSK coefficients
- **Membership functions** — `trimf`, `trapmf`, `gaussmf`, including open shoulders
- **`rule!` macro** — declarative DSL: `IF x IS NOT cold AND y IS high THEN z IS fast` (up to 5 antecedents)
- **`fuzzy_var!` / `antecedent!` / `consequent!` macros** — build variables in one block
- **`tsk_rule!` / `tsk_output!` macros** — TSK rule DSL and output registration
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
- **CI with coverage** — separate `doc-test` job, `coverage` job with `cargo-llvm-cov`, Codecov upload, and SonarCloud continuous analysis
- **Mutation testing** — `cargo-mutants` with dedicated CI workflow and dynamic badge (see badge above)
- **SonarCloud integration** — continuous static analysis covering code smells, duplications, complexity, and vulnerabilities
- **Comprehensive test suite** — 460+ unit, 14 integration/E2E/concurrency, and 45+ doc-tests
- **Zero external dependencies** — only Rust `std` (built-in SplitMix64 PRNG for PSO)

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

## TSK (Takagi-Sugeno-Kang) Quick Start

TSK uses fuzzy antecedents but crisp polynomial consequents. The output is a weighted average of each rule's polynomial evaluated at the crisp inputs.

```rust
use logicfuzzy_academic::{
    TskEngine, TskRule, TskConsequent,
    FuzzyVariable, Universe, Term, MembershipFn,
    rule::Antecedent, rule::Connector,
};

let mut engine = TskEngine::new();

// Input variables (same as Mamdani)
let mut x = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
x.add_term(Term::new("small", MembershipFn::Trimf([0.0, 0.0, 5.0])));
x.add_term(Term::new("large", MembershipFn::Trimf([5.0, 10.0, 10.0])));
engine.add_antecedent(x);

// Output universe (no fuzzy terms needed — consequents are polynomials)
engine.add_output("y", Universe::new(0.0, 100.0, 101));

// Zero-order TSK: constant consequents
engine.add_rule(TskRule::new(
    vec![Antecedent::new("x", "small")],
    Connector::And,
    vec![TskConsequent::new("y", vec![25.0, 0.0])], // y = 25 + 0*x
));

// First-order TSK: linear function of inputs (coefficients: [bias, c_x])
engine.add_rule(TskRule::new(
    vec![Antecedent::new("x", "large")],
    Connector::And,
    vec![TskConsequent::new("y", vec![10.0, 8.0])], // y = 10 + 8*x
));

engine.set_input("x", 3.0).unwrap();
let result = engine.compute().unwrap();
println!("y = {:.2}", result["y"]);
```

### TSK features

- **Zero-order TSK**: constant output per rule (bias coefficient only)
- **First-order TSK**: linear polynomial `c₀ + c₁·x₁ + c₂·x₂ + ...`
- **Multiple outputs**: each rule can have several consequents
- **Rule weights**: `rule.with_weight(0.8)` scales contribution
- **Expression antecedents**: `TskRule::from_expression(expr, ...)` for nested AND/OR logic
- **Output clamping**: results clamped to the output universe bounds

---

## PSO (Particle Swarm Optimization)

Optimize fuzzy system parameters — membership function parameters, TSK coefficients, or rule weights — with a zero-dependency PSO implementation (uses built-in SplitMix64 PRNG).

```rust
use logicfuzzy_academic::{PsoConfig, PsoOptimizer};

// Minimize the sphere function f(x) = x² + y²
let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();

let config = PsoConfig {
    population_size: 50,
    max_iterations: 500,
    inertia_weight: 0.729,          // w
    cognitive_coefficient: 1.494,   // c1
    social_coefficient: 1.494,      // c2
    bounds: vec![(-10.0, 10.0), (-10.0, 10.0)],
    velocity_limit: Some(2.0),
    tolerance: 1e-8,
    patience: 50,
    seed: Some(42),                 // None = use system time
};

let mut optimizer = PsoOptimizer::new(config);
let (best_pos, best_fit, state) = optimizer.optimize(sphere);
println!("Best: {best_pos:?}  fitness: {best_fit:.6}");
println!("Converged: {}  iterations: {}", state.converged, state.iteration);
```

### PSO configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 30 | Number of particles |
| `max_iterations` | 1000 | Maximum iterations |
| `inertia_weight` | 0.729 | Velocity retention (w) |
| `cognitive_coefficient` | 1.494 | Attraction to personal best (c1) |
| `social_coefficient` | 1.494 | Attraction to global best (c2) |
| `bounds` | — | Per-dimension `(min, max)` |
| `velocity_limit` | `None` | Max velocity per dimension |
| `tolerance` | 1e-8 | Early-stopping convergence threshold |
| `patience` | 50 | Iterations without improvement before stop |
| `seed` | `None` | Reproducible PRNG seed |

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

### `tsk_rule!` / `tsk_output!` — TSK helpers

```rust
use logicfuzzy_academic::tsk_rule;

// Zero-order: single bias coefficient
let rule = tsk_rule!(IF x IS small THEN y => [25.0]);

// First-order: bias + one coefficient per antecedent
let rule = tsk_rule!(IF x IS small AND y IS low THEN z => [5.0, 2.0, 3.0]);

// NOT and OR also supported
let rule = tsk_rule!(IF x IS NOT cold OR y IS high THEN z => [10.0, 1.0, 1.0]);

// Register output in TskEngine
tsk_output!(engine, "y", 0.0, 100.0, 101);
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
├── macros.rs     — rule!, fuzzy_var!, antecedent!, consequent!, var_svg!, export_svg!, tsk_rule!, tsk_output!
├── tsk.rs        — TskEngine, TskRule, TskConsequent (Takagi-Sugeno-Kang inference)
├── pso.rs        — PsoOptimizer, PsoConfig, PsoState (zero-dependency SplitMix64 PRNG)
└── lib.rs        — public re-exports

examples/
└── demo.rs       — two complete Mamdani systems (tip control + irrigation) with SVG export

.github/workflows/
├── ci.yml        — test, clippy, fmt, coverage (llvm-cov + Codecov + SonarCloud)
└── mutation.yml  — cargo-mutants with dynamic badge update
```

### Pipeline

**Mamdani:**
```
crisp inputs  →  fuzzification  →  inference (AND=min, OR=max, NOT=1-μ)
             →  clip (Mamdani implication)  →  aggregation (max)
             →  defuzzification (Centroid / Bisector / MOM / SOM / LOM)
             →  crisp output
```

**TSK (Takagi-Sugeno-Kang):**
```
crisp inputs  →  fuzzification  →  inference (same AND/OR/NOT)
             →  polynomial consequents  →  weighted average Σ(α·f(x)) / Σ(α)
             →  crisp output
```

---

## Running

```bash
git clone https://github.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic
cd logicfuzzy-academic
cargo run --example demo          # two Mamdani systems + SVG export to output/
cargo test                        # full test suite (460 unit + 14 integration + 45 doc-tests)
cargo clippy --tests -- -D warnings  # lint check (tests included)
cargo mutants                     # mutation testing (full suite)
cargo mutants -f src/svg.rs --timeout 60   # mutation testing for a specific module
```

### Using TSK and PSO in your project

Add to your `Cargo.toml`:
```toml
[dependencies]
logicfuzzy_academic = "0.1.9"
```

Then:
```rust
use logicfuzzy_academic::{
    // Mamdani
    MamdaniEngine, DefuzzMethod,
    // TSK
    TskEngine, TskRule, TskConsequent,
    // PSO
    PsoConfig, PsoOptimizer, PsoState,
    // Shared
    FuzzyVariable, Universe, Term, MembershipFn,
    FuzzyError,
    rule::{Antecedent, Connector, RuleBuilder},
    macros::rule,
};
```

---

## Mutation Testing

This project uses [`cargo-mutants`](https://github.com/sourcefrog/cargo-mutants) to measure the **quality of the test suite** — not just coverage, but whether tests actually catch logical errors.

The mutation workflow runs automatically on every push to `main` and updates the badge dynamically via a dedicated `mutation-badge` branch. The current score is always reflected in the badge at the top of this file.

> A mutation score above 70% is considered strong for a pure-logic library with complex floating-point arithmetic.

---

## SonarCloud

This project is continuously analyzed by [SonarCloud](https://sonarcloud.io/summary/new_code?id=Benjamin-Yuji-Suzuki_logicfuzzy-academic), covering:

- Code smells and maintainability
- Security vulnerabilities and hotspots
- Code duplication
- Test coverage integration (via lcov)

Current status is always reflected in the badges at the top of this file.

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

## 🤖 AI Usage Declaration

The development of this library involved AI assistance throughout the project. The table below declares all tools used, their purpose, and the human review performed, in compliance with the academic integrity requirements of the course.

| Tool | Purpose | Summary of use | Human review performed |
|------|---------|----------------|------------------------|
| **Claude Pro (Anthropic)** | Initial architecture, base code generation, bug fixing, and suggestions | Used at the start of the project to establish the library structure, core pipeline (fuzzification → inference → defuzzification), macro system, and SVG renderer. Also used for early test suggestions and code corrections. | All generated code was reviewed, tested, and validated by the author. Architectural decisions were understood and justified before acceptance. |
| **Claude Free (Anthropic)** | Code correction, test suggestions, CI/CD configuration, README writing | After Claude Pro credits were exhausted, Claude Free was used to continue fixing bugs, suggesting and reviewing tests, configuring GitHub Actions workflows (coverage, mutation testing, SonarCloud), and writing this README. | Every suggestion was evaluated, tested locally, and accepted only after validation. CI pipeline failures were debugged collaboratively. |
| **DeepSeek** | Code analysis, test creation, and autonomous self-correction | Used in parallel as a free alternative to listen to the same analyses made with Claude, recommend additional tests, and generate more code. DeepSeek was particularly useful for autonomous self-correction cycles. | All DeepSeek output was reviewed critically. Suggestions that conflicted with existing design decisions were rejected or adapted. |
| **OpenCode (DeepSeek V4 Flash)** | TSK engine implementation, PSO optimizer implementation, documentation update | Used to implement the entire TSK module (`src/tsk.rs`), PSO module (`src/pso.rs`) with zero-dependency SplitMix64 PRNG, update README with TSK/PSO documentation, update CHANGELOG, update `.gitignore`, and create OpenCode skills. | All generated code was reviewed, tested (430+ tests passing), and validated with clippy and rustfmt. The `rand` dependency initially added was later removed to preserve the zero-dependency philosophy. |

> **Note:** AI usage was declared openly and does not reduce the academic evaluation score. What matters is that the team understood, reviewed, and validated everything that was accepted into the codebase.

---

## License

MIT © 2026 [Benjamin Yuji Suzuki](https://github.com/Benjamin-Yuji-Suzuki)

See [LICENSE](./LICENSE) for the full text.