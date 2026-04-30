# logicfuzzy-academic

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

* **AI Assistance:** The logic, architecture, and development of this library were built with the assistance of Anthropic's **Claude Pro**. It was used as an advanced pair-programming tool to ensure high-quality, safe, and idiomatic Rust code.

---

## Features

- **Complete Mamdani pipeline** — fuzzification → inference → aggregation → centroid defuzzification
- **Membership functions** — `trimf` (triangular), `trapmf` (trapezoidal), `gaussmf` (gaussian), including open shoulders
- **`rule!` macro** — declarative DSL for writing fuzzy rules in natural language
- **`fuzzy_var!` macro** — creates a `FuzzyVariable` with universe and all terms in one block
- **`antecedent!` / `consequent!` macros** — creates and registers variables directly in the engine
- **`var_svg!` macro** — generates an SVG string from a variable, with or without an input marker
- **`export_svg!` macro** — exports all engine variables to SVG files in one call
- **`RuleBuilder`** — fluent builder API as an alternative to the `rule!` macro
- **AND / OR connectors** — min (t-norm) and max (s-norm)
- **`explain()`** — full pipeline report: fuzzification degrees, rule firing strengths, crisp output
- **`discrete_cog()`** — step-by-step Centre-of-Gravity table matching textbook notation
- **SVG visualization** — colour legend, μ annotations, clipped activation areas, aggregated output plot
- **Zero fuzzy dependencies** — only Rust `std`
- **165 tests** — unit tests + doctests covering the full pipeline

---

## Quick start

```toml
[dependencies]
fuzzy_mamdani = { path = "." }
```

```rust
use fuzzy_mamdani::{MamdaniEngine, antecedent, consequent, rule, export_svg};

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

    engine.set_input("temperature", 45.0);
    engine.set_input("humidity",    90.0);

    let result = engine.compute();
    println!("fan_speed = {:.2}%", result["fan_speed"]); // ~67%

    // Export SVGs for all variables + aggregated output
    export_svg!(engine, "output/", aggregated);
}
```

---

## Macros

### `fuzzy_var!` — create a variable

Creates a `FuzzyVariable` with its universe and all linguistic terms in one expression.
Supports `trimf`, `trapmf`, and `gaussmf` — mixed in the same block.

```rust
let temp = fuzzy_var!("temperature", 0.0, 50.0, 501,
    "cold"   => trimf   [0.0,  0.0, 25.0],
    "warm"   => trimf   [0.0, 25.0, 50.0],
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

### `rule!` — natural language rules

```rust
rule!(IF temperature IS hot AND humidity IS high THEN fan_speed IS fast)
rule!(IF temperature IS hot OR  humidity IS high THEN fan_speed IS fast)
rule!(IF smoke_level IS high AND ambient_temp IS critical THEN alert_level IS maximum)
```

Or with the fluent `RuleBuilder`:

```rust
RuleBuilder::new()
    .when("temperature", "hot")
    .or("humidity", "high")
    .then("fan_speed", "fast")
```

### `var_svg!` — generate SVG from a variable

```rust
use fuzzy_mamdani::{fuzzy_var, var_svg};

let var = fuzzy_var!("temperature", 0.0, 50.0, 501,
    "cold" => trimf [0.0,  0.0, 25.0],
    "hot"  => trimf [25.0,50.0, 50.0],
);

// Clean MF plot — no input marker
let svg = var_svg!(var);
std::fs::write("temperature.svg", svg).unwrap();

// With input marker at x = 35.0
let svg = var_svg!(var, 35.0);
std::fs::write("temperature_35.svg", svg).unwrap();
```

### `export_svg!` — export all engine variables

```rust
use fuzzy_mamdani::export_svg;

engine.set_input("temperature", 45.0);
engine.compute();

// Membership function SVGs only
export_svg!(engine, "output/");

// Membership + aggregated output SVGs
export_svg!(engine, "output/", aggregated);
```

Each call prints `✓` on success or `✗ Error: …` on failure. Antecedents automatically include the input marker when `set_input` has been called.

---

## SVG visualization

Every SVG is self-contained and opens in any browser — no matplotlib, no Python, no tooling.

Each plot includes:
- Coloured curves per linguistic term
- Clipped activation area when an input is set
- Horizontal dashed lines + dot annotations showing `μ_term(x) = value`
- Vertical dashed marker at the crisp input value
- Colour legend strip at the bottom

The aggregated output SVG additionally shows:
- Original MF curves (dashed)
- Each term clipped at its firing degree `α`
- Grey aggregated envelope
- Yellow centroid marker

```rust
// Individual variable — no input marker
std::fs::write("temp.svg", var.to_svg()).unwrap();

// Individual variable — with input marker
std::fs::write("temp.svg", var.to_svg_with_input(35.0)).unwrap();

// All engine variables at once (antecedents include input markers)
engine.export_svg("output/").unwrap();

// All engine variables + aggregated output SVGs
engine.export_aggregated_svg("output/").unwrap();
```

---

## `explain()` — inspecting the pipeline

```rust
engine.set_input("temperature", 5.0);
engine.set_input("humidity",    10.0);

let report = engine.explain();
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
  ✓ [0.8000]  IF (temperature is cold) AND (humidity is low) THEN fan_speed is slow
  ✗ [0.0000]  IF (temperature is warm) AND (humidity is medium) THEN fan_speed is medium

[ Defuzzification Output ]
  fan_speed = 18.4956
```

You can also inspect the report programmatically:

```rust
for rf in &report.rule_firings {
    if rf.fired {
        println!("{} -> alpha {:.4}", rf.rule_text, rf.firing_degree);
    }
}
for fv in &report.fuzzification {
    println!("{}: dominant = {:?}", fv.variable, fv.dominant_term());
}
println!("fan_speed = {:.4}", report.outputs["fan_speed"]);
```

---

## `discrete_cog()` — step-by-step centroid table

Computes the Centre-of-Gravity at evenly-spaced discrete points, matching the
step-by-step calculation shown in fuzzy control textbooks.

```rust
engine.set_input("moisture",     38.0);
engine.set_input("temperature",  31.0);
engine.compute();

let table = engine.discrete_cog("valve", 10.0).unwrap();
table.print("valve");
```

```
  [ COG table — valve ]
     I_i      mu_agg(I_i)       I_i * mu_agg(I_i)
  ──────────────────────────────────────────────────
     0.0        0.142857                  0.000000
    10.0        0.142857                  1.428571
    ...
    80.0        0.500000                 40.000000
    90.0        0.500000                 45.000000
   100.0        0.500000                 50.000000
  ──────────────────────────────────────────────────
              3.828571                240.285714  <- sums
  Numerator   = 240.285714
  Denominator = 3.828571
  Centroid    = 62.761194
```

---

## Project structure

```
src/
├── lib.rs          — re-exports all public items
├── membership.rs   — trimf, trapmf, gaussmf, MembershipFn enum
├── variable.rs     — Universe, Term, FuzzyVariable, to_svg / to_svg_with_input
├── rule.rs         — Connector (And/Or), Rule, RuleBuilder
├── engine.rs       — MamdaniEngine — full pipeline, explain(), export_svg(), discrete_cog()
├── explain.rs      — ExplainReport, RuleFiring, FuzzifiedVariable, CogTable
├── svg.rs          — pure-Rust SVG renderer (zero dependencies)
└── macros.rs       — rule!, fuzzy_var!, antecedent!, consequent!, var_svg!, export_svg!
```

### Pipeline

```
crisp inputs
    │
    ▼  fuzzification   — μ(x) for each antecedent term
firing strengths
    │
    ▼  inference       — AND = min, OR = max
    │
    ▼  clip            — min(mf, firing_strength)   [Mamdani implication]
    │
    ▼  aggregation     — max across all rules
aggregated MF
    │
    ▼  defuzzification — centroid = Σ(x·μ) / Σ(μ)
crisp output
```

---

## Running the demo

```bash
git clone https://github.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic
cd logicfuzzy-academic
cargo run       # two complete systems + SVG export to output/
cargo test      # 165 tests (unit + doctests)
```

`cargo run` executes two Mamdani systems — Tip Control and Irrigation Control —
printing fuzzification tables, rule firing degrees, discrete COG tables, scenario
results, and writing SVGs to `output/gorjeta/` and `output/irrigacao/`.

---

## Changelog

### v0.1.2
- Added `svg.rs` — pure-Rust SVG renderer with colour legend, μ annotations, clipped activation areas
- Added `FuzzyVariable::to_svg()` and `to_svg_with_input(value)`
- Added `MamdaniEngine::export_svg(dir)` and `export_aggregated_svg(dir)`
- Added `var_svg!` and `export_svg!` macros
- Added `MamdaniEngine::discrete_cog(name, step)` — step-by-step COG table
- Added `CogTable` with `print()` method
- All output text translated to English; clippy clean

### v0.1.1
- Added `fuzzy_var!`, `antecedent!`, `consequent!` macros
- Added `explain()` and `ExplainReport`
- Implemented `fmt::Display` for `Rule`
- All public doc comments translated to English

### v0.1.0
- Initial release — complete Mamdani pipeline, `rule!` macro, `trimf` / `trapmf` / `gaussmf`

---

## Acknowledgements

This library was designed to be a functional equivalent of **[scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy)** for Python, which served as the primary reference for the pipeline architecture, membership function definitions, and defuzzification method.

The theoretical foundation follows the original work by:

- **Lotfi A. Zadeh** — *Fuzzy Sets* (1965), Information and Control
- **E. H. Mamdani & S. Assilian** — *An experiment in linguistic synthesis with a fuzzy logic controller* (1975), International Journal of Man-Machine Studies

---

## License

MIT © 2025 [Benjamin Yuji Suzuki](https://github.com/Benjamin-Yuji-Suzuki)

See [LICENSE](./LICENSE) for the full text.