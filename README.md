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
let svg = engine.antecedent("temperature").to_svg_with_input(45.0);
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
- **`RuleBuilder`** — fluent builder API as an alternative to the macro
- **AND / OR connectors** — min (t-norm) and max (s-norm)
- **`explain()`** — detailed pipeline report: fuzzification degrees, per-rule firing strengths, and crisp output with human-readable summary
- **SVG visualization** — `to_svg()`, `to_svg_with_input()`, `export_svg()` — zero dependencies, opens in any browser
- **Zero fuzzy dependencies** — only Rust `std`
- **171 tests** — unit tests + doctests covering the full pipeline

---

## Quick start

```toml
# Cargo.toml
[dependencies]
fuzzy_mamdani = { path = "." }
```

```rust
use fuzzy_mamdani::{MamdaniEngine, antecedent, consequent, rule};

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

    // Generate SVG for each variable (with input marker on antecedents)
    engine.export_svg("output/").unwrap();
}
```

---

## Macros

### `fuzzy_var!` — create a variable

Creates a `FuzzyVariable` with universe and all linguistic terms in a single expression.
Supports `trimf`, `trapmf`, and `gaussmf` — can be mixed in the same block.

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

Or using the fluent `RuleBuilder`:

```rust
RuleBuilder::new()
    .when("temperature", "hot")
    .or("humidity",  "high")
    .then("fan_speed", "fast")
```

---

## SVG visualization

The library generates self-contained SVG files for any `FuzzyVariable` — no matplotlib, no Python, no dependencies.

### `FuzzyVariable::to_svg()`

Returns the SVG as a `String` — embeddable in HTML or saveable to disk.

```rust
let var = fuzzy_var!("temperature", 0.0, 50.0, 501,
    "cold" => trimf [0.0,  0.0, 25.0],
    "warm" => trimf [0.0, 25.0, 50.0],
    "hot"  => trimf [25.0,50.0, 50.0],
);
std::fs::write("temperature.svg", var.to_svg()).unwrap();
```

### `FuzzyVariable::to_svg_with_input(value)`

Adds a vertical dashed marker at the current crisp input — useful for showing how a value is fuzzified.

```rust
std::fs::write("temperature.svg", var.to_svg_with_input(35.0)).unwrap();
```

### `MamdaniEngine::export_svg(dir)`

Generates one `.svg` per variable in the given directory. Antecedents automatically include the input marker if `set_input` was called.

```rust
engine.set_input("temperature", 45.0);
engine.set_input("humidity",    90.0);
engine.export_svg("output/").unwrap();
// Writes: output/temperature.svg  output/humidity.svg  output/fan_speed.svg
```

Open any `.svg` directly in a browser — no tooling required.

---

## `explain()` — inspecting the pipeline

`explain()` returns an `ExplainReport` with the full intermediate state of every stage.

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
          low  1.0000  [██████████]
       medium  0.2000  [██░░░░░░░░]
         high  0.0000  [░░░░░░░░░░]
    → dominant term: low

[ Rule Evaluation ] (2 fired, 2 skipped)
  ✓ [0.8000]  IF (temperature is cold) AND (humidity is low) THEN fan_speed is slow
  ✗ [0.0000]  IF (temperature is warm) AND (humidity is medium) THEN fan_speed is medium

[ Defuzzification Output ]
  fan_speed = 18.4956
```

---

## Included examples (`cargo run`)

`src/main.rs` includes two complete systems based on the course Jupyter notebooks:

**Sistema 1 — Gorjeta** (`SISTEMA_FUZZY_GORJETA.ipynb`)

| Variable | Universe | Terms |
|---|---|---|
| qualidade | [0, 10] | ruim / boa / muito_boa |
| servico   | [0, 10] | ruim / aceitavel / otimo |
| gorjeta   | [0, 25]% | baixa / media / alta |

**Sistema 2 — Irrigação** (`SISTEMA_FUZZY_IRRIGACAO.ipynb`)

| Variable    | Universe  | Functions | Terms |
|---|---|---|---|
| umidade     | [0, 100]% | trapmf / trimf / trapmf | baixa / media / alta |
| temperatura | [0, 40]°C | trapmf / trimf / trapmf | fria / morna / quente |
| irrigacao   | [0, 100]% | trapmf / trimf / trapmf | baixa / media / alta |

Running `cargo run` executes both systems, prints fuzzification tables, rule firing degrees, scenario results, and writes SVGs to `output/gorjeta/` and `output/irrigacao/`.

---

## Project structure

```
src/
├── lib.rs          — re-exports all public items
├── membership.rs   — trimf, trapmf, gaussmf, MembershipFn enum
├── variable.rs     — Universe, Term, FuzzyVariable + to_svg / to_svg_with_input
├── rule.rs         — Connector (And/Or), Rule, RuleBuilder
├── engine.rs       — MamdaniEngine — full pipeline + explain() + export_svg()
├── explain.rs      — ExplainReport, RuleFiring, FuzzifiedVariable
├── svg.rs          — pure-Rust SVG renderer (no deps)
└── macros.rs       — rule!, fuzzy_var!, antecedent!, consequent!
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

## Running

```bash
git clone https://github.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic
cd logicfuzzy-academic
cargo run       # runs gorjeta + irrigacao examples, writes SVGs to output/
cargo test      # 171 tests (unit + doctests)
```

---

## Changelog

### v0.1.2
- Added `svg.rs` — pure-Rust SVG renderer (zero dependencies)
- Added `FuzzyVariable::to_svg()` — membership function plot as SVG string
- Added `FuzzyVariable::to_svg_with_input(value)` — SVG with vertical input marker
- Added `MamdaniEngine::export_svg(dir)` — writes one SVG per variable to disk
- Rewrote `src/main.rs` — two full examples based on course Jupyter notebooks (gorjeta + irrigação), including fuzzification tables, rule firing degrees, scenario tables, and SVG export

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
