# Contributing

This is primarily a solo academic project, but issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/Benjamin-Yuji-Suzuki/logicfuzzy-academic
cd logicfuzzy-academic
cargo build
```

## Running tests

```bash
cargo test              # unit tests + integration + doctests
```

## Linting and formatting

```bash
cargo clippy --tests -- -D warnings   # must produce zero warnings
cargo fmt                              # auto-format
cargo fmt --check                      # check without modifying
```

## Running the demo

```bash
cargo run --example demo
# SVGs are written to output/gorjeta/ and output/irrigacao/
```

## Before submitting a PR

- `cargo test` passes with zero failures
- `cargo clippy --tests -- -D warnings` produces no warnings
- `cargo fmt --check` passes
- New public API has `///` doc comments in English
- New behaviour has at least one unit test

## Project structure

```
src/
├── error.rs      — FuzzyError enum
├── membership.rs — trimf, trapmf, gaussmf
├── variable.rs   — Universe, Term, FuzzyVariable
├── rule.rs       — Antecedent, Rule, RuleBuilder, Connector
├── engine.rs     — MamdaniEngine (full pipeline)
├── explain.rs    — ExplainReport, CogTable
├── svg.rs        — pure-Rust SVG renderer
├── macros.rs     — rule!, fuzzy_var!, antecedent!, consequent!, var_svg!, export_svg!, tsk_rule!, tsk_output!
├── tsk.rs        — TskEngine, TskRule, TskConsequent (TSK inference)
├── pso.rs        — PsoOptimizer, PsoConfig, PsoState (zero-dependency PSO)
└── lib.rs        — public re-exports

examples/
└── demo.rs       — two complete Mamdani systems with SVG export
```
