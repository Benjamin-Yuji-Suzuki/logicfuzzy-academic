# Changelog

All notable changes to `logicfuzzy-academic` are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.9] — 2026-05-14

### Added
- **TSK (Takagi-Sugeno-Kang) inference engine** (`src/tsk.rs`):
  - `TskEngine` — manages antecedents, outputs, and TSK rules; computes weighted-average output
  - `TskRule` — fuzzy antecedents + crisp polynomial consequents (`TskConsequent`)
  - Zero-order (constant) and first-order (linear) polynomial consequents
  - Multiple outputs per rule, rule weights, expression-based antecedents
  - Output clamping to universe bounds, full `Result`-based error handling
  - 23 unit tests covering zero-order, first-order, weighted average, multi-output, expression-based rules, and edge cases

- **PSO (Particle Swarm Optimization) optimizer** (`src/pso.rs`):
  - `PsoOptimizer` with configurable population, inertia, cognitive/social coefficients
  - Per-dimension bounds, velocity limit, early stopping via tolerance + patience
  - Reproducible via `seed: Option<u64>`, system time used when `None`
  - Built-in SplitMix64 PRNG — **zero external dependencies**
  - 15 unit tests covering sphere function (1D/2D/10D), Rosenbrock, bounds enforcement, convergence, and seed determinism

### Changed
- **`engine.rs` — `validate_rules`**: reduced cognitive complexity from 26 → 15 by extracting helpers `rule_antecedents`, `validate_antecedent`, and `validate_consequent`.
- **`engine.rs` — `firing_degrees_by_consequent`**: reduced cognitive complexity from 16 → 15 by extracting static helper `update_firing_entry`.
- **`engine.rs` — `defuzzify`**: reduced cognitive complexity from 20 → 15 by extracting dedicated functions per defuzzification method and shared helper `max_membership`.
- **`svg.rs`**: reduced cognitive complexity of `render_variable_svg` from 24 → 15 and test helper from 22 → 15.
- **`src/engine.rs`, `src/tsk.rs`, `src/pso.rs`**: added mutation-killing boundary tests (30+ new tests targeting comparison and arithmetic mutants)
- `README.md`: expanded with TSK quick start, PSO quick start, updated pipeline diagrams, updated project structure, updated AI usage declaration, tsk_rule!/tsk_output! macro documentation
- `CHANGELOG.md`: updated test counts for v0.1.9
- `CONTRIBUTING.md`: updated clippy command and project structure
- `.gitignore`: added `/regras_de_negocio` to ignore business rule files

### CI
- 460 unit tests (427 existing + 30+ new) + 14 integration/E2E/concurrency + 45 doc-tests = 519 total
- Clippy `--tests -- -D warnings` and `cargo fmt --check` clean

---

## [0.1.8] — 2026-05-06
### Added
- **Mutation testing** with `cargo-mutants` and CI workflow (`mutation.yml`):
  - 716 mutants tested, detection rate of ~72.6% (520 caught, 178 missed, 18 unviable).
  - Dynamic badge deployed to `mutation-badge` branch via shields.io.
- **SonarCloud** integration in CI for code quality and coverage analysis.
- Integration tests (`tests/integration_tests.rs`):
  - Full tip system with multiple scenarios.
  - Consistency between `compute()` and `explain()`.
  - `NoRulesFired` error handling.
  - Weighted rule firing verification.
  - Discrete COG centroid accuracy.
- End‑to‑end test (`tests/e2e_tests.rs`):
  - Simulates a complete irrigation control system from construction to SVG export.
  - Validates SVG generation (aggregated output, membership plots with input marker).
  - Ensures fuzzy pipeline output stays within expected bounds.
- Concurrency test (`tests/concurrency_tests.rs`): verifies `MamdaniEngine: Clone` can be used across threads.
- **`svg.rs` — comprehensive test suite**: exact coordinate validation, boundary conditions,
  literal assertions for `px`, `py`, `fv`, `draw_grid_axes`, `draw_legend`,
  `draw_intersection`, `sample_curve`, `render_variable_svg`, `render_aggregated_svg`,
  and multiple intersection label placement.
- Complete SVG tests with hardcoded numeric results to kill arithmetic and comparison mutants.

### Fixed
- **`discrete_cog`**: replaced `while` loop with deterministic `for` loop to prevent
  infinite loops caused by `+=` → `-=` mutation in `cargo-mutants`.
- **`mutation.yml`**: stabilized CI workflow by using `shell: bash {0}` and
  `grep`/`sed` for robust log parsing.
- **`svg.rs` legend positioning** and **draw_intersection coordinate clamping** adjusted to match actual SVG output.

### Changed
- Test count increased from 274 to 391 (unit tests only).
- Mutation score badge now live at
  `https://img.shields.io/endpoint?url=…/mutation-badge/mutation.json`.

## [0.1.7] — 2026-05-03

### Fixed
- `compute()` and `explain()` restored to `&self` (using `AtomicBool` for interior mutability), fixing a breaking change in `0.1.6`.
- `InvalidRule` variant no longer includes a misleading `index` field.
- `MamdaniEngine` kept `Sync` via `AtomicBool` and manual `Clone`.

### Added
- Added `# Examples` code blocks to all public API items; documentation coverage improved significantly.
- `rules_dirty` caching: first `compute()`/`explain()` call after modifications triggers automatic rule validation.
- `reset_inputs()` method to clear all crisp inputs.
- `Universe::contains(x)` helper.
- `Rule::is_expression_based()` helper.
- Doc comments on all public methods and structs.

### Changed
- `set_input_unchecked` documentation improved to indicate internal/test use; visibility remains `pub`.
- `ExplainReport::bar` now clamps degrees to [0,1] to avoid UB with NaN.
- Improved documentation for `interp_membership`, `Rule::connector`, and other items.

## [0.1.6] — 2026-05-03

### Fixed
- `src/engine.rs`: `set_input` now rejects NaN and infinite values (new `InvalidInput` error).
- `src/engine.rs`: `explain()` now returns `Err(MissingInput)` when crisp inputs are missing, instead of silently using `0.0`.
- `src/engine.rs`: `validate_rules()` now correctly inspects expression‑based rules via `Expression::antecedents()`.
- `src/rule.rs`: `RuleBuilder::build()` ensures weight validation panics for negative weights.

### Added
- `src/error.rs`: new `InvalidInput` variant in `FuzzyError` for non‑finite values.
- `src/rule.rs`: `Expression::antecedents()` method to collect leaf antecedents from an AST.
- `src/engine.rs`: new tests for edge cases — NaN/inf rejection, missing input in `explain()`, `validate_rules()` with expressions, empty rule bases, defuzzification fallbacks, denominator‑zero in discrete COG, monotonicity with `Bisector`, `when_not` end‑to‑end, etc.
- `membership.rs`: tests for open‑shoulder `trimf` intermediate values.
- `variable.rs`: smoke tests for `to_svg()` output and `Universe::with_resolution` alias.

### Changed
- `src/error.rs`: updated `Display` for all error variants to provide clearer diagnostics.
- CI configuration now includes separate `doc‑test` job, `coverage` job with `cargo-llvm-cov` and Codecov upload, and enhanced SVG verification in the `demo` job.

## [0.1.5] — 2026-05-03

### Fixed
- `examples/demo.rs`: replaced invalid `result["$1"]` with `result["tip"]` and `result["valve"]`, preventing runtime panic. Scenario tables now display correct values.
- `src/rule.rs`: `firing_strength` now returns `0.0` immediately if any antecedent variable or term is missing, instead of silently ignoring the problem.
- `src/rule.rs`: `RuleBuilder::build()` now calls `rule.with_weight(self.weight)`, ensuring weight validation (panic for weights outside `[0.0, 1.0]`).
- `src/engine.rs`: `set_input_unchecked` now panics if the variable is not registered as an antecedent, avoiding silent misleading behavior.
- `src/engine.rs`: `export_aggregated_svg` and `discrete_cog` now correctly handle multiple consequents per rule (added via `also()`).
- `src/engine.rs`: unified aggregation pipeline via private `aggregated_mfs()` method, eliminating code duplication across `compute()`, `explain()`, `export_aggregated_svg()`, and `discrete_cog()`.
- `src/explain.rs`: `dominant_term()` now uses `total_cmp` to safely handle NaN membership degrees.
- `src/engine.rs`: `discrete_cog` now asserts `step > 0.0` to prevent infinite loops.
- `src/error.rs`: `try_add_antecedent` and `try_add_consequent` now return `DuplicateVariable` error instead of misusing `MissingInput`.
- `src/svg.rs`: annotation box width now uses `chars().count()` instead of `len()`, fixing layout with Unicode characters (e.g., `"médio"`).

### Added
- `src/error.rs`: new `DuplicateVariable` variant in `FuzzyError` enum.
- `src/error.rs`: `NoRulesFired` now carries a `diagnostics: Vec<String>` field for detailed debugging.
- `src/engine.rs`: `try_add_antecedent` and `try_add_consequent` methods returning `Result<(), FuzzyError>`.
- `src/engine.rs`: `validate_rules()` method to check that all referenced variables and terms exist.
- `src/engine.rs`: `build_no_rules_fired_error()` private helper for constructing detailed `NoRulesFired` errors.
- `src/engine.rs`: `firing_degrees_by_consequent()` private helper to compute per-consequent firing strengths, reused by `export_aggregated_svg`.
- `src/explain.rs`: `RuleFiring` now stores `consequents: Vec<(String, String)>` instead of single fields, supporting multiple consequents.
- `src/rule.rs`: `Expression` enum (AST) for arbitrary logical combinations of antecedents with `AND`/`OR` nesting.
- `src/rule.rs`: `Rule::from_expression()` constructor and `Rule::expression()` accessor.
- `src/rule.rs`: `RuleBuilder::when_expr()` to accept an `Expression` tree.
- `src/macros.rs`: `rule!` macro expanded to support 4 and 5 antecedents (uniform `AND` or `OR`).
- `src/variable.rs`: `Universe` now caches discrete points internally; `points()` returns `&[f64]` instead of `Vec<f64>`.
- `src/engine.rs`: `MamdaniEngine` now derives `Clone`.
- `.github/workflows/ci.yml`: added verification that SVG outputs are produced by the demo job.
- Comprehensive unit tests for all new features (228 tests total).

### Changed
- Internal maps in `MamdaniEngine` changed from `HashMap` to `BTreeMap`, ensuring deterministic iteration order in `print_summary`, `explain`, and SVG exports.
- `Rule`, `Antecedent`, and `RuleBuilder` now use `BTreeMap` in method signatures to match the engine.
- `examples/demo.rs`: version string updated to `v0.1.5`; audit table "Op" column now reads the connector directly from the rule instead of using a hardcoded array.
- All tests updated to reflect the new APIs and data structures.

---

## [0.1.4] — 2026-04-30

### Fixed
- `MamdaniEngine::export_aggregated_svg` now respects the configured
  `DefuzzMethod` (previously always used hardcoded centroid regardless of
  `set_defuzz_method`)
- `MamdaniEngine::explain` now returns `Err(NoRulesFired)` when no rule
  fires, making it symmetric with `compute()` (previously the error was
  silently swallowed and `Ok` was returned)
- `engine::tests::explain_graus_dentro_do_intervalo` updated to handle
  `Err(NoRulesFired)` at the boundary input `x=25.0` where all membership
  degrees are zero

### Added
- 30 new unit tests in `engine.rs` covering previously untested code paths:
  - `set_input`: `Ok` in-range, `Err(InputOutOfRange)` above/below bounds,
    clamped value still inserted, `Err(MissingInput)` for unknown variable,
    `Ok` at exact universe limits
  - `set_defuzz_method` / `defuzz_method()` getter
  - `DefuzzMethod::Bisector` — uniform MF returns midpoint, result in universe
  - `DefuzzMethod::MeanOfMaximum` — single peak, plateau centre, result in universe
  - `DefuzzMethod::SmallestOfMaximum` — returns ≤ MOM, result in universe
  - `DefuzzMethod::LargestOfMaximum` — returns ≥ MOM, result in universe
  - Invariant `SOM ≤ MOM ≤ LOM` for any MF
  - `discrete_cog`: `None` for unknown consequent, point count, limits
    included, μ in [0,1], products = x×μ, numerator/denominator sums,
    centroid formula, consistency with `compute()`
- `rust-version = "1.70"` in `Cargo.toml`

### Changed
- `src/main.rs` removed (stub redirecting to `examples/demo.rs`);
  `src/main.rs` added to `.gitignore`

---

## [0.1.3] — 2026-04-30

### Fixed
- `rule!` macro: `IS NOT` arms now correctly precede `IS` arms — previously
  `IS NOT cold` was parsed as `IS $t=NOT` causing a compile error
- Clippy `double_ended_iterator_last`: `.filter(..).next_back()` replaced
  with `.rfind(..)` in `DefuzzMethod::LargestOfMaximum`
- Clippy `too_many_arguments`: added `#[allow]` to the private `text()`
  helper in `svg.rs`
- All failing doctests updated for the v0.1.2 API changes

### Added
- `FuzzyError` enum (`MissingInput`, `InputOutOfRange`, `NoRulesFired`)
  implementing `std::error::Error`
- `MamdaniEngine::set_input` now returns `Result<(), FuzzyError>` and
  validates that the value is within the variable's universe; out-of-range
  values are clamped and an `InputOutOfRange` error is returned
- `MamdaniEngine::set_input_unchecked` — convenience wrapper for tests and
  demos that silently clamps without returning a Result
- `MamdaniEngine::compute` now returns `Result<HashMap<String, f64>, FuzzyError>`;
  returns `Err(NoRulesFired)` when all firing degrees are zero
- `MamdaniEngine::explain` now returns `Result<ExplainReport, FuzzyError>`
- `Universe::with_resolution(min, max, n)` — readable alias for `Universe::new`
- `.github/workflows/ci.yml` — GitHub Actions CI running `cargo test`,
  `cargo clippy -D warnings`, and `cargo fmt --check`
- `examples/demo.rs` — demo moved from `src/main.rs` to the standard
  `examples/` directory; run with `cargo run --example demo`
- `CHANGELOG.md` — this file, separated from README

### Changed
- `[[bin]]` in `Cargo.toml` replaced by `[[example]]` pointing to `examples/demo.rs`
- All callers of `compute()` and `explain()` updated to handle `Result`

---

## [0.1.2] — 2026-04-28

### Added
- `svg.rs` — pure-Rust SVG renderer with colour legend, μ annotations,
  clipped activation areas; zero dependencies
- `FuzzyVariable::to_svg()` and `to_svg_with_input(value)`
- `MamdaniEngine::export_svg(dir)` and `export_aggregated_svg(dir)`
- `var_svg!` and `export_svg!` macros
- `MamdaniEngine::discrete_cog(name, step)` — step-by-step COG table
- `CogTable` struct with `print()` method
- `rule!` macro: `NOT` negation, multi-consequent `THEN a IS x AND b IS y`
- `Rule::with_weight(w)` — scales firing degree by `w ∈ [0.0, 1.0]`
- `Antecedent` struct replacing `(String, String)` tuples — carries `negated` flag
- `Rule`: multiple consequents via `Vec<(String, String)>`
- `DefuzzMethod` enum: `Centroid`, `Bisector`, `MeanOfMaximum`,
  `SmallestOfMaximum`, `LargestOfMaximum`
- `MamdaniEngine::set_defuzz_method(method)`
- `RuleBuilder`: `when_not()`, `and_not()`, `or_not()`, `also()`, `weight()`, `build()`

---

## [0.1.1] — 2026-04-27

### Added
- `fuzzy_var!`, `antecedent!`, `consequent!` macros
- `MamdaniEngine::explain()` returning `ExplainReport`
- `ExplainReport::summary()` with visual membership bars
- `FuzzifiedVariable::dominant_term()`
- `impl fmt::Display for Rule`
- All public doc comments (`///`) translated to English

---

## [0.1.0] — 2026-04-25

### Added
- Initial release
- Complete Mamdani pipeline: fuzzification → inference → aggregation → centroid
- Membership functions: `trimf`, `trapmf`, `gaussmf`
- `rule!` declarative macro
- `MamdaniEngine`, `FuzzyVariable`, `Universe`, `Term`, `MembershipFn`