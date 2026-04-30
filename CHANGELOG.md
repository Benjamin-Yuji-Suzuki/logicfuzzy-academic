# Changelog

All notable changes to `logicfuzzy-academic` are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Pending (next major feature)
- Takagi-Sugeno inference model (alternative to Mamdani)

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
