use crate::error::FuzzyError;
use crate::rule::{Antecedent, Connector, Expression};
use crate::variable::FuzzyVariable;
use std::collections::BTreeMap;

pub(crate) fn set_input_impl(
    antecedents: &BTreeMap<String, FuzzyVariable>,
    inputs: &mut BTreeMap<String, f64>,
    name: &str,
    value: f64,
) -> Result<(), FuzzyError> {
    if !value.is_finite() {
        return Err(FuzzyError::InvalidInput {
            variable: name.to_string(),
            value,
        });
    }

    let var = antecedents
        .get(name)
        .ok_or_else(|| FuzzyError::MissingInput(name.to_string()))?;

    let min = var.universe.min;
    let max = var.universe.max;

    if value < min || value > max {
        let clamped = value.clamp(min, max);
        inputs.insert(name.to_string(), clamped);
        return Err(FuzzyError::InputOutOfRange {
            variable: name.to_string(),
            value,
            min,
            max,
        });
    }

    inputs.insert(name.to_string(), value);
    Ok(())
}

pub(crate) fn set_input_unchecked_impl(
    antecedents: &BTreeMap<String, FuzzyVariable>,
    inputs: &mut BTreeMap<String, f64>,
    name: &str,
    value: f64,
) {
    if let Err(e) = set_input_impl(antecedents, inputs, name, value) {
        match e {
            FuzzyError::MissingInput(_) => panic!("Variable '{}' not registered", name),
            FuzzyError::InputOutOfRange { .. } => {}
            FuzzyError::InvalidInput { .. } => {
                panic!("Invalid input value for variable '{}'", name)
            }
            FuzzyError::InvalidRule { .. } => unreachable!(),
            FuzzyError::NoRulesFired { .. } => unreachable!(),
            FuzzyError::DuplicateVariable(_) => unreachable!(),
        }
    }
}

pub(crate) fn firing_strength_impl(
    expression: Option<&Expression>,
    antecedents: &[Antecedent],
    connector: &Connector,
    weight: f64,
    inputs: &BTreeMap<String, f64>,
    ant_vars: &BTreeMap<String, FuzzyVariable>,
) -> f64 {
    if let Some(expr) = expression {
        return (expr.eval(inputs, ant_vars) * weight).clamp(0.0, 1.0);
    }

    let mut degrees = Vec::with_capacity(antecedents.len());
    for ant in antecedents {
        match ant.eval(inputs, ant_vars) {
            Some(d) => degrees.push(d),
            None => return 0.0,
        }
    }

    if degrees.is_empty() {
        return 0.0;
    }

    let firing = match connector {
        Connector::And => degrees.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        Connector::Or => degrees.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
    };

    (firing * weight).clamp(0.0, 1.0)
}

/// Core diagnostics for when no rules fire: checks each antecedent's membership.
pub(crate) fn build_no_rules_diagnostics_impl(
    antecedents: &BTreeMap<String, FuzzyVariable>,
    inputs: &BTreeMap<String, f64>,
) -> Vec<String> {
    let mut diagnostics = Vec::new();
    for (name, var) in antecedents {
        if let Some(&crisp) = inputs.get(name) {
            let degrees = var.fuzzify(crisp);
            let max_deg = degrees.iter().map(|(_, d)| *d).fold(0.0_f64, f64::max);
            if max_deg <= 0.0 {
                diagnostics.push(format!(
                    "Antecedent '{}' has crisp value {} but all membership degrees are zero",
                    name, crisp
                ));
            } else {
                diagnostics.push(format!(
                    "Antecedent '{}' has non-zero degrees (max {:.4}) but no rule matched the combination",
                    name, max_deg
                ));
            }
        }
    }
    diagnostics
}

/// Exports SVG for all antecedents into the given directory.
pub(crate) fn export_antecedent_svgs(
    antecedents: &BTreeMap<String, FuzzyVariable>,
    inputs: &BTreeMap<String, f64>,
    dir: &str,
) -> std::io::Result<()> {
    use std::fs;
    use std::path::Path;
    for (name, var) in antecedents {
        let input = inputs.get(name.as_str()).copied();
        let svg = crate::svg::render_variable_svg(var, input);
        let path = Path::new(dir).join(format!("{}.svg", name));
        fs::write(path, svg)?;
    }
    Ok(())
}

pub(crate) fn format_rule_condition(
    expression: Option<&Expression>,
    antecedents: &[Antecedent],
    connector: &Connector,
) -> String {
    if let Some(expr) = expression {
        expr.to_string()
    } else {
        let conn = match connector {
            Connector::And => "AND",
            Connector::Or => "OR",
        };
        let ants: Vec<String> = antecedents
            .iter()
            .map(|a| {
                if a.negated {
                    format!("({} IS NOT {})", a.var, a.term)
                } else {
                    format!("({} IS {})", a.var, a.term)
                }
            })
            .collect();
        ants.join(&format!(" {} ", conn))
    }
}
