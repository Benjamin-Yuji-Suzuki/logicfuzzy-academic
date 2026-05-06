use logicfuzzy_academic::{antecedent, consequent, rule, FuzzyError, MamdaniEngine};

#[test]
fn test_input_below_universe_yields_error() {
    let mut engine = MamdaniEngine::new();
    let var = logicfuzzy_academic::FuzzyVariable::new(
        "x",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    engine.add_antecedent(var);
    let err = engine.set_input("x", -5.0).unwrap_err();
    assert!(matches!(err, FuzzyError::InputOutOfRange { .. }));
}

#[test]
fn test_input_above_universe_yields_error() {
    let mut engine = MamdaniEngine::new();
    let var = logicfuzzy_academic::FuzzyVariable::new(
        "x",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    engine.add_antecedent(var);
    let err = engine.set_input("x", 15.0).unwrap_err();
    assert!(matches!(err, FuzzyError::InputOutOfRange { .. }));
}

#[test]
fn test_nan_input_rejected() {
    let mut engine = MamdaniEngine::new();
    let var = logicfuzzy_academic::FuzzyVariable::new(
        "x",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    engine.add_antecedent(var);
    let err = engine.set_input("x", f64::NAN).unwrap_err();
    assert!(matches!(err, FuzzyError::InvalidInput { .. }));
}

#[test]
fn test_infinite_input_rejected() {
    let mut engine = MamdaniEngine::new();
    let var = logicfuzzy_academic::FuzzyVariable::new(
        "x",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    engine.add_antecedent(var);
    let err = engine.set_input("x", f64::INFINITY).unwrap_err();
    assert!(matches!(err, FuzzyError::InvalidInput { .. }));
}

#[test]
fn test_all_defuzz_methods_produce_values_in_range() {
    // Use a simple system and run all five methods
    let mut engine = MamdaniEngine::new();
    antecedent!(engine, "temp", 0.0, 50.0, 501,
        "cold" => trimf [0.0, 0.0, 25.0],
        "hot"  => trimf [25.0, 50.0, 50.0],
    );
    consequent!(engine, "speed", 0.0, 100.0, 1001,
        "slow" => trimf [0.0, 0.0, 50.0],
        "fast" => trimf [50.0, 100.0, 100.0],
    );
    engine.add_rule(rule!(IF temp IS cold THEN speed IS slow));
    engine.add_rule(rule!(IF temp IS hot  THEN speed IS fast));
    engine.set_input("temp", 30.0).unwrap();

    use logicfuzzy_academic::DefuzzMethod;
    let methods = [
        DefuzzMethod::Centroid,
        DefuzzMethod::Bisector,
        DefuzzMethod::MeanOfMaximum,
        DefuzzMethod::SmallestOfMaximum,
        DefuzzMethod::LargestOfMaximum,
    ];
    for m in &methods {
        let mut e = engine.clone();
        e.set_defuzz_method(m.clone());
        let r = e.compute().unwrap();
        let v = r["speed"];
        // All outputs must lie in the output universe
        assert!(v >= 0.0 && v <= 100.0, "Method {:?} gave {}", m, v);
    }
}

#[test]
fn test_no_rules_fired_with_diagnostics() {
    let mut engine = MamdaniEngine::new();
    let mut x = logicfuzzy_academic::FuzzyVariable::new(
        "x",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    x.add_term(logicfuzzy_academic::Term::new(
        "range",
        logicfuzzy_academic::MembershipFn::Trimf([0.0, 0.0, 1.0]),
    ));
    engine.add_antecedent(x);
    let mut y = logicfuzzy_academic::FuzzyVariable::new(
        "y",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    y.add_term(logicfuzzy_academic::Term::new(
        "out",
        logicfuzzy_academic::MembershipFn::Trimf([0.0, 0.0, 10.0]),
    ));
    engine.add_consequent(y);
    engine.add_rule(
        logicfuzzy_academic::rule::RuleBuilder::new()
            .when("x", "range")
            .then("y", "out")
            .build(),
    );
    engine.set_input("x", 5.0).unwrap(); // degree 0 for "range"
    let err = engine.compute().unwrap_err();
    if let logicfuzzy_academic::FuzzyError::NoRulesFired { diagnostics } = err {
        assert!(
            !diagnostics.is_empty(),
            "should contain diagnostic messages"
        );
        // Verifica que a mensagem correta aparece (max_deg == 0.0)
        assert!(
            diagnostics[0].contains("all membership degrees are zero"),
            "Diagnostic should mention zero membership, got: {}",
            diagnostics[0]
        );
    } else {
        panic!("Expected NoRulesFired error");
    }
}
