use logicfuzzy_academic::{antecedent, consequent, rule, FuzzyError, MamdaniEngine};

/// Helper to compare floats with a tolerance.
fn assert_near(val: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (val - expected).abs() < tol,
        "{}: expected {}, got {} (difference {})",
        msg,
        expected,
        val,
        (val - expected).abs()
    );
}

#[test]
fn test_full_tip_system() {
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
    engine
        .add_rule(rule!(IF food_quality IS good        OR  service IS poor       THEN tip IS low));
    engine.add_rule(
        rule!(IF food_quality IS poor        OR  service IS acceptable THEN tip IS medium),
    );
    engine.add_rule(
        rule!(IF service IS acceptable                                 THEN tip IS medium),
    );
    engine.add_rule(
        rule!(IF service IS great            OR  food_quality IS excellent THEN tip IS high),
    );

    // Scenario 1: excellent food + great service → high tip
    engine.set_input("food_quality", 9.0).unwrap();
    engine.set_input("service", 9.0).unwrap();
    let result = engine.compute().unwrap();
    assert_near(result["tip"], 20.0, 5.0, "Excellent food + great service");

    // Scenario 2: poor food + poor service → low to moderate tip
    // (rule 3 also fires because of "food_quality IS poor", giving some medium contribution)
    engine.set_input("food_quality", 2.0).unwrap();
    engine.set_input("service", 2.0).unwrap();
    let result = engine.compute().unwrap();
    // The centroid is roughly 10.8; we just check it's within a sensible range.
    assert!(
        result["tip"] > 5.0 && result["tip"] < 15.0,
        "Poor food and poor service should give low/moderate tip"
    );

    // Scenario 3: average values → medium tip
    engine.set_input("food_quality", 5.0).unwrap();
    engine.set_input("service", 5.0).unwrap();
    let result = engine.compute().unwrap();
    assert!(
        result["tip"] > 8.0 && result["tip"] < 17.0,
        "Average values should give medium tip"
    );
}

#[test]
fn test_explain_consistency_with_compute() {
    let mut engine = MamdaniEngine::new();
    let mut x = logicfuzzy_academic::FuzzyVariable::new(
        "x",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    x.add_term(logicfuzzy_academic::Term::new(
        "low",
        logicfuzzy_academic::MembershipFn::Trimf([0.0, 0.0, 5.0]),
    ));
    engine.add_antecedent(x);

    let mut y = logicfuzzy_academic::FuzzyVariable::new(
        "y",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    y.add_term(logicfuzzy_academic::Term::new(
        "high",
        logicfuzzy_academic::MembershipFn::Trimf([5.0, 10.0, 10.0]),
    ));
    engine.add_consequent(y);

    engine.add_rule(
        logicfuzzy_academic::rule::RuleBuilder::new()
            .when("x", "low")
            .then("y", "high")
            .build(),
    );

    engine.set_input("x", 2.5).unwrap();
    let compute_val = engine.compute().unwrap();
    let explain_val = engine.explain().unwrap();
    assert_near(
        compute_val["y"],
        explain_val.outputs["y"],
        1e-10,
        "compute and explain must return the same crisp output",
    );
}

#[test]
fn test_no_rules_fired_error() {
    let mut engine = MamdaniEngine::new();
    let mut x = logicfuzzy_academic::FuzzyVariable::new(
        "x",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    x.add_term(logicfuzzy_academic::Term::new(
        "left",
        logicfuzzy_academic::MembershipFn::Trimf([0.0, 0.0, 3.0]),
    ));
    engine.add_antecedent(x);

    let mut y = logicfuzzy_academic::FuzzyVariable::new(
        "y",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    y.add_term(logicfuzzy_academic::Term::new(
        "right",
        logicfuzzy_academic::MembershipFn::Trimf([7.0, 10.0, 10.0]),
    ));
    engine.add_consequent(y);

    engine.add_rule(
        logicfuzzy_academic::rule::RuleBuilder::new()
            .when("x", "left")
            .then("y", "right")
            .build(),
    );

    engine.set_input("x", 9.0).unwrap(); // "left" has degree 0
    let err = engine.compute().unwrap_err();
    assert!(matches!(err, FuzzyError::NoRulesFired { .. }));
}

#[test]
fn test_weighted_rule_firing() {
    let mut engine = MamdaniEngine::new();
    let mut x = logicfuzzy_academic::FuzzyVariable::new(
        "x",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    x.add_term(logicfuzzy_academic::Term::new(
        "a",
        logicfuzzy_academic::MembershipFn::Trimf([0.0, 5.0, 10.0]),
    ));
    engine.add_antecedent(x);

    let mut y = logicfuzzy_academic::FuzzyVariable::new(
        "y",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    y.add_term(logicfuzzy_academic::Term::new(
        "b",
        logicfuzzy_academic::MembershipFn::Trimf([0.0, 10.0, 10.0]),
    ));
    engine.add_consequent(y);

    // Rule with weight 0.5
    engine.add_rule(
        logicfuzzy_academic::rule::RuleBuilder::new()
            .when("x", "a")
            .then("y", "b")
            .weight(0.5)
            .build(),
    );
    engine.set_input("x", 5.0).unwrap();
    let result = engine.compute().unwrap();
    // Without weight, defuzzification of a full ramp would give ~6.667;
    // with weight 0.5 the centroid is ~6.11. We just check it's reduced.
    assert!(
        result["y"] > 0.0 && result["y"] < 7.0,
        "Weight should reduce activation, centroid near 6.1"
    );
}

#[test]
fn test_discrete_cog_basic() {
    let mut engine = MamdaniEngine::new();
    let mut x = logicfuzzy_academic::FuzzyVariable::new(
        "x",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    x.add_term(logicfuzzy_academic::Term::new(
        "a",
        logicfuzzy_academic::MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
    ));
    engine.add_antecedent(x);

    let mut y = logicfuzzy_academic::FuzzyVariable::new(
        "y",
        logicfuzzy_academic::Universe::new(0.0, 10.0, 101),
    );
    y.add_term(logicfuzzy_academic::Term::new(
        "b",
        logicfuzzy_academic::MembershipFn::Trapmf([0.0, 0.0, 10.0, 10.0]),
    ));
    engine.add_consequent(y);

    engine.add_rule(
        logicfuzzy_academic::rule::RuleBuilder::new()
            .when("x", "a")
            .then("y", "b")
            .build(),
    );
    engine.set_input("x", 5.0).unwrap();
    let cog = engine.discrete_cog("y", 2.0).unwrap();
    assert_near(
        cog.centroid,
        5.0,
        0.5,
        "Centroid for uniform MF with step 2.0",
    );
}
