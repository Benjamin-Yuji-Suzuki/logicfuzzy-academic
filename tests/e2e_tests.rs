use logicfuzzy_academic::*;
use std::fs;

/// Checks that the SVG string contains all the given substrings.
fn assert_svg_contains(svg: &str, expected: &[&str]) {
    for token in expected {
        assert!(svg.contains(token), "SVG must contain '{}'", token);
    }
}

#[test]
fn test_irrigation_system_e2e() {
    // Build irrigation system (same as in demo)
    let mut engine = MamdaniEngine::new();

    antecedent!(engine, "moisture", 0.0, 100.0, 1001,
        "low"    => trapmf [ 0.0,  0.0, 30.0,  50.0],
        "medium" => trimf  [30.0, 50.0, 70.0],
        "high"   => trapmf [60.0, 80.0,100.0, 100.0],
    );

    antecedent!(engine, "temperature", 0.0, 40.0, 1001,
        "cold" => trapmf [ 0.0,  0.0, 15.0, 22.0],
        "warm" => trimf  [18.0, 25.0, 32.0],
        "hot"  => trapmf [28.0, 34.0, 40.0, 40.0],
    );

    consequent!(engine, "valve", 0.0, 100.0, 1001,
        "low"    => trapmf [ 0.0,  0.0, 20.0, 40.0],
        "medium" => trimf  [25.0, 50.0, 75.0],
        "high"   => trapmf [60.0, 80.0,100.0,100.0],
    );

    // Rules
    let rules = [
        rule!(IF moisture IS low    AND temperature IS hot  THEN valve IS high),
        rule!(IF moisture IS low    AND temperature IS warm THEN valve IS medium),
        rule!(IF moisture IS medium AND temperature IS hot  THEN valve IS medium),
        rule!(IF moisture IS medium AND temperature IS warm THEN valve IS low),
        rule!(IF moisture IS high    OR temperature IS cold THEN valve IS low),
    ];
    for r in &rules {
        engine.add_rule(r.clone());
    }

    // Main scenario: moisture=38%, temperature=31°C
    engine.set_input("moisture", 38.0).unwrap();
    engine.set_input("temperature", 31.0).unwrap();

    // 1. Compute and check output
    let result = engine.compute().unwrap();
    let valve = result["valve"];
    assert!(
        (0.0..=100.0).contains(&valve),
        "Valve opening must be in [0,100]"
    );
    assert!(
        valve > 20.0 && valve < 80.0,
        "Valve should be moderate for main scenario"
    );

    // 2. Explain and verify something fired
    let report = engine.explain().unwrap();
    assert_eq!(report.fuzzification.len(), 2);
    assert!(report.rules_fired > 0);

    // 3. Export SVGs to a temporary directory
    let dir = std::env::temp_dir().join("logicfuzzy_e2e_test");
    let _ = fs::remove_dir_all(&dir);
    let dir_str = dir.to_str().unwrap();

    engine.export_svg(dir_str).expect("export_svg failed");
    engine
        .export_aggregated_svg(dir_str)
        .expect("export_aggregated_svg failed");

    // 4. Check existence of expected files
    assert!(dir.join("moisture.svg").exists());
    assert!(dir.join("temperature.svg").exists());
    assert!(dir.join("valve.svg").exists());
    assert!(dir.join("valve_aggregated.svg").exists());

    // 5. Inspect aggregated SVG content
    let agg_svg = fs::read_to_string(dir.join("valve_aggregated.svg"))
        .expect("Failed to read aggregated SVG");
    assert_svg_contains(&agg_svg, &["centroid", "aggregated", "valve", "α="]);

    // 6. Test var_svg! with a specific input
    let moisture_var = fuzzy_var!("moisture", 0.0, 100.0, 1001,
        "low"    => trapmf [ 0.0,  0.0, 30.0,  50.0],
        "medium" => trimf  [30.0, 50.0, 70.0],
        "high"   => trapmf [60.0, 80.0,100.0, 100.0],
    );
    let svg_with_input = moisture_var.to_svg_with_input(38.0);
    assert_svg_contains(&svg_with_input, &["μ_low", "38"]);

    // 7. Clean up
    let _ = fs::remove_dir_all(&dir);

    // Verify exact fuzzification values (as documented in demo)
    let report = engine.explain().unwrap();
    let moisture_fv = report
        .fuzzification
        .iter()
        .find(|f| f.variable == "moisture")
        .unwrap();
    let temp_fv = report
        .fuzzification
        .iter()
        .find(|f| f.variable == "temperature")
        .unwrap();

    // mu_low(38) = 0.6, mu_medium(38) = 0.4, mu_high(38) = 0.0
    let low = moisture_fv
        .term_degrees
        .iter()
        .find(|(t, _)| t == "low")
        .unwrap()
        .1;
    let med = moisture_fv
        .term_degrees
        .iter()
        .find(|(t, _)| t == "medium")
        .unwrap()
        .1;
    let high = moisture_fv
        .term_degrees
        .iter()
        .find(|(t, _)| t == "high")
        .unwrap()
        .1;
    assert!((low - 0.6).abs() < 0.001, "mu_low(38) should be 0.6");
    assert!((med - 0.4).abs() < 0.001, "mu_medium(38) should be 0.4");
    assert!((high - 0.0).abs() < 0.001, "mu_high(38) should be 0.0");

    // mu_cold(31) = 0.0, mu_warm(31) ≈ 0.142857, mu_hot(31) = 0.5
    let cold = temp_fv
        .term_degrees
        .iter()
        .find(|(t, _)| t == "cold")
        .unwrap()
        .1;
    let warm = temp_fv
        .term_degrees
        .iter()
        .find(|(t, _)| t == "warm")
        .unwrap()
        .1;
    let hot = temp_fv
        .term_degrees
        .iter()
        .find(|(t, _)| t == "hot")
        .unwrap()
        .1;
    assert!((cold - 0.0).abs() < 0.001, "mu_cold(31) should be 0.0");
    assert!((warm - 0.142857).abs() < 0.001, "mu_warm(31) ~ 0.1429");
    assert!((hot - 0.5).abs() < 0.001, "mu_hot(31) should be 0.5");
}

#[test]
fn test_fan_and_light_system_e2e() {
    // System with two outputs: fan_speed [0,100]% and light_intensity [0,100]%
    let mut engine = MamdaniEngine::new();

    antecedent!(engine, "temperature", 0.0, 40.0, 1001,
        "cold" => trapmf [ 0.0,  0.0, 15.0, 22.0],
        "warm" => trimf  [18.0, 25.0, 32.0],
        "hot"  => trapmf [28.0, 34.0, 40.0, 40.0],
    );

    antecedent!(engine, "humidity", 0.0, 100.0, 1001,
        "low"    => trapmf [ 0.0,  0.0, 30.0, 50.0],
        "medium" => trimf  [30.0, 50.0, 70.0],
        "high"   => trapmf [60.0, 80.0,100.0,100.0],
    );

    consequent!(engine, "fan_speed", 0.0, 100.0, 1001,
        "slow"   => trapmf [ 0.0,  0.0, 20.0, 40.0],
        "medium" => trimf  [25.0, 50.0, 75.0],
        "fast"   => trapmf [60.0, 80.0,100.0,100.0],
    );

    consequent!(engine, "light_intensity", 0.0, 100.0, 1001,
        "low"    => trimf [ 0.0,  0.0, 40.0],
        "medium" => trimf [20.0, 50.0, 80.0],
        "high"   => trimf [60.0,100.0,100.0],
    );

    // Rules with multiple consequents — use RuleBuilder for complex combinations
    engine.add_rule(
        logicfuzzy_academic::rule::RuleBuilder::new()
            .when("temperature", "hot")
            .and("humidity", "high")
            .then("fan_speed", "fast")
            .also("light_intensity", "low")
            .build(),
    );
    engine.add_rule(
        logicfuzzy_academic::rule::RuleBuilder::new()
            .when("temperature", "warm")
            .and("humidity", "medium")
            .then("fan_speed", "medium")
            .also("light_intensity", "medium")
            .build(),
    );
    engine.add_rule(
        logicfuzzy_academic::rule::RuleBuilder::new()
            .when("temperature", "cold")
            .or("humidity", "low")
            .then("fan_speed", "slow")
            .also("light_intensity", "high")
            .build(),
    );

    // Scenario: hot + dry → fast fan, high light? Let's test hot + low humidity
    engine.set_input("temperature", 35.0).unwrap();
    engine.set_input("humidity", 20.0).unwrap();

    let result = engine.compute().unwrap();
    let fan = result["fan_speed"];
    let light = result["light_intensity"];

    assert!(
        (0.0..=100.0).contains(&fan),
        "Fan speed must be within [0,100]"
    );
    assert!(
        (0.0..=100.0).contains(&light),
        "Light intensity must be within [0,100]"
    );

    // Both outputs should be present in explain report
    let report = engine.explain().unwrap();
    assert!(report.outputs.contains_key("fan_speed"));
    assert!(report.outputs.contains_key("light_intensity"));

    // Export SVGs for both consequents
    let dir = std::env::temp_dir().join("logicfuzzy_e2e_multi_test");
    let _ = std::fs::remove_dir_all(&dir);
    engine
        .export_aggregated_svg(dir.to_str().unwrap())
        .expect("export failed");

    assert!(dir.join("fan_speed_aggregated.svg").exists());
    assert!(dir.join("light_intensity_aggregated.svg").exists());

    let _ = std::fs::remove_dir_all(&dir);
}
