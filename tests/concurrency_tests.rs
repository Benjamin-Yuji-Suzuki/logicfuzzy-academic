use logicfuzzy_academic::{antecedent, consequent, rule, MamdaniEngine};
use std::thread;

#[test]
fn test_engine_can_be_shared_across_threads() {
    let mut engine = MamdaniEngine::new();

    antecedent!(engine, "temp", 0.0, 50.0, 501,
        "cold" => trimf [0.0,  0.0, 25.0],
        "hot"  => trimf [25.0,50.0, 50.0],
    );
    consequent!(engine, "speed", 0.0, 100.0, 1001,
        "slow" => trimf [0.0,  0.0,  50.0],
        "fast" => trimf [50.0,100.0,100.0],
    );
    engine.add_rule(rule!(IF temp IS cold THEN speed IS slow));
    engine.add_rule(rule!(IF temp IS hot  THEN speed IS fast));

    // Clone the engine – clones share the same structure but are independent
    let e1 = engine.clone();
    let e2 = engine.clone();

    let h1 = thread::spawn(move || {
        let mut e = e1;
        e.set_input("temp", 10.0).unwrap();
        let result = e.compute().unwrap();
        result["speed"]
    });

    let h2 = thread::spawn(move || {
        let mut e = e2;
        e.set_input("temp", 40.0).unwrap();
        let result = e.compute().unwrap();
        result["speed"]
    });

    let speed1 = h1.join().unwrap();
    let speed2 = h2.join().unwrap();

    assert!(speed1 < 50.0, "Cold should give slow speed, got {}", speed1);
    assert!(speed2 > 50.0, "Hot should give fast speed, got {}", speed2);
}
