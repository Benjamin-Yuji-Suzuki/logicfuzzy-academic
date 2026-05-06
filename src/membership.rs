//! # membership.rs
//!
//! Membership functions for Mamdani fuzzy logic.
//! Equivalent to the skfuzzy module from scikit-fuzzy.

// ─────────────────────────────────────────────
// Pure membership functions
// ─────────────────────────────────────────────

/// Triangular membership function (trimf).
/// Parameters: `a` (start), `b` (peak), `c` (end). Requires: `a <= b <= c`.
pub fn trimf(x: f64, a: f64, b: f64, c: f64) -> f64 {
    assert!(a <= b && b <= c, "trimf: requires a <= b <= c");

    // Left open shoulder (a == b)
    if (a - b).abs() < f64::EPSILON {
        if x <= b {
            return 1.0;
        }
        if x >= c {
            return 0.0;
        }
        return (c - x) / (c - b);
    }

    // Right open shoulder (b == c)
    if (b - c).abs() < f64::EPSILON {
        if x >= b {
            return 1.0;
        }
        if x <= a {
            return 0.0;
        }
        return (x - a) / (b - a);
    }

    // Standard triangle: a < b < c
    if x <= a || x >= c {
        return 0.0;
    }
    if x == b {
        return 1.0;
    }
    if x < b {
        (x - a) / (b - a)
    } else {
        (c - x) / (c - b)
    }
}

/// Trapezoidal membership function (trapmf).
/// Parameters: `a` (rise start), `b` (rise end), `c` (fall start), `d` (fall end).
/// Requires: `a <= b <= c <= d`.
pub fn trapmf(x: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
    assert!(
        a <= b && b <= c && c <= d,
        "trapmf: requires a <= b <= c <= d"
    );

    if x >= b && x <= c {
        return 1.0;
    }

    if x >= a && x < b {
        if (b - a).abs() < f64::EPSILON {
            return 1.0;
        }
        return (x - a) / (b - a);
    }

    if x > c && x <= d {
        if (d - c).abs() < f64::EPSILON {
            return 1.0;
        }
        return (d - x) / (d - c);
    }

    0.0
}

/// Gaussian membership function (gaussmf).
/// Parameters: `mean` (center/peak), `sigma` (width, must be > 1e-6 to avoid degeneracy).
pub fn gaussmf(x: f64, mean: f64, sigma: f64) -> f64 {
    assert!(
        sigma > 1e-6,
        "gaussmf: sigma must be > 1e-6 (got {})",
        sigma
    );
    let exp = -((x - mean).powi(2)) / (2.0 * sigma.powi(2));
    exp.exp()
}

// ─────────────────────────────────────────────
// MembershipFn enum
// ─────────────────────────────────────────────

/// Enum representing a fuzzy membership function.
#[derive(Debug, Clone)]
pub enum MembershipFn {
    /// Triangular: [a, b, c]
    Trimf([f64; 3]),
    /// Trapezoidal: [a, b, c, d]
    Trapmf([f64; 4]),
    /// Gaussian: (mean, sigma)
    Gaussmf { mean: f64, sigma: f64 },
}

impl MembershipFn {
    /// Evaluates the membership degree of `x` for this function. Returns a value in [0.0, 1.0].
    pub fn eval(&self, x: f64) -> f64 {
        match self {
            MembershipFn::Trimf([a, b, c]) => trimf(x, *a, *b, *c),
            MembershipFn::Trapmf([a, b, c, d]) => trapmf(x, *a, *b, *c, *d),
            MembershipFn::Gaussmf { mean, sigma } => gaussmf(x, *mean, *sigma),
        }
    }

    /// Evaluates the membership degree for a slice of points (discrete universe).
    pub fn eval_universe(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|&x| self.eval(x)).collect()
    }
}

// ─────────────────────────────────────────────
// Utility: interp_membership
// ─────────────────────────────────────────────

/// Interpolates the membership degree of `value` given a discrete universe
/// and its pre-computed membership vector.
///
/// Values outside the universe are clamped to the nearest boundary membership
/// (constant extrapolation), not linearly extrapolated.
pub fn interp_membership(universe: &[f64], memberships: &[f64], value: f64) -> f64 {
    assert_eq!(
        universe.len(),
        memberships.len(),
        "interp_membership: universe and memberships must have the same length"
    );

    if universe.is_empty() {
        return 0.0;
    }

    let first = universe[0];
    let last = *universe.last().unwrap();

    if value <= first {
        return memberships[0];
    }
    if value >= last {
        return *memberships.last().unwrap();
    }

    let pos = universe.partition_point(|&u| u <= value);
    if pos == 0 {
        return memberships[0];
    }

    let i = pos - 1;
    let x0 = universe[i];
    let x1 = universe[i + 1];
    let y0 = memberships[i];
    let y1 = memberships[i + 1];

    if (x1 - x0).abs() < f64::EPSILON {
        return y0;
    }
    y0 + (y1 - y0) * (value - x0) / (x1 - x0)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // trimf
    #[test]
    fn trimf_peak() {
        assert_eq!(trimf(50.0, 0.0, 50.0, 100.0), 1.0);
    }
    #[test]
    fn trimf_outside_left() {
        assert_eq!(trimf(-1.0, 0.0, 50.0, 100.0), 0.0);
    }
    #[test]
    fn trimf_outside_right() {
        assert_eq!(trimf(101.0, 0.0, 50.0, 100.0), 0.0);
    }
    #[test]
    fn trimf_mid_rise() {
        assert!((trimf(25.0, 0.0, 50.0, 100.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn trimf_mid_fall() {
        assert!((trimf(75.0, 0.0, 50.0, 100.0) - 0.5).abs() < 1e-10);
    }

    // trapmf
    #[test]
    fn trapmf_flat() {
        assert_eq!(trapmf(50.0, 20.0, 30.0, 70.0, 80.0), 1.0);
    }
    #[test]
    fn trapmf_edge_b() {
        assert_eq!(trapmf(30.0, 20.0, 30.0, 70.0, 80.0), 1.0);
    }
    #[test]
    fn trapmf_edge_c() {
        assert_eq!(trapmf(70.0, 20.0, 30.0, 70.0, 80.0), 1.0);
    }
    #[test]
    fn trapmf_outside_left() {
        assert_eq!(trapmf(10.0, 20.0, 30.0, 70.0, 80.0), 0.0);
    }
    #[test]
    fn trapmf_outside_right() {
        assert_eq!(trapmf(90.0, 20.0, 30.0, 70.0, 80.0), 0.0);
    }
    #[test]
    fn trapmf_open_left() {
        assert_eq!(trapmf(0.0, 0.0, 0.0, 30.0, 50.0), 1.0);
    }
    #[test]
    fn trapmf_open_right() {
        assert_eq!(trapmf(100.0, 60.0, 80.0, 100.0, 100.0), 1.0);
    }

    // gaussmf
    #[test]
    fn gaussmf_peak() {
        assert!((gaussmf(50.0, 50.0, 10.0) - 1.0).abs() < 1e-10);
    }
    #[test]
    fn gaussmf_symmetry() {
        assert!((gaussmf(40.0, 50.0, 10.0) - gaussmf(60.0, 50.0, 10.0)).abs() < 1e-10);
    }
    #[test]
    #[should_panic(expected = "sigma must be > 1e-6")]
    fn gaussmf_rejects_zero_sigma() {
        gaussmf(0.0, 0.0, 0.0);
    }
    #[test]
    #[should_panic(expected = "sigma must be > 1e-6")]
    fn gaussmf_rejects_negative_sigma() {
        gaussmf(0.0, 0.0, -1.0);
    }
    #[test]
    #[should_panic(expected = "sigma must be > 1e-6")]
    fn gaussmf_rejects_tiny_sigma() {
        gaussmf(0.0, 0.0, 1e-300);
    }

    // MembershipFn enum
    #[test]
    fn enum_trimf_eval() {
        let mf = MembershipFn::Trimf([0.0, 50.0, 100.0]);
        assert_eq!(mf.eval(50.0), 1.0);
        assert!((mf.eval(25.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn enum_trapmf_eval() {
        let mf = MembershipFn::Trapmf([0.0, 0.0, 30.0, 50.0]);
        assert_eq!(mf.eval(0.0), 1.0);
        assert_eq!(mf.eval(15.0), 1.0);
    }
    #[test]
    fn enum_eval_universe() {
        let mf = MembershipFn::Trimf([0.0, 50.0, 100.0]);
        let u: Vec<f64> = (0..=100).map(|x| x as f64).collect();
        let g = mf.eval_universe(&u);
        assert_eq!(g.len(), 101);
        assert_eq!(g[50], 1.0);
        assert_eq!(g[0], 0.0);
        assert_eq!(g[100], 0.0);
    }

    // interp_membership
    #[test]
    fn interp_exact() {
        let u = vec![0.0, 50.0, 100.0];
        let m = vec![0.0, 1.0, 0.0];
        assert_eq!(interp_membership(&u, &m, 50.0), 1.0);
    }
    #[test]
    fn interp_linear() {
        let u = vec![0.0, 100.0];
        let m = vec![0.0, 1.0];
        assert!((interp_membership(&u, &m, 50.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn interp_outside() {
        let u = vec![0.0, 100.0];
        let m = vec![0.5, 0.8];
        assert_eq!(interp_membership(&u, &m, -10.0), 0.5);
        assert_eq!(interp_membership(&u, &m, 200.0), 0.8);
    }

    #[test]
    fn trimf_left_shoulder_intermediate() {
        let result = trimf(7.0, 5.0, 5.0, 10.0);
        assert!((result - 0.6).abs() < 1e-10);
    }

    #[test]
    fn trimf_right_shoulder_intermediate() {
        let result = trimf(7.0, 0.0, 10.0, 10.0);
        assert!((result - 0.7).abs() < 1e-10);
    }

    // ─── NEW TESTS TO KILL MEMBERSHIP MUTANTS ──────────────────

    // boundary: x exactly at a, b, c
    #[test]
    fn trimf_boundary_at_a() {
        assert_eq!(trimf(0.0, 0.0, 50.0, 100.0), 0.0);
    }
    #[test]
    fn trimf_boundary_at_c() {
        assert_eq!(trimf(100.0, 0.0, 50.0, 100.0), 0.0);
    }
    #[test]
    fn trimf_boundary_inside_rise() {
        // x between a and b, not symmetric to catch arithmetic changes
        assert!((trimf(20.0, 10.0, 30.0, 90.0) - 0.5).abs() < 1e-10);
    }

    // trapmf: test points exactly at the beginning/end of slopes
    #[test]
    fn trapmf_at_a() {
        // exactly at a (should be 0.0, but if < becomes <= might become something)
        assert_eq!(trapmf(20.0, 20.0, 30.0, 70.0, 80.0), 0.0);
    }
    #[test]
    fn trapmf_at_d() {
        assert_eq!(trapmf(80.0, 20.0, 30.0, 70.0, 80.0), 0.0);
    }
    #[test]
    fn trapmf_just_above_a() {
        // x slightly > a, slope should be linear
        let val = trapmf(22.0, 20.0, 30.0, 70.0, 80.0);
        assert!(val > 0.0 && val < 0.3);
    }
    #[test]
    fn trapmf_just_below_d() {
        let val = trapmf(78.0, 20.0, 30.0, 70.0, 80.0);
        assert!(val > 0.0 && val < 0.3);
    }

    // gaussmf: test non-symmetric sigma to catch * / swaps
    #[test]
    fn gaussmf_non_symmetric() {
        // mean=0, sigma=2, x=1 -> e^{-1/8} approx 0.8825
        let expected = (-(1.0_f64.powi(2)) / (2.0 * 2.0_f64.powi(2))).exp();
        assert!((gaussmf(1.0, 0.0, 2.0) - expected).abs() < 1e-10);
    }

    // interp_membership: test exact interpolation between two points with non-trivial slope
    #[test]
    fn interp_midpoint() {
        let u = vec![0.0, 10.0];
        let m = vec![2.0, 4.0]; // not [0,1] to catch constant replacement
        let result = interp_membership(&u, &m, 5.0);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn interp_exact_grid_point() {
        let u = vec![0.0, 5.0, 10.0];
        let m = vec![1.0, 3.0, 5.0];
        // value exactly at grid point 5.0 should return 3.0
        assert_eq!(interp_membership(&u, &m, 5.0), 3.0);
    }

    #[test]
    fn interp_value_below_first() {
        let u = vec![5.0, 10.0];
        let m = vec![2.0, 4.0];
        // value 0.0 < 5.0 -> returns m[0] = 2.0
        assert_eq!(interp_membership(&u, &m, 0.0), 2.0);
    }

    #[test]
    fn interp_value_above_last() {
        let u = vec![5.0, 10.0];
        let m = vec![2.0, 4.0];
        assert_eq!(interp_membership(&u, &m, 15.0), 4.0);
    }

    /// Kill the mutant that replaces (b - a) with (b + a) in trimf left shoulder branch.
    /// For a=b, the shoulder branch is taken; if the subtraction is replaced by addition,
    /// the condition (a - b).abs() < EPS changes, but (a - b).abs() would become (a + b).abs(),
    /// which is huge, so it wouldn't enter the shoulder, falling through to the standard
    /// triangle code and returning 0.0.
    #[test]
    fn trimf_left_shoulder_exact() {
        // a=2, b=2, c=10, x=1 -> should return 1.0 (shoulder)
        assert_eq!(trimf(1.0, 2.0, 2.0, 10.0), 1.0);
    }

    /// Similar for right shoulder
    #[test]
    fn trimf_right_shoulder_exact() {
        // b=8, c=8, x=9 -> should return 1.0
        assert_eq!(trimf(9.0, 0.0, 8.0, 8.0), 1.0);
    }

    /// For interp_membership, test with value exactly at a grid point to kill
    /// the mutator that changes (x1 - x0) to (x1 + x0).
    #[test]
    fn interp_exact_grid_point_non_zero() {
        let u = vec![1.0, 3.0, 5.0];
        let m = vec![2.0, 4.0, 6.0];
        assert_eq!(interp_membership(&u, &m, 3.0), 4.0);
    }

    /// Test interp with decreasing values to catch mutants that replace (y1 - y0) by (y1 + y0).
    #[test]
    fn interp_decreasing() {
        let u = vec![0.0, 5.0, 10.0];
        let m = vec![10.0, 5.0, 0.0];
        let result = interp_membership(&u, &m, 2.5);
        // linear interpolation between (0,10) and (5,5)
        assert!((result - 7.5).abs() < 1e-9);
    }

    /// trapmf with a nearly zero slope (a very small difference between a and b)
    /// forces the absolute value check to decide between open shoulder and normal ramp.
    #[test]
    fn trapmf_very_small_rise() {
        // a=0, b=1e-15, c=2, d=3, x=0.5e-15 -> b-a = 1e-15 > EPS? No, EPS is 2.2e-16, so 1e-15 > EPS.
        // The condition (b - a).abs() < EPS is false, so it goes to normal ramp formula,
        // and (x - a)/(b - a) = 0.5e-15 / 1e-15 = 0.5.
        let result = trapmf(0.5e-15, 0.0, 1e-15, 2.0, 3.0);
        assert!((result - 0.5).abs() < 1e-9);
    }

    /// Kill mutants that replace - with + or / in trimf rise formula.
    #[test]
    fn trimf_asymmetric_rise() {
        // a=2, b=4, c=6, x=3 -> (3-2)/(4-2)=0.5
        // mutation +: (3+2)/(4+2)=5/6≈0.833 -> would fail
        assert!((trimf(3.0, 2.0, 4.0, 6.0) - 0.5).abs() < 1e-9);
    }

    /// Kill mutants in trimf fall formula.
    #[test]
    fn trimf_asymmetric_fall() {
        // a=2, b=4, c=6, x=5 -> (6-5)/(6-4)=0.5
        // mutation - to + in (c - x): (6+5)=11 -> 11/2=5.5 not 0.5
        assert!((trimf(5.0, 2.0, 4.0, 6.0) - 0.5).abs() < 1e-9);
    }

    /// Kill mutants in trapmf rise formula (x - a) / (b - a).
    #[test]
    fn trapmf_asymmetric_rise() {
        // a=10, b=20, c=30, d=40, x=15 -> (15-10)/(20-10)=0.5
        let result = trapmf(15.0, 10.0, 20.0, 30.0, 40.0);
        assert!((result - 0.5).abs() < 1e-9);
    }

    /// Kill mutants in trapmf fall formula (d - x) / (d - c).
    #[test]
    fn trapmf_asymmetric_fall() {
        // a=10, b=20, c=30, d=40, x=35 -> (40-35)/(40-30)=0.5
        let result = trapmf(35.0, 10.0, 20.0, 30.0, 40.0);
        assert!((result - 0.5).abs() < 1e-9);
    }

    /// Kill mutants in interp_membership formula.
    #[test]
    fn interp_non_uniform() {
        // universe with non-zero start and non-unit slope
        let u = vec![10.0, 20.0, 30.0];
        let m = vec![1.0, 3.0, 5.0];
        // value=15 -> linear interpolation between (10,1) and (20,3)
        let result = interp_membership(&u, &m, 15.0);
        assert!((result - 2.0).abs() < 1e-9);
    }

    /// Left shoulder with a == b and x between b and c — forces arithmetic (c - x) / (c - b)
    /// and kills any mutation that changes '-' or the formula.
    #[test]
    fn trimf_left_shoulder_formula() {
        // a=2, b=2, c=10, x=5 -> (10-5)/(10-2) = 5/8 = 0.625
        let result = trimf(5.0, 2.0, 2.0, 10.0);
        assert!(
            (result - 0.625).abs() < 1e-9,
            "Expected 0.625, got {}",
            result
        );
    }

    /// Right shoulder with b == c and x between a and b — forces arithmetic (x - a) / (b - a)
    #[test]
    fn trimf_right_shoulder_formula() {
        // a=0, b=8, c=8, x=5 -> (5-0)/(8-0) = 5/8 = 0.625
        let result = trimf(5.0, 0.0, 8.0, 8.0);
        assert!(
            (result - 0.625).abs() < 1e-9,
            "Expected 0.625, got {}",
            result
        );
    }

    /// trapmf rise with a != b, b - a not tiny → uses (x - a) / (b - a).
    /// Catches mutations in numerator or denominator.
    #[test]
    fn trapmf_rise_formula() {
        // a=0, b=10, c=20, d=30, x=3 -> (3-0)/(10-0) = 0.3
        let result = trapmf(3.0, 0.0, 10.0, 20.0, 30.0);
        assert!((result - 0.3).abs() < 1e-9, "Expected 0.3, got {}", result);
    }

    /// trapmf fall with c != d → uses (d - x) / (d - c)
    #[test]
    fn trapmf_fall_formula() {
        // a=0, b=10, c=20, d=30, x=25 -> (30-25)/(30-20) = 0.5
        let result = trapmf(25.0, 0.0, 10.0, 20.0, 30.0);
        assert!((result - 0.5).abs() < 1e-9, "Expected 0.5, got {}", result);
    }

    /// Kill <= vs < mutant on the first value boundary.
    #[test]
    fn interp_value_exactly_equal_to_first() {
        let u = vec![5.0, 10.0, 15.0];
        let m = vec![10.0, 20.0, 30.0];
        // value == u[0] -> should return m[0] = 10.0
        let result = interp_membership(&u, &m, 5.0);
        assert!(
            (result - 10.0).abs() < 1e-9,
            "Expected 10.0, got {}",
            result
        );
    }

    /// Kill mutants in interp_membership line 165 by forcing x0 == x1.
    /// The guard (x1 - x0).abs() < EPSILON must be exercised exactly.
    #[test]
    fn interp_duplicate_universe_point_returns_first() {
        // x0 == x1 -> must return y0 without division
        let u = vec![5.0, 5.0, 10.0];
        let m = vec![0.3, 0.7, 1.0];
        // (x1 - x0) = 0.0 < EPSILON -> returns y0 = 0.3
        // Mutant "- → +": (x1 + x0).abs() = 10.0 >> EPSILON -> division by zero -> NaN/panic
        let result = interp_membership(&u, &m, 5.0);
        assert!((result - 0.3).abs() < 1e-9, "Expected 0.3, got {}", result);
    }

    /// Kill arithmetic mutants in interp_membership line 168 formula.
    /// Uses an asymmetric segment where (value - x0) and (x1 - x0) differ.
    #[test]
    fn interp_asymmetric_segment_kills_arithmetic_mutant() {
        // Segment [3.0, 7.0] with memberships [0.0, 1.0]
        // value = 4.0 -> (4-3)/(7-3) = 0.25
        // mutant (value + x0): (4+3)/(7-3) = 1.75 -> > 1.0, fails
        let u = vec![3.0, 7.0];
        let m = vec![0.0, 1.0];
        let result = interp_membership(&u, &m, 4.0);
        assert!(
            (result - 0.25).abs() < 1e-9,
            "Expected 0.25, got {}",
            result
        );
    }
}
