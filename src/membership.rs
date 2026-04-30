//! # membership.rs
//!
//! Membership functions for Mamdani fuzzy logic.
//! Equivalent to the skfuzzy module from scikit-fuzzy.

// ─────────────────────────────────────────────
// Funcoes puras de pertinencia
// ─────────────────────────────────────────────

/// Triangular membership function (trimf).
/// Parameters: `a` (start), `b` (peak), `c` (end). Requires: `a <= b <= c`.
/// Equivalent to: `skfuzzy.trimf(universe, [a, b, c])`
pub fn trimf(x: f64, a: f64, b: f64, c: f64) -> f64 {
    assert!(a <= b && b <= c, "trimf: requires a <= b <= c");

    // Ombro esquerdo: a == b (plato aberto a esquerda)
    // Equivalente ao inicio de um trapmf com a==b
    if (a - b).abs() < f64::EPSILON {
        if x <= b {
            return 1.0;
        }
        if x >= c {
            return 0.0;
        }
        return (c - x) / (c - b);
    }

    // Ombro direito: b == c (plato aberto a direita)
    // Equivalente ao fim de um trapmf com c==d
    if (b - c).abs() < f64::EPSILON {
        if x >= b {
            return 1.0;
        }
        if x <= a {
            return 0.0;
        }
        return (x - a) / (b - a);
    }

    // Triangulo padrao: a < b < c
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
/// Open left shoulder: use `a == b`. Open right shoulder: use `c == d`.
/// Equivalent to: `skfuzzy.trapmf(universe, [a, b, c, d])`
pub fn trapmf(x: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
    assert!(
        a <= b && b <= c && c <= d,
        "trapmf: requires a <= b <= c <= d"
    );

    // Plano central: grau maximo
    if x >= b && x <= c {
        return 1.0;
    }

    // Rampa de subida: [a, b)
    if x >= a && x < b {
        if (b - a).abs() < f64::EPSILON {
            return 1.0;
        }
        return (x - a) / (b - a);
    }

    // Rampa de descida: (c, d]
    if x > c && x <= d {
        if (d - c).abs() < f64::EPSILON {
            return 1.0;
        }
        return (d - x) / (d - c);
    }

    0.0
}

/// Gaussian membership function (gaussmf).
/// Parameters: `mean` (center/peak), `sigma` (width, must be > 0).
/// Has no abrupt edges — smoothly converges to 0 away from the center.
pub fn gaussmf(x: f64, mean: f64, sigma: f64) -> f64 {
    assert!(sigma > 0.0, "gaussmf: sigma must be > 0");
    let exp = -((x - mean).powi(2)) / (2.0 * sigma.powi(2));
    exp.exp()
}

// ─────────────────────────────────────────────
// Enum principal: MembershipFn
// ─────────────────────────────────────────────

/// Enum representing a fuzzy membership function.
/// Used internally by `Term` to store and evaluate any supported MF type.
///
/// # Example
/// ```
/// use logicfuzzy_academic::MembershipFn;
/// let mf = MembershipFn::Trimf([20.0, 50.0, 80.0]);
/// assert_eq!(mf.eval(50.0), 1.0);
/// assert!((mf.eval(35.0) - 0.5).abs() < 1e-10);
/// ```
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
    /// Equivalent to what scikit-fuzzy does with numpy over the universe.
    pub fn eval_universe(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|&x| self.eval(x)).collect()
    }
}

// ─────────────────────────────────────────────
// Utilitario: interpolar grau em universo discreto
// ─────────────────────────────────────────────

/// Interpolates the membership degree of `value` given a discrete universe
/// and its pre-computed membership vector.
/// Equivalent to `skfuzzy.interp_membership(universe, mf_values, value)`.
/// Uses linear interpolation between the two closest points.
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
// Testes unitarios
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // trimf
    #[test]
    fn trimf_pico() {
        assert_eq!(trimf(50.0, 0.0, 50.0, 100.0), 1.0);
    }
    #[test]
    fn trimf_fora_esq() {
        assert_eq!(trimf(-1.0, 0.0, 50.0, 100.0), 0.0);
    }
    #[test]
    fn trimf_fora_dir() {
        assert_eq!(trimf(101.0, 0.0, 50.0, 100.0), 0.0);
    }
    #[test]
    fn trimf_meio_subida() {
        assert!((trimf(25.0, 0.0, 50.0, 100.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn trimf_meio_descida() {
        assert!((trimf(75.0, 0.0, 50.0, 100.0) - 0.5).abs() < 1e-10);
    }

    // trapmf
    #[test]
    fn trapmf_plano() {
        assert_eq!(trapmf(50.0, 20.0, 30.0, 70.0, 80.0), 1.0);
    }
    #[test]
    fn trapmf_borda_b() {
        assert_eq!(trapmf(30.0, 20.0, 30.0, 70.0, 80.0), 1.0);
    }
    #[test]
    fn trapmf_borda_c() {
        assert_eq!(trapmf(70.0, 20.0, 30.0, 70.0, 80.0), 1.0);
    }
    #[test]
    fn trapmf_fora_esq() {
        assert_eq!(trapmf(10.0, 20.0, 30.0, 70.0, 80.0), 0.0);
    }
    #[test]
    fn trapmf_fora_dir() {
        assert_eq!(trapmf(90.0, 20.0, 30.0, 70.0, 80.0), 0.0);
    }
    #[test]
    fn trapmf_aberta_esq() {
        assert_eq!(trapmf(0.0, 0.0, 0.0, 30.0, 50.0), 1.0);
    }
    #[test]
    fn trapmf_aberta_dir() {
        assert_eq!(trapmf(100.0, 60.0, 80.0, 100.0, 100.0), 1.0);
    }

    // gaussmf
    #[test]
    fn gaussmf_pico() {
        assert!((gaussmf(50.0, 50.0, 10.0) - 1.0).abs() < 1e-10);
    }
    #[test]
    fn gaussmf_simetria() {
        assert!((gaussmf(40.0, 50.0, 10.0) - gaussmf(60.0, 50.0, 10.0)).abs() < 1e-10);
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
    fn interp_exato() {
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
    fn interp_fora() {
        let u = vec![0.0, 100.0];
        let m = vec![0.5, 0.8];
        assert_eq!(interp_membership(&u, &m, -10.0), 0.5);
        assert_eq!(interp_membership(&u, &m, 200.0), 0.8);
    }
}
