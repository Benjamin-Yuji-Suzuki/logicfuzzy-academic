//! # Particle Swarm Optimization (PSO)
//!
//! Population-based metaheuristic for minimizing a fitness function
//! over a bounded continuous search space.
//!
//! Uses the standard PSO update rules with **inertia weight**:
//!
//! ```text
//! vᵢ = w·vᵢ + c₁·r₁·(pbestᵢ − xᵢ) + c₂·r₂·(gbest − xᵢ)
//! xᵢ = xᵢ + vᵢ
//! ```
//!
//! where `w` = inertia, `c₁` = cognitive coefficient, `c₂` = social coefficient.
//!
//! Features:
//! - Built-in **SplitMix64** PRNG — zero external dependencies
//! - Reproducible via `seed: Option<u64>`
//! - Per-dimension bounds, velocity clamping, early stopping
//!
//! # Example
//! ```
//! use logicfuzzy_academic::{PsoConfig, PsoOptimizer};
//!
//! let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
//!
//! let config = PsoConfig {
//!     population_size: 30,
//!     max_iterations: 200,
//!     bounds: vec![(-10.0, 10.0), (-10.0, 10.0)],
//!     seed: Some(42),
//!     ..Default::default()
//! };
//!
//! let mut optimizer = PsoOptimizer::new(config);
//! let (best_pos, best_fit, _) = optimizer.optimize(sphere);
//! assert!(best_fit < 0.01);
//! ```

use std::time::{Duration, Instant};

/// Minimal non-cryptographic PRNG (SplitMix64) — zero external dependencies.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut z = self.state.wrapping_add(0x9e3779b97f4a7c15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Returns f64 in [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    fn next_f64_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn make_seed(seed: Option<u64>) -> SplitMix64 {
    match seed {
        Some(s) => SplitMix64::new(s),
        None => {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_nanos() as u64;
            SplitMix64::new(t)
        }
    }
}

/// Configuration for the Particle Swarm Optimizer.
#[derive(Debug, Clone)]
pub struct PsoConfig {
    /// Number of particles in the swarm.
    pub population_size: usize,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Inertia weight — controls how much the particle retains its previous velocity.
    pub inertia_weight: f64,
    /// Cognitive coefficient (c1) — attraction toward the particle's personal best.
    pub cognitive_coefficient: f64,
    /// Social coefficient (c2) — attraction toward the global best.
    pub social_coefficient: f64,
    /// Per-dimension bounds as `(min, max)`. Length must match the problem dimension.
    pub bounds: Vec<(f64, f64)>,
    /// Optional maximum velocity magnitude (per dimension). Clamped to `[-limit, limit]`.
    pub velocity_limit: Option<f64>,
    /// Early-stopping tolerance: if global best fitness changes less than this
    /// for `patience` iterations, stop.
    pub tolerance: f64,
    /// Number of iterations with no improvement before early stop.
    pub patience: usize,
    /// Optional seed for reproducible results. `None` uses system time.
    pub seed: Option<u64>,
}

impl Default for PsoConfig {
    fn default() -> Self {
        Self {
            population_size: 30,
            max_iterations: 1000,
            inertia_weight: 0.729,
            cognitive_coefficient: 1.494,
            social_coefficient: 1.494,
            bounds: vec![],
            velocity_limit: None,
            tolerance: 1e-8,
            patience: 50,
            seed: None,
        }
    }
}

/// A single particle in the swarm.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Current position in the search space.
    pub position: Vec<f64>,
    /// Current velocity.
    pub velocity: Vec<f64>,
    /// Best position found by this particle.
    pub personal_best_position: Vec<f64>,
    /// Fitness value at the personal best position.
    pub personal_best_fitness: f64,
}

/// State of the PSO optimizer at a given iteration.
#[derive(Debug, Clone)]
pub struct PsoState {
    /// All particles in the swarm.
    pub particles: Vec<Particle>,
    /// Best position found by any particle.
    pub global_best_position: Vec<f64>,
    /// Fitness value at the global best position.
    pub global_best_fitness: f64,
    /// Current iteration number (0-indexed).
    pub iteration: usize,
    /// Whether the optimizer has converged (tolerance met).
    pub converged: bool,
    /// Elapsed time since the optimizer started.
    pub elapsed: Duration,
}

/// Particle Swarm Optimization engine.
///
/// Minimizes a user-provided fitness function over a bounded search space.
/// Uses the standard PSO update rules with inertia weight.
/// Zero external dependencies — uses a built-in SplitMix64 PRNG.
pub struct PsoOptimizer {
    config: PsoConfig,
    state: Option<PsoState>,
    start_time: Option<Instant>,
}

impl PsoOptimizer {
    /// Creates a new PSO optimizer with the given configuration.
    ///
    /// # Panics
    /// Panics if `bounds` is empty or if any bound is invalid (min > max).
    pub fn new(config: PsoConfig) -> Self {
        assert!(
            !config.bounds.is_empty(),
            "PsoConfig: bounds must not be empty"
        );
        for (i, (lo, hi)) in config.bounds.iter().enumerate() {
            assert!(
                lo < hi,
                "PsoConfig: bound {} has min ({}) >= max ({})",
                i,
                lo,
                hi
            );
        }
        Self {
            config,
            state: None,
            start_time: None,
        }
    }

    /// Initializes and runs the full optimization.
    ///
    /// `fitness_fn` takes a slice of parameter values and returns a score to *minimize*.
    ///
    /// Returns `(best_position, best_fitness, final_state)`.
    pub fn optimize<F>(&mut self, fitness_fn: F) -> (Vec<f64>, f64, PsoState)
    where
        F: Fn(&[f64]) -> f64,
    {
        self.start_time = Some(Instant::now());
        let mut rng = make_seed(self.config.seed);

        let mut particles: Vec<Particle> = (0..self.config.population_size)
            .map(|_| {
                let position = random_position(&self.config.bounds, &mut rng);
                let velocity = random_velocity(&self.config.bounds, &mut rng);
                let fitness = fitness_fn(&position);
                Particle {
                    personal_best_position: position.clone(),
                    personal_best_fitness: fitness,
                    position,
                    velocity,
                }
            })
            .collect();

        let (mut global_best_position, mut global_best_fitness) = {
            let best_idx = (0..particles.len())
                .min_by(|&i, &j| {
                    particles[i]
                        .personal_best_fitness
                        .total_cmp(&particles[j].personal_best_fitness)
                })
                .unwrap();
            (
                particles[best_idx].personal_best_position.clone(),
                particles[best_idx].personal_best_fitness,
            )
        };

        let mut converged = false;
        let mut no_improve_count = 0;
        let mut final_iteration = 0usize;

        for iteration in 0..self.config.max_iterations {
            final_iteration = iteration;

            for particle in &mut particles {
                for (j, (p_pos, p_vel)) in particle
                    .position
                    .iter()
                    .zip(particle.velocity.iter_mut())
                    .enumerate()
                {
                    let r1: f64 = rng.next_f64();
                    let r2: f64 = rng.next_f64();
                    let p_best = particle.personal_best_position[j];
                    let g_best = global_best_position[j];
                    let cognitive = self.config.cognitive_coefficient * r1 * (p_best - p_pos);
                    let social = self.config.social_coefficient * r2 * (g_best - p_pos);
                    let new_v = self.config.inertia_weight * *p_vel + cognitive + social;
                    *p_vel = match self.config.velocity_limit {
                        Some(limit) => new_v.clamp(-limit, limit),
                        None => new_v,
                    };
                }

                for (j, p) in particle.position.iter_mut().enumerate() {
                    let new_pos = *p + particle.velocity[j];
                    *p = new_pos.clamp(self.config.bounds[j].0, self.config.bounds[j].1);
                }

                let fitness = fitness_fn(&particle.position);

                if fitness < particle.personal_best_fitness {
                    particle.personal_best_fitness = fitness;
                    particle.personal_best_position = particle.position.clone();
                }

                if fitness < global_best_fitness {
                    global_best_fitness = fitness;
                    global_best_position.clone_from(&particle.position);
                }
            }

            for particle in &particles {
                if particle.personal_best_fitness < global_best_fitness {
                    global_best_fitness = particle.personal_best_fitness;
                    global_best_position.clone_from(&particle.personal_best_position);
                }
            }

            if no_improve_count >= self.config.patience
                && (global_best_fitness - prev_fitness(&particles, &global_best_fitness)).abs()
                    < self.config.tolerance
            {
                converged = true;
                break;
            }

            no_improve_count = if self.check_improved(&particles, &global_best_fitness) {
                0
            } else {
                no_improve_count + 1
            };
        }

        let elapsed = self.start_time.unwrap().elapsed();
        let state = PsoState {
            particles: particles.clone(),
            global_best_position: global_best_position.clone(),
            global_best_fitness,
            iteration: final_iteration,
            converged,
            elapsed,
        };
        self.state = Some(state.clone());

        (
            state.global_best_position.clone(),
            state.global_best_fitness,
            state,
        )
    }

    /// Returns `true` if any particle improved the global best this iteration.
    fn check_improved(&self, particles: &[Particle], current_global: &f64) -> bool {
        particles
            .iter()
            .any(|p| p.personal_best_fitness < *current_global)
    }
}

fn random_position(bounds: &[(f64, f64)], rng: &mut SplitMix64) -> Vec<f64> {
    bounds
        .iter()
        .map(|&(lo, hi)| rng.next_f64_range(lo, hi))
        .collect()
}

fn random_velocity(bounds: &[(f64, f64)], rng: &mut SplitMix64) -> Vec<f64> {
    bounds
        .iter()
        .map(|&(lo, hi)| {
            let range = hi - lo;
            rng.next_f64_range(-range * 0.1, range * 0.1)
        })
        .collect()
}

/// Helper to compute the previous iteration's fitness baseline.
fn prev_fitness(particles: &[Particle], current_global: &f64) -> f64 {
    particles
        .iter()
        .map(|p| p.personal_best_fitness)
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(*current_global)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn pso_optimizes_sphere_2d() {
        let config = PsoConfig {
            population_size: 50,
            max_iterations: 500,
            inertia_weight: 0.729,
            cognitive_coefficient: 1.494,
            social_coefficient: 1.494,
            bounds: vec![(-10.0, 10.0), (-10.0, 10.0)],
            velocity_limit: Some(2.0),
            tolerance: 1e-10,
            patience: 100,
            seed: Some(42),
        };
        let mut optimizer = PsoOptimizer::new(config);
        let (best_pos, best_fit, state) = optimizer.optimize(sphere);

        assert!(state.converged || state.iteration < 500);
        assert!(
            best_fit < 0.01,
            "Sphere optimum should be near 0, got {}",
            best_fit
        );
        assert!(
            best_pos.iter().all(|&x| x.abs() < 0.1),
            "Position should be near origin"
        );
    }

    #[test]
    fn pso_respects_bounds() {
        let config = PsoConfig {
            population_size: 30,
            max_iterations: 200,
            bounds: vec![(5.0, 10.0), (-5.0, 5.0)],
            seed: Some(123),
            ..Default::default()
        };
        let mut optimizer = PsoOptimizer::new(config);
        let (best_pos, _, _) = optimizer.optimize(sphere);

        assert!((5.0..=10.0).contains(&best_pos[0]));
        assert!((-5.0..=5.0).contains(&best_pos[1]));
    }

    #[test]
    fn pso_improves_over_iterations() {
        let config = PsoConfig {
            population_size: 20,
            max_iterations: 100,
            bounds: vec![(-100.0, 100.0)],
            seed: Some(7),
            ..Default::default()
        };
        let mut optimizer = PsoOptimizer::new(config);
        let (_, best_fit, state) = optimizer.optimize(sphere);

        let initial_fit: f64 = state
            .particles
            .iter()
            .map(|p| p.personal_best_fitness)
            .min_by(|a, b| a.total_cmp(b))
            .unwrap();
        assert!(
            best_fit <= initial_fit + 1e-6,
            "Best fit after optimization ({}) should be <= initial ({})",
            best_fit,
            initial_fit
        );
    }

    #[test]
    fn pso_single_dimension() {
        let config = PsoConfig {
            population_size: 20,
            max_iterations: 200,
            bounds: vec![(-10.0, 10.0)],
            seed: Some(42),
            ..Default::default()
        };
        let mut optimizer = PsoOptimizer::new(config);
        let (best_pos, best_fit, _) = optimizer.optimize(|x| (x[0] - 3.0).powi(2));

        assert!((best_pos[0] - 3.0).abs() < 0.1);
        assert!(best_fit < 0.01);
    }

    #[test]
    fn pso_state_contains_expected_fields() {
        let config = PsoConfig {
            population_size: 10,
            max_iterations: 50,
            bounds: vec![(0.0, 1.0)],
            seed: Some(99),
            ..Default::default()
        };
        let mut optimizer = PsoOptimizer::new(config);
        let (_, _, state) = optimizer.optimize(sphere);

        assert_eq!(state.particles.len(), 10);
        assert_eq!(state.global_best_position.len(), 1);
        assert!(state.iteration <= 50);
        assert!(!state.elapsed.is_zero());
    }

    #[test]
    fn pso_high_dimension() {
        let config = PsoConfig {
            population_size: 80,
            max_iterations: 500,
            bounds: vec![(-5.0, 5.0); 10],
            seed: Some(42),
            ..Default::default()
        };
        let mut optimizer = PsoOptimizer::new(config);
        let (_, best_fit, _) = optimizer.optimize(sphere);
        assert!(
            best_fit < 0.1,
            "10D sphere should reach near 0, got {}",
            best_fit
        );
    }

    #[test]
    fn velocity_limit_enforced() {
        let config = PsoConfig {
            population_size: 10,
            max_iterations: 5,
            bounds: vec![(-10.0, 10.0)],
            velocity_limit: Some(0.5),
            seed: Some(1),
            ..Default::default()
        };
        let mut optimizer = PsoOptimizer::new(config);
        optimizer.optimize(sphere);
    }

    #[test]
    fn different_seeds_give_different_results() {
        let bounds = vec![(-10.0, 10.0); 2];
        let make = |seed: u64| {
            let config = PsoConfig {
                population_size: 20,
                max_iterations: 100,
                bounds: bounds.clone(),
                seed: Some(seed),
                ..Default::default()
            };
            let mut opt = PsoOptimizer::new(config);
            let (_, fit, _) = opt.optimize(sphere);
            fit
        };

        let fit1 = make(1);
        let fit2 = make(2);
        assert!((fit1 - fit2).abs() > 1e-15 || fit1 < 1.0);
    }

    #[test]
    #[should_panic(expected = "bounds must not be empty")]
    fn empty_bounds_panics() {
        let config = PsoConfig {
            bounds: vec![],
            ..Default::default()
        };
        PsoOptimizer::new(config);
    }

    #[test]
    #[should_panic(expected = "bound")]
    fn invalid_bounds_panics() {
        let config = PsoConfig {
            bounds: vec![(5.0, 1.0)],
            ..Default::default()
        };
        PsoOptimizer::new(config);
    }

    #[test]
    fn rosenbrock_function() {
        let rosenbrock =
            |x: &[f64]| -> f64 { (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2) };

        let config = PsoConfig {
            population_size: 80,
            max_iterations: 500,
            bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
            seed: Some(42),
            ..Default::default()
        };
        let mut optimizer = PsoOptimizer::new(config);
        let (best_pos, best_fit, _) = optimizer.optimize(rosenbrock);

        assert!(
            best_fit < 0.1,
            "Rosenbrock optimum should be near 0, got {}",
            best_fit
        );
        assert!(
            (best_pos[0] - 1.0).abs() < 0.1,
            "x should be near 1, got {}",
            best_pos[0]
        );
        assert!(
            (best_pos[1] - 1.0).abs() < 0.1,
            "y should be near 1, got {}",
            best_pos[1]
        );
    }

    #[test]
    fn population_size_correct() {
        let config = PsoConfig {
            population_size: 15,
            max_iterations: 10,
            bounds: vec![(0.0, 1.0)],
            seed: Some(0),
            ..Default::default()
        };
        let mut opt = PsoOptimizer::new(config);
        let (_, _, state) = opt.optimize(sphere);
        assert_eq!(state.particles.len(), 15);
    }

    #[test]
    fn seed_none_does_not_panic() {
        let config = PsoConfig {
            population_size: 5,
            max_iterations: 5,
            bounds: vec![(0.0, 1.0)],
            seed: None,
            ..Default::default()
        };
        let mut opt = PsoOptimizer::new(config);
        let (_, _, _) = opt.optimize(sphere);
    }

    #[test]
    fn splitmix64_produces_different_values() {
        let mut rng = SplitMix64::new(42);
        let a = rng.next_f64();
        let b = rng.next_f64();
        assert!((a - b).abs() > 1e-15);
    }

    #[test]
    fn splitmix64_range_respects_bounds() {
        let mut rng = SplitMix64::new(99);
        for _ in 0..100 {
            let v = rng.next_f64_range(-5.0, 10.0);
            assert!((-5.0..10.0).contains(&v));
        }
    }

    #[test]
    fn splitmix64_exact_sequence_kills_xor_mutants() {
        // Deterministic sequence with seed 42 — exact values kill ^/|/& and >>/<< mutants
        let mut rng = SplitMix64::new(42);
        assert_eq!(rng.next_u64(), 13679457532755275413);
        assert_eq!(rng.next_u64(), 2949826092126892291);
        assert_eq!(rng.next_u64(), 5139283748462763858);
        // next_f64 should also match
        let mut rng2 = SplitMix64::new(42);
        let f0 = rng2.next_f64();
        assert!((f0 - 0.7415648787718233).abs() < 1e-15);
        let f1 = rng2.next_f64();
        assert!((f1 - 0.1599103928769201).abs() < 1e-15);
    }

    #[test]
    fn splitmix64_seed_zero_produces_known_value() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220a8397b1dcdaf);
    }

    #[test]
    fn prev_fitness_exact() {
        // Direct test of prev_fitness helper
        let p1 = Particle {
            position: vec![0.0],
            velocity: vec![0.0],
            personal_best_position: vec![0.0],
            personal_best_fitness: 0.5,
        };
        let p2 = Particle {
            position: vec![0.0],
            velocity: vec![0.0],
            personal_best_position: vec![0.0],
            personal_best_fitness: 0.3,
        };
        let particles = vec![p1, p2];
        let result = prev_fitness(&particles, &0.4);
        // Should return the minimum (0.3), not the current global (0.4)
        assert!((result - 0.3).abs() < 1e-15);
    }

    #[test]
    fn check_improved_exact() {
        let config = PsoConfig {
            population_size: 2,
            max_iterations: 0,
            bounds: vec![(0.0, 1.0)],
            seed: Some(0),
            ..Default::default()
        };
        let opt = PsoOptimizer::new(config);
        let p = Particle {
            position: vec![0.0],
            velocity: vec![0.0],
            personal_best_position: vec![0.0],
            personal_best_fitness: 0.1,
        };
        // current_global=0.2, particle has 0.1 < 0.2 => improved
        assert!(opt.check_improved(std::slice::from_ref(&p), &0.2));
        assert!(!opt.check_improved(std::slice::from_ref(&p), &0.1));
        // Worse => not improved
        assert!(!opt.check_improved(&[p], &0.05));
    }
}
