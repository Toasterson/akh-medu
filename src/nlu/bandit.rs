//! LinUCB contextual bandit for NLU backend selection.
//!
//! Each backend gets one arm. The bandit selects the arm with the highest
//! upper confidence bound given the current query feature vector:
//!
//! `score = θᵀx + α√(xᵀA⁻¹x)`
//!
//! After observing a reward, the arm's A matrix and b vector are updated
//! and θ = A⁻¹b is recomputed.
//!
//! Feature dimension is fixed at [`D`] = 8 so all matrix operations use
//! plain stack-allocated arrays for vectors and heap-allocated `Vec` for
//! D×D matrices (64 > serde's built-in 32-element array limit).

use serde::{Deserialize, Serialize};

use super::router::BackendId;

/// Dimension of the query feature vector.
pub const D: usize = 8;

/// A fixed-size feature vector extracted from a query.
pub type FeatureVec = [f32; D];

/// One arm of the LinUCB bandit, corresponding to a single backend.
///
/// D-vectors (`b`, `theta`) are stack-allocated; D×D matrices (`a`, `a_inv`)
/// use `Vec<f32>` to remain compatible with serde (fixed arrays > 32 are not
/// supported by serde's default derive).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinUcbArm {
    /// Which backend this arm selects.
    pub backend: BackendId,
    /// A: d×d context correlation matrix (row-major). Starts as identity.
    a: Vec<f32>,
    /// b: d×1 reward correlation vector.
    b: [f32; D],
    /// Cached θ = A⁻¹b (recomputed after each update).
    theta: [f32; D],
    /// Cached A⁻¹ (row-major, recomputed after each update).
    a_inv: Vec<f32>,
    /// Total number of times this arm has been selected.
    pub pulls: u64,
}

impl LinUcbArm {
    /// Create a new arm for `backend`.
    ///
    /// A is initialised to the identity, giving A⁻¹ = I and θ = 0,
    /// an uninformed uniform prior.
    pub fn new(backend: BackendId) -> Self {
        let mut a = vec![0.0f32; D * D];
        let mut a_inv = vec![0.0f32; D * D];
        for i in 0..D {
            a[i * D + i] = 1.0;
            a_inv[i * D + i] = 1.0;
        }
        Self {
            backend,
            a,
            b: [0.0; D],
            theta: [0.0; D],
            a_inv,
            pulls: 0,
        }
    }

    /// Compute the UCB score for this arm given feature vector `x` and
    /// exploration parameter `alpha`.
    ///
    /// `score = θᵀx + alpha * √(xᵀA⁻¹x)`
    pub fn score(&self, x: &FeatureVec, alpha: f32) -> f32 {
        let exploit = dot(&self.theta, x);
        let explore = alpha * quad_form(&self.a_inv, x).max(0.0).sqrt();
        exploit + explore
    }

    /// Update this arm after observing reward `reward` for feature vector `x`.
    ///
    /// Updates: A += xxᵀ, b += r·x, then recomputes A⁻¹ and θ.
    pub fn update(&mut self, x: &FeatureVec, reward: f32) {
        self.pulls += 1;
        for i in 0..D {
            for j in 0..D {
                self.a[i * D + j] += x[i] * x[j];
            }
        }
        for i in 0..D {
            self.b[i] += reward * x[i];
        }
        if let Some(inv) = invert_matrix(&self.a) {
            self.a_inv = inv;
            self.theta = mat_vec_mul(&self.a_inv, &self.b);
        }
    }
}

/// The LinUCB bandit: a collection of arms, one per routable backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinUcbBandit {
    /// All arms (one per backend, in order supplied at construction).
    pub arms: Vec<LinUcbArm>,
    /// Exploration parameter α (higher = more exploration, range [0,2]).
    pub alpha: f32,
}

impl LinUcbBandit {
    /// Create a bandit with one arm per backend in `backends`.
    pub fn new(backends: &[BackendId], alpha: f32) -> Self {
        Self {
            arms: backends.iter().map(|&b| LinUcbArm::new(b)).collect(),
            alpha,
        }
    }

    /// Select the arm with the highest UCB score.
    ///
    /// Returns `(arm_index, backend_id)`. Falls back to `(0, RuleParser)`
    /// if the arm list is somehow empty.
    pub fn select(&self, x: &FeatureVec) -> (usize, BackendId) {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.score(x, self.alpha)
                    .partial_cmp(&b.score(x, self.alpha))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, arm)| (i, arm.backend))
            .unwrap_or((0, BackendId::RuleParser))
    }

    /// Report a reward for the arm at `arm_idx` given feature vector `x`.
    pub fn update(&mut self, arm_idx: usize, x: &FeatureVec, reward: f32) {
        if let Some(arm) = self.arms.get_mut(arm_idx) {
            arm.update(x, reward);
        }
    }

    /// Increase the exploration rate, capped at 2.0.
    ///
    /// Called when model weights are hot-swapped (Phase 27 integration hook).
    pub fn bump_exploration(&mut self, delta: f32) {
        self.alpha = (self.alpha + delta).min(2.0);
    }

    /// Gradually decay exploration back toward `base`, stepping by `decay`.
    pub fn decay_exploration(&mut self, base: f32, decay: f32) {
        if self.alpha > base {
            self.alpha = (self.alpha - decay).max(base);
        }
    }

    /// Serialize bandit state to bytes for persistence (redb / file).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize bandit state from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

// ── Linear algebra helpers (D×D matrices as flat row-major Vec<f32>) ────────

fn dot(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Multiply a D×D matrix by a D-vector: y = M·x.
fn mat_vec_mul(m: &[f32], x: &[f32; D]) -> [f32; D] {
    let mut y = [0.0f32; D];
    for i in 0..D {
        for j in 0..D {
            y[i] += m[i * D + j] * x[j];
        }
    }
    y
}

/// Quadratic form: xᵀMx.
fn quad_form(m: &[f32], x: &[f32; D]) -> f32 {
    let mx = mat_vec_mul(m, x);
    dot(x, &mx)
}

/// Invert a D×D matrix using Gauss-Jordan elimination with partial pivoting.
///
/// Returns `None` if the matrix is (near-) singular.
fn invert_matrix(m: &[f32]) -> Option<Vec<f32>> {
    // Build augmented matrix [M | I]
    let mut aug = [[0.0f32; D * 2]; D];
    for i in 0..D {
        for j in 0..D {
            aug[i][j] = m[i * D + j];
        }
        aug[i][D + i] = 1.0;
    }

    for col in 0..D {
        // Partial pivot: find row with largest absolute value in this column
        let (mut max_row, mut max_val) = (col, aug[col][col].abs());
        for row in (col + 1)..D {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-10 {
            return None; // singular
        }
        aug.swap(col, max_row);

        // Scale pivot row
        let pivot = aug[col][col];
        for j in 0..(D * 2) {
            aug[col][j] /= pivot;
        }

        // Eliminate column entries in all other rows
        for row in 0..D {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(D * 2) {
                let sub = factor * aug[col][j];
                aug[row][j] -= sub;
            }
        }
    }

    let mut result = vec![0.0f32; D * D];
    for i in 0..D {
        for j in 0..D {
            result[i * D + j] = aug[i][D + j];
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Vec<f32> {
        let mut m = vec![0.0f32; D * D];
        for i in 0..D {
            m[i * D + i] = 1.0;
        }
        m
    }

    #[test]
    fn identity_inversion() {
        let inv = invert_matrix(&identity()).expect("identity is invertible");
        for i in 0..D {
            for j in 0..D {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i * D + j] - expected).abs() < 1e-5,
                    "inv[{i}][{j}] = {} ≠ {expected}",
                    inv[i * D + j]
                );
            }
        }
    }

    #[test]
    fn arm_initial_score() {
        let arm = LinUcbArm::new(BackendId::RuleParser);
        // With identity A, theta=0, A_inv=I:
        // score = 0 + alpha * sqrt(x^T I x) = alpha * ||x||
        let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let s = arm.score(&x, 0.25);
        assert!((s - 0.25).abs() < 1e-5, "expected 0.25 got {s}");
    }

    #[test]
    fn arm_learns_from_reward() {
        let mut arm = LinUcbArm::new(BackendId::RuleParser);
        let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let score_before = arm.score(&x, 0.25);
        arm.update(&x, 1.0);
        let score_after = arm.score(&x, 0.25);

        // Positive reward should increase the exploitation term
        assert!(
            score_after > score_before,
            "score should increase after positive reward: before={score_before} after={score_after}"
        );
    }

    #[test]
    fn bandit_selects_valid_arm() {
        let bandit = LinUcbBandit::new(
            &[
                BackendId::RuleParser,
                BackendId::CandleLlm,
                BackendId::VsaDirect,
            ],
            0.25,
        );
        let x = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5];
        let (idx, _backend) = bandit.select(&x);
        assert!(idx < 3, "arm index {idx} out of range");
    }

    #[test]
    fn bandit_converges_on_rewarded_arm() {
        let mut bandit = LinUcbBandit::new(
            &[BackendId::RuleParser, BackendId::CandleLlm],
            0.1, // low alpha → more exploitation
        );
        let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Reward arm 0 repeatedly with the same feature vector
        for _ in 0..20 {
            bandit.update(0, &x, 1.0);
            bandit.update(1, &x, 0.0);
        }

        let (idx, _) = bandit.select(&x);
        assert_eq!(idx, 0, "bandit should prefer the rewarded arm");
    }

    #[test]
    fn bump_exploration_clamped() {
        let mut bandit = LinUcbBandit::new(&[BackendId::RuleParser], 0.25);
        bandit.bump_exploration(5.0);
        assert_eq!(bandit.alpha, 2.0, "alpha should be clamped at 2.0");
    }

    #[test]
    fn serialization_roundtrip() {
        let mut bandit = LinUcbBandit::new(&[BackendId::RuleParser, BackendId::CandleLlm], 0.25);
        let x = [0.5; D];
        bandit.update(0, &x, 0.8);

        let bytes = bandit.to_bytes();
        let restored = LinUcbBandit::from_bytes(&bytes).expect("should deserialize");
        assert_eq!(restored.arms.len(), 2);
        assert!((restored.alpha - 0.25).abs() < 1e-6);
    }
}
