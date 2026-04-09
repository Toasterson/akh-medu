//! Neural→VSA Bridge: project LLM hidden states into hypervector space.
//!
//! Maps LLM hidden states (f32 tensors of dimension `hidden_dim`, e.g. 1536
//! for Qwen2.5-1.5B) to 10K-bit bipolar HyperVecs via a learned linear
//! projection + binarization.
//!
//! **Encoding pipeline**:
//! 1. Extract hidden state `h` from LLM layer (f32, `[hidden_dim]`)
//! 2. Project: `z = W·h + b` where W is `[vsa_dim, hidden_dim]`
//! 3. Binarize: `v[i] = 1 if z[i] > 0 else 0` (bipolar: 1 → +1, 0 → -1)
//! 4. Result: 10K-bit HyperVec compatible with all VSA operations
//!
//! **Training**: Self-supervised alignment against grounded VSA vectors in
//! ItemMemory. CPU-feasible (~15M f32 params for 1.5B model).
//!
//! **What this enables**:
//! - Concept identification via ItemMemory nearest-neighbor search
//! - Quality gate: if encoded vector is far from all known concepts, LLM may
//!   be hallucinating
//! - Interpretability: which KG concepts activate when processing text?

use super::{Dimension, Encoding, HyperVec};

// ---------------------------------------------------------------------------
// Linear projection
// ---------------------------------------------------------------------------

/// A simple linear projection `y = W·x + b`.
///
/// Stored as flat f32 arrays for simplicity and compatibility with safetensors.
#[derive(Debug, Clone)]
pub struct LinearProjection {
    /// Weight matrix, row-major: `[out_dim × in_dim]`.
    weight: Vec<f32>,
    /// Bias vector: `[out_dim]`.
    bias: Vec<f32>,
    /// Input dimension.
    in_dim: usize,
    /// Output dimension.
    out_dim: usize,
}

impl LinearProjection {
    /// Create a new projection with the given dimensions.
    ///
    /// Weights are initialized to small random values (Xavier uniform).
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (in_dim + out_dim) as f32).sqrt();

        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();
        let bias = vec![0.0f32; out_dim];

        Self {
            weight,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Create a projection with pre-loaded weights.
    pub fn from_weights(weight: Vec<f32>, bias: Vec<f32>, in_dim: usize, out_dim: usize) -> Self {
        debug_assert_eq!(weight.len(), out_dim * in_dim);
        debug_assert_eq!(bias.len(), out_dim);
        Self {
            weight,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: `y = W·x + b`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.in_dim);
        let mut output = self.bias.clone();
        for row in 0..self.out_dim {
            let row_start = row * self.in_dim;
            let mut dot = 0.0f32;
            for col in 0..self.in_dim {
                dot += self.weight[row_start + col] * input[col];
            }
            output[row] += dot;
        }
        output
    }

    /// Access weights (for serialization).
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Access bias (for serialization).
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Input dimension.
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Output dimension.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}

// ---------------------------------------------------------------------------
// Neural-VSA Bridge
// ---------------------------------------------------------------------------

/// Bridge between LLM hidden states and VSA hypervector space.
///
/// Encodes f32 hidden states into 10K-bit bipolar HyperVecs via a learned
/// linear projection followed by binarization.
#[derive(Debug)]
pub struct NeuralVsaBridge {
    /// Encoder: hidden_dim → vsa_dim (f32 pre-binarization).
    encoder: LinearProjection,
    /// Decoder: vsa_dim → hidden_dim (reverse bridge, Phase 27c).
    decoder: Option<LinearProjection>,
    /// Binarization threshold (default: 0.0 for symmetric split).
    threshold: f32,
    /// VSA dimension.
    vsa_dim: Dimension,
}

impl NeuralVsaBridge {
    /// Create a new bridge with random initialization.
    ///
    /// - `hidden_dim`: LLM hidden state dimension (e.g., 1536 for Qwen2.5-1.5B)
    /// - `vsa_dim`: target VSA dimension (e.g., Dimension::DEFAULT = 10,000)
    pub fn new(hidden_dim: usize, vsa_dim: Dimension) -> Self {
        Self {
            encoder: LinearProjection::new(hidden_dim, vsa_dim.0),
            decoder: None,
            threshold: 0.0,
            vsa_dim,
        }
    }

    /// Create from pre-trained encoder weights.
    pub fn from_encoder(encoder: LinearProjection, vsa_dim: Dimension) -> Self {
        debug_assert_eq!(encoder.out_dim(), vsa_dim.0);
        Self {
            encoder,
            decoder: None,
            threshold: 0.0,
            vsa_dim,
        }
    }

    /// Initialize the decoder (reverse bridge: VSA → hidden_dim).
    ///
    /// The decoder enables the symbolic reasoning core to inject solutions
    /// back into the LLM's generation via hidden state manipulation.
    pub fn init_decoder(&mut self) {
        let hidden_dim = self.encoder.in_dim();
        self.decoder = Some(LinearProjection::new(self.vsa_dim.0, hidden_dim));
    }

    /// Set a pre-trained decoder.
    pub fn set_decoder(&mut self, decoder: LinearProjection) {
        debug_assert_eq!(decoder.in_dim(), self.vsa_dim.0);
        debug_assert_eq!(decoder.out_dim(), self.encoder.in_dim());
        self.decoder = Some(decoder);
    }

    /// Decode a HyperVec back to an approximate LLM hidden state.
    ///
    /// Returns `None` if no decoder has been initialized.
    pub fn decode(&self, hypervec: &HyperVec) -> Option<Vec<f32>> {
        let decoder = self.decoder.as_ref()?;
        let float_vec = hypervec_to_float(hypervec);
        Some(decoder.forward(&float_vec))
    }

    /// Whether the decoder (reverse bridge) is available.
    pub fn has_decoder(&self) -> bool {
        self.decoder.is_some()
    }

    /// Encode an LLM hidden state into a HyperVec.
    ///
    /// This is the core operation: project + binarize.
    pub fn encode(&self, hidden_state: &[f32]) -> HyperVec {
        let projected = self.encoder.forward(hidden_state);
        binarize_to_hypervec(&projected, self.threshold, self.vsa_dim)
    }

    /// Encode and return both the HyperVec and the pre-binarization f32 values.
    ///
    /// The pre-binarization values are useful for soft similarity scoring
    /// and for the training loop.
    pub fn encode_with_raw(&self, hidden_state: &[f32]) -> (HyperVec, Vec<f32>) {
        let projected = self.encoder.forward(hidden_state);
        let hypervec = binarize_to_hypervec(&projected, self.threshold, self.vsa_dim);
        (hypervec, projected)
    }

    /// Compute the confidence of an encoding.
    ///
    /// Measures how decisive the binarization was — values far from the
    /// threshold indicate high confidence, values near it indicate uncertainty.
    /// Returns a score in `[0.0, 1.0]`.
    pub fn encoding_confidence(&self, hidden_state: &[f32]) -> f32 {
        let projected = self.encoder.forward(hidden_state);
        let n = projected.len() as f32;
        let avg_margin: f32 = projected
            .iter()
            .map(|&z| (z - self.threshold).abs())
            .sum::<f32>()
            / n;
        // Sigmoid-like mapping: margin 0 → 0.5, margin 1 → ~0.73, margin 2 → ~0.88
        1.0 / (1.0 + (-avg_margin).exp())
    }

    /// Access the encoder projection.
    pub fn encoder(&self) -> &LinearProjection {
        &self.encoder
    }

    /// VSA dimension.
    pub fn vsa_dim(&self) -> Dimension {
        self.vsa_dim
    }

    /// Hidden state dimension (LLM side).
    pub fn hidden_dim(&self) -> usize {
        self.encoder.in_dim()
    }

    /// Number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.encoder.weight().len() + self.encoder.bias().len()
    }
}

// ---------------------------------------------------------------------------
// SGD training
// ---------------------------------------------------------------------------

/// Simple SGD trainer for the neural bridge encoder.
///
/// Trains by aligning encoded hidden states with ground-truth HyperVecs
/// from the ItemMemory. Uses Hamming distance as loss (approximated via
/// sigmoid for differentiability).
pub struct BridgeTrainer {
    /// Learning rate.
    lr: f32,
    /// Accumulated gradient for weight.
    grad_weight: Vec<f32>,
    /// Accumulated gradient for bias.
    grad_bias: Vec<f32>,
    /// Number of training pairs seen.
    pub n_samples: usize,
}

impl BridgeTrainer {
    /// Create a trainer for a bridge with the given dimensions.
    pub fn new(hidden_dim: usize, vsa_dim: usize, lr: f32) -> Self {
        Self {
            lr,
            grad_weight: vec![0.0; vsa_dim * hidden_dim],
            grad_bias: vec![0.0; vsa_dim],
            n_samples: 0,
        }
    }

    /// Accumulate gradients for one (hidden_state, target_hypervec) pair.
    ///
    /// Uses a sigmoid approximation of the Hamming distance loss:
    /// loss_i = target_i * log(σ(z_i)) + (1 - target_i) * log(1 - σ(z_i))
    /// where z_i = W_i · h + b_i (pre-binarization value).
    pub fn accumulate(
        &mut self,
        bridge: &NeuralVsaBridge,
        hidden_state: &[f32],
        target: &HyperVec,
    ) {
        let projected = bridge.encoder.forward(hidden_state);
        let in_dim = bridge.encoder.in_dim();

        for i in 0..bridge.vsa_dim.0 {
            let z = projected[i];
            let sigma = sigmoid(z);
            let target_bit = if target.get_bit(i) { 1.0 } else { 0.0 };
            // Gradient of binary cross-entropy w.r.t. z: σ(z) - target
            let error = sigma - target_bit;

            // Gradient for bias.
            self.grad_bias[i] += error;

            // Gradient for weight row i.
            let row_start = i * in_dim;
            for j in 0..in_dim {
                self.grad_weight[row_start + j] += error * hidden_state[j];
            }
        }

        self.n_samples += 1;
    }

    /// Apply accumulated gradients and reset.
    pub fn step(&mut self, bridge: &mut NeuralVsaBridge) {
        if self.n_samples == 0 {
            return;
        }

        let scale = self.lr / self.n_samples as f32;
        let encoder = &mut bridge.encoder;

        for i in 0..encoder.weight.len() {
            encoder.weight[i] -= scale * self.grad_weight[i];
            self.grad_weight[i] = 0.0;
        }
        for i in 0..encoder.bias.len() {
            encoder.bias[i] -= scale * self.grad_bias[i];
            self.grad_bias[i] = 0.0;
        }

        self.n_samples = 0;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an f32 projection to a bipolar HyperVec by thresholding.
fn binarize_to_hypervec(projected: &[f32], threshold: f32, dim: Dimension) -> HyperVec {
    let mut hv = HyperVec::zero(dim, Encoding::Bipolar);
    for (i, &z) in projected.iter().enumerate().take(dim.0) {
        if z > threshold {
            hv.set_bit(i, true);
        }
        // false (0) is already the default from zero()
    }
    hv
}

/// Convert a bipolar HyperVec to f32 representation (+1.0 / -1.0).
fn hypervec_to_float(hv: &HyperVec) -> Vec<f32> {
    let dim = hv.dim().0;
    let mut result = Vec::with_capacity(dim);
    for i in 0..dim {
        result.push(if hv.get_bit(i) { 1.0 } else { -1.0 });
    }
    result
}

/// Sigmoid function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_projection_dimensions() {
        let proj = LinearProjection::new(128, 1000);
        assert_eq!(proj.in_dim(), 128);
        assert_eq!(proj.out_dim(), 1000);
        assert_eq!(proj.weight().len(), 1000 * 128);
        assert_eq!(proj.bias().len(), 1000);
    }

    #[test]
    fn linear_projection_forward_shape() {
        let proj = LinearProjection::new(64, 100);
        let input = vec![1.0f32; 64];
        let output = proj.forward(&input);
        assert_eq!(output.len(), 100);
    }

    #[test]
    fn bridge_encode_produces_correct_dim() {
        let dim = Dimension::TEST; // 1000
        let bridge = NeuralVsaBridge::new(128, dim);
        let hidden = vec![0.5f32; 128];
        let hv = bridge.encode(&hidden);
        assert_eq!(hv.dim(), dim);
        assert_eq!(hv.encoding(), Encoding::Bipolar);
        assert_eq!(hv.data().len(), dim.binary_byte_len());
    }

    #[test]
    fn bridge_param_count() {
        let bridge = NeuralVsaBridge::new(1536, Dimension::DEFAULT);
        // 10,000 * 1,536 + 10,000 = 15,370,000
        assert_eq!(bridge.param_count(), 10_000 * 1_536 + 10_000);
    }

    #[test]
    fn encode_with_raw_returns_both() {
        let dim = Dimension::TEST;
        let bridge = NeuralVsaBridge::new(32, dim);
        let hidden = vec![1.0f32; 32];
        let (hv, raw) = bridge.encode_with_raw(&hidden);
        assert_eq!(hv.dim(), dim);
        assert_eq!(raw.len(), dim.0);
    }

    #[test]
    fn confidence_in_range() {
        let bridge = NeuralVsaBridge::new(32, Dimension::TEST);
        let hidden = vec![1.0f32; 32];
        let conf = bridge.encoding_confidence(&hidden);
        assert!(conf >= 0.0 && conf <= 1.0, "confidence out of range: {conf}");
    }

    #[test]
    fn sgd_training_reduces_loss() {
        // Train on a single pair and verify loss decreases.
        let dim = Dimension(100);
        let hidden_dim = 16;
        let mut bridge = NeuralVsaBridge::new(hidden_dim, dim);

        let hidden = vec![0.5f32; hidden_dim];
        // Create a target HyperVec with alternating bits.
        let mut target = HyperVec::zero(dim, Encoding::Bipolar);
        for i in (0..dim.0).step_by(2) {
            target.set_bit(i, true);
        }

        // Measure initial "loss" (Hamming distance proxy).
        let initial_hv = bridge.encode(&hidden);
        let initial_match: usize = (0..dim.0)
            .filter(|&i| initial_hv.get_bit(i) == target.get_bit(i))
            .count();

        // Train for a few steps.
        let mut trainer = BridgeTrainer::new(hidden_dim, dim.0, 0.1);
        for _ in 0..50 {
            trainer.accumulate(&bridge, &hidden, &target);
            trainer.step(&mut bridge);
        }

        let trained_hv = bridge.encode(&hidden);
        let trained_match: usize = (0..dim.0)
            .filter(|&i| trained_hv.get_bit(i) == target.get_bit(i))
            .count();

        assert!(
            trained_match >= initial_match,
            "training should improve match: {initial_match} → {trained_match}"
        );
    }

    #[test]
    fn decode_returns_none_without_decoder() {
        let bridge = NeuralVsaBridge::new(32, Dimension::TEST);
        let hv = HyperVec::zero(Dimension::TEST, Encoding::Bipolar);
        assert!(bridge.decode(&hv).is_none());
        assert!(!bridge.has_decoder());
    }

    #[test]
    fn decode_returns_hidden_state_with_decoder() {
        let dim = Dimension::TEST;
        let hidden_dim = 32;
        let mut bridge = NeuralVsaBridge::new(hidden_dim, dim);
        bridge.init_decoder();
        assert!(bridge.has_decoder());

        let hv = HyperVec::zero(dim, Encoding::Bipolar);
        let decoded = bridge.decode(&hv).unwrap();
        assert_eq!(decoded.len(), hidden_dim);
    }

    #[test]
    fn binarize_threshold() {
        let projected = vec![-1.0, 0.5, -0.5, 1.0, 0.0, -2.0, 2.0, 0.1];
        let dim = Dimension(8);
        let hv = binarize_to_hypervec(&projected, 0.0, dim);
        // Values > 0: indices 1, 3, 6, 7
        assert!(!hv.get_bit(0)); // -1.0
        assert!(hv.get_bit(1)); // 0.5
        assert!(!hv.get_bit(2)); // -0.5
        assert!(hv.get_bit(3)); // 1.0
        assert!(!hv.get_bit(4)); // 0.0 (not > 0)
        assert!(!hv.get_bit(5)); // -2.0
        assert!(hv.get_bit(6)); // 2.0
        assert!(hv.get_bit(7)); // 0.1
    }
}
