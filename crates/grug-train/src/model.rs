//! A minimal 2-layer MLP -- the smallest real Burn [`Module`] that can prove
//! forward + loss + backward + optimizer-step all actually work together on
//! the ndarray/autodiff CPU backend.
//!
//! This is *not* the Grug-Brain transformer. It exists solely to de-risk the
//! Burn integration (Phase 51 P0 gate, ADR 087) before anyone invests time in
//! the real architecture.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Configuration for [`Mlp`].
#[derive(Config, Debug)]
pub struct MlpConfig {
    /// Number of input features.
    pub d_input: usize,
    /// Number of hidden units.
    pub d_hidden: usize,
    /// Number of output features.
    pub d_output: usize,
}

impl MlpConfig {
    /// Build an [`Mlp`] on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            linear1: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            activation: Relu::new(),
            linear2: LinearConfig::new(self.d_hidden, self.d_output).init(device),
        }
    }
}

/// A 2-layer MLP: `Linear -> ReLU -> Linear`.
///
/// Generic over any Burn [`Backend`] `B` -- swap `B` for an autodiff backend
/// (see [`crate::CpuAutodiffBackend`]) to train it, or a plain backend to run
/// inference only.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    /// Forward pass: `input` is `[batch_size, d_input]`, output is
    /// `[batch_size, d_output]`.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuAutodiffBackend;
    use crate::error::{GrugError, GrugResult};
    use burn::module::AutodiffModule;
    use burn::nn::loss::{MseLoss, Reduction};
    use burn::optim::{GradientsParams, Optimizer, SgdConfig};
    use burn::tensor::TensorData;
    use burn::tensor::backend::AutodiffBackend;

    /// R1 gate, part 1: a forward pass through the MLP on the ndarray
    /// backend produces the expected output shape, and does not panic.
    #[test]
    fn forward_pass_produces_expected_shape() {
        let device = Default::default();
        let model = MlpConfig::new(4, 8, 2).init::<CpuAutodiffBackend>(&device);

        let input: Tensor<CpuAutodiffBackend, 2> =
            Tensor::from_data(TensorData::from([[1.0f32, 2.0, 3.0, 4.0]]), &device);

        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 2]);
    }

    /// R1 gate, part 2 (the real proof): forward -> MSE loss -> `.backward()`
    /// -> every parameter has a *present and non-zero* gradient. This is the
    /// load-bearing assertion for "autodiff actually works on the ndarray
    /// backend" -- a missing or all-zero gradient here means Burn is wired
    /// up but not actually differentiating.
    #[test]
    fn backward_pass_produces_nonzero_gradients() -> GrugResult<()> {
        let device = Default::default();
        let model = MlpConfig::new(4, 8, 2).init::<CpuAutodiffBackend>(&device);

        let input: Tensor<CpuAutodiffBackend, 2> = Tensor::from_data(
            TensorData::from([[1.0f32, -2.0, 0.5, 3.0], [0.1, 0.2, -0.3, 0.4]]),
            &device,
        );
        let target: Tensor<CpuAutodiffBackend, 2> =
            Tensor::from_data(TensorData::from([[1.0f32, 0.0], [0.0, 1.0]]), &device);

        let output = model.forward(input);
        let loss = MseLoss::new().forward(output, target, Reduction::Mean);

        let grads = loss.backward();

        // Check gradients on both linear layers' weights: present, and not
        // all-zero (a real signal flowed all the way back from the loss).
        assert_gradient_nonzero(&model.linear1.weight, &grads, "linear1.weight")?;
        assert_gradient_nonzero(&model.linear2.weight, &grads, "linear2.weight")?;

        Ok(())
    }

    /// Helper: fetch the gradient for a `Param<Tensor<_, 2>>` from a
    /// `Gradients` record and assert it is present and non-zero, returning a
    /// rich miette-diagnostic `GrugError` on failure instead of panicking
    /// blindly, per this repo's error-handling convention.
    fn assert_gradient_nonzero(
        param: &burn::module::Param<Tensor<CpuAutodiffBackend, 2>>,
        grads: &<CpuAutodiffBackend as AutodiffBackend>::Gradients,
        name: &str,
    ) -> GrugResult<()> {
        let grad = param
            .grad(grads)
            .ok_or_else(|| GrugError::MissingGradient {
                param: name.to_string(),
            })?;

        let sum_abs: f32 = grad.abs().sum().into_scalar();
        if sum_abs == 0.0 {
            return Err(GrugError::ZeroGradient {
                param: name.to_string(),
            });
        }
        Ok(())
    }

    /// R1 gate, part 3: a handful of real SGD steps against a fixed target
    /// measurably reduce the loss. This is the strongest available proof
    /// that Burn's optimizer + autodiff loop is not just "not panicking" but
    /// actually doing gradient descent on the ndarray backend.
    #[test]
    fn sgd_steps_decrease_loss_on_fixed_target() -> GrugResult<()> {
        let device = Default::default();
        let mut model = MlpConfig::new(4, 8, 2).init::<CpuAutodiffBackend>(&device);
        let mut optim = SgdConfig::new().init::<CpuAutodiffBackend, Mlp<CpuAutodiffBackend>>();

        let input: Tensor<CpuAutodiffBackend, 2> = Tensor::from_data(
            TensorData::from([[1.0f32, -2.0, 0.5, 3.0], [0.1, 0.2, -0.3, 0.4]]),
            &device,
        );
        let target: Tensor<CpuAutodiffBackend, 2> =
            Tensor::from_data(TensorData::from([[1.0f32, 0.0], [0.0, 1.0]]), &device);

        let lr = 0.1;
        let steps = 50;

        let initial_loss: f32 = MseLoss::new()
            .forward(
                model.forward(input.clone()),
                target.clone(),
                Reduction::Mean,
            )
            .into_scalar();

        for _ in 0..steps {
            let output = model.forward(input.clone());
            let loss = MseLoss::new().forward(output, target.clone(), Reduction::Mean);
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);
        }

        let final_loss: f32 = MseLoss::new()
            .forward(model.forward(input), target, Reduction::Mean)
            .into_scalar();

        if final_loss >= initial_loss {
            return Err(GrugError::LossDidNotDecrease {
                start_loss: initial_loss as f64,
                end_loss: final_loss as f64,
                steps,
            });
        }

        // Also sanity-check `valid()` (the AutodiffModule -> inference-mode
        // projection Burn uses for eval) does not panic and preserves shape.
        let model_valid = model.valid();
        let input_valid: Tensor<CpuBackendAlias, 2> =
            Tensor::from_data(TensorData::from([[1.0f32, -2.0, 0.5, 3.0]]), &device);
        let output = model_valid.forward(input_valid);
        assert_eq!(output.dims(), [1, 2]);

        Ok(())
    }

    /// Local alias so the `valid()` sanity check reads clearly: the inner
    /// (non-autodiff) backend of `CpuAutodiffBackend`.
    type CpuBackendAlias = crate::CpuBackend;
}
