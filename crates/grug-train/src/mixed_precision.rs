//! Mixed-precision training (Phase 51 P6.1b(c), ADR 107 decision 3c): f16
//! activations on the trainable graph paired with f32 master weights and
//! f32 Adam moments.
//!
//! # Why this exists
//!
//! ADR 107's whole-backend `Metal<f16>` measurement halved memory and step
//! time versus `Metal<f32>`, but the loss curve lagged (`loss@4` 5.25 vs
//! 4.35): Burn 0.21 has no built-in mixed precision, so running the entire
//! backend in f16 also puts the *weights*, *gradients* and *Adam moments*
//! in f16 (`burn-optim` keeps optimizer state as `Tensor<B, D>` in whatever
//! dtype `B` is). f16 activations are fine for this model; f16 optimizer
//! state is not (ADR 107, "What the memory actually is").
//!
//! This module keeps the forward/backward *graph* in f16 (memory win, the
//! actual goal) while keeping the thing Adam accumulates state for -- the
//! weights themselves -- in f32.
//!
//! # Design
//!
//! [`MixedPrecisionTrainer<BLow, BHigh>`] owns the *master* copy of the
//! model and its Adam optimizer state, both on `BHigh` (e.g.
//! `Autodiff<Metal<f32>>`). The caller keeps driving forward/backward on a
//! *separate* low-precision model (`TinyTransformer<BLow>`, e.g.
//! `Autodiff<Metal<f16>>`) exactly as the plain training loop does -- this
//! module never touches `transformer.rs`. Each guarded step
//! ([`MixedPrecisionTrainer::train_step_guarded`]):
//!
//! 1. forward + masked cross-entropy on the low-precision model (f16)
//! 2. `.backward()` on a **scaled** loss (see [`DEFAULT_LOSS_SCALE`]) to
//!    push f16 gradients up out of denormal/underflow range
//! 3. the resulting f16 gradients are cast to f32, unscaled, and applied to
//!    the f32 master via Adam
//! 4. the updated f32 master weights are cast back down to f16 and written
//!    into a fresh low-precision model for the next step
//!
//! `BHigh: AutodiffBackend<Device = BLow::Device>` is enforced in the type
//! itself: both models must live on the same physical device (this is what
//! makes step 2's and step 4's tensor casts pure dtype conversions, not
//! also a device transfer).
//!
//! Every float parameter keeps the *same* `ParamId` across both models for
//! the trainer's whole lifetime -- [`burn::optim::GradientsParams`] and the
//! optimizer's own state are both keyed by `ParamId`, so an id mismatch
//! would silently drop every update rather than error. The two models are
//! linked once, at construction ([`MixedPrecisionTrainer::link`]), via a
//! full-precision in-memory record round trip
//! ([`BinBytesRecorder<FullPrecisionSettings>`]), which already preserves
//! `ParamId` on load (this is also how `checkpoint::load_checkpoint`
//! reconnects saved Adam moments to the right parameter after a resume).
//! After that one-time link, every per-step cast reuses
//! [`burn::optim::GradientsParams`] purely as a generic `ParamId -> Tensor<B,
//! D>` bag (its gradient-flavored name notwithstanding) keyed by those same
//! ids, via [`burn::module::ModuleVisitor`]/[`burn::module::ModuleMapper`] --
//! no serialization on the hot path.

use std::marker::PhantomData;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

use crate::error::{GrugError, GrugResult};
use crate::train::{Batch, global_grad_norm, masked_cross_entropy};
use crate::transformer::{TinyTransformer, TinyTransformerConfig};

/// Static loss-scale factor applied before `.backward()` on the
/// low-precision graph, and divided back out of the f32 gradients before
/// they reach the master Adam optimizer.
///
/// f16 has an 11-bit mantissa but only 5 exponent bits (smallest normal
/// value ~6.1e-5); gradients for a healthy loss are routinely well below
/// that, so without scaling they can silently flush to zero partway through
/// the low-precision backward pass, well before they ever reach the
/// (correctly f32) master. 1024 is the standard mixed-precision default
/// (NVIDIA's mixed-precision training guide, PyTorch AMP's initial scale)
/// and leaves headroom below `f16::MAX` (65504) for this model's loss
/// magnitudes (a handful of nats).
pub const DEFAULT_LOSS_SCALE: f32 = 1024.0;

/// Owns the f32 master weights + Adam optimizer state for a
/// [`TinyTransformer`] whose forward/backward graph runs at lower precision
/// on a separate backend `BLow` (see the module docs).
///
/// `BHigh: AutodiffBackend<Device = BLow::Device>` is the load-bearing part
/// of this signature: the master and the low-precision model must share a
/// physical device, or every cast in [`Self::train_step_guarded`] would
/// also need a cross-device transfer (unsupported by the plain
/// `Tensor::from_data` casts this module uses).
pub struct MixedPrecisionTrainer<BLow, BHigh>
where
    BLow: AutodiffBackend,
    BHigh: AutodiffBackend<Device = BLow::Device>,
{
    master: TinyTransformer<BHigh>,
    optim: OptimizerAdaptor<Adam, TinyTransformer<BHigh>, BHigh>,
    loss_scale: f32,
    device_high: BHigh::Device,
    /// `BLow` is not otherwise stored in a field -- every method that needs
    /// it takes the low-precision model or its device as an argument -- but
    /// it must stay fixed for this trainer's whole lifetime (see the module
    /// docs on `ParamId` stability), so it stays part of the type.
    _low: PhantomData<fn() -> BLow>,
}

impl<BLow, BHigh> MixedPrecisionTrainer<BLow, BHigh>
where
    BLow: AutodiffBackend,
    BHigh: AutodiffBackend<Device = BLow::Device>,
{
    /// Build the f32 master + a fresh AdamW optimizer by casting
    /// `low_model`'s current weights up to `BHigh`, preserving every
    /// parameter's `ParamId` (see the module docs). `low_model` itself is
    /// left untouched -- callers keep driving it directly and hand it back
    /// to [`Self::train_step_guarded`] each step.
    ///
    /// `config` must be the exact [`TinyTransformerConfig`] `low_model` was
    /// initialized from -- it is used to allocate the (immediately
    /// overwritten) master shell before loading `low_model`'s cast values
    /// into it, exactly as `checkpoint::load_checkpoint` requires a
    /// matching config for its model/optimizer shapes.
    pub fn link(
        low_model: &TinyTransformer<BLow>,
        config: &TinyTransformerConfig,
        device_high: &BHigh::Device,
        loss_scale: f32,
    ) -> GrugResult<Self> {
        let master = clone_across_backends::<BLow, BHigh>(low_model, config, device_high)?;
        let optim = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init::<BHigh, TinyTransformer<BHigh>>();
        Ok(Self {
            master,
            optim,
            loss_scale,
            device_high: device_high.clone(),
            _low: PhantomData,
        })
    }

    /// Resume from a checkpointed master + optimizer
    /// (`checkpoint::load_checkpoint::<BHigh>(...)` already returns exactly
    /// these two values, since the master is what gets checkpointed -- see
    /// the module docs).
    pub fn from_master(
        master: TinyTransformer<BHigh>,
        optim: OptimizerAdaptor<Adam, TinyTransformer<BHigh>, BHigh>,
        device_high: BHigh::Device,
        loss_scale: f32,
    ) -> Self {
        Self {
            master,
            optim,
            loss_scale,
            device_high,
            _low: PhantomData,
        }
    }

    /// The current f32 master weights -- what `checkpoint::save_checkpoint`
    /// should be called with, so the saved `foundation-model` stays loadable
    /// by the existing plain-f32 eval/generation code (see the module
    /// docs).
    pub fn master(&self) -> &TinyTransformer<BHigh> {
        &self.master
    }

    /// The master's Adam optimizer state -- pass alongside
    /// [`Self::master`] to `checkpoint::save_checkpoint`.
    pub fn optim(&self) -> &OptimizerAdaptor<Adam, TinyTransformer<BHigh>, BHigh> {
        &self.optim
    }

    /// Cast the current f32 master down into a fresh low-precision model on
    /// `BLow`, ready for the next forward/backward pass. Shares `ParamId`s
    /// with the master (and, by the [`Self::link`] invariant, with the
    /// `low_model` the trainer was linked from).
    ///
    /// `config` must match the config [`Self::link`] (or the original
    /// checkpoint) was built from, for the same reason as there.
    pub fn low_precision_model(
        &self,
        config: &TinyTransformerConfig,
        device_low: &BLow::Device,
    ) -> GrugResult<TinyTransformer<BLow>> {
        clone_across_backends::<BHigh, BLow>(&self.master, config, device_low)
    }

    /// One guarded training step: forward + masked cross-entropy on
    /// `low_model` (f16), backward under [`Self::loss_scale`], cast +
    /// unscale the resulting gradients onto the f32 master, Adam-step the
    /// master, then cast the updated master back down into a fresh
    /// low-precision model for the caller's next step.
    ///
    /// Mirrors [`crate::train::train_step_guarded`]'s loss-cap /
    /// grad-norm-cap gating exactly (loss is read and gated *before*
    /// `.backward()`; grad norm is computed and gated *before* the
    /// optimizer step), so callers can swap between the two without
    /// changing their guard logic. Returns `(low_model, loss, grad_norm)`;
    /// `grad_norm` is `f32::NAN` when the loss gate skipped before backward
    /// ran, exactly as in the plain path.
    #[allow(clippy::too_many_arguments)]
    pub fn train_step_guarded(
        &mut self,
        low_model: TinyTransformer<BLow>,
        batch: &Batch<BLow>,
        lr: f64,
        loss_cap: f32,
        grad_norm_cap: f32,
        device_low: &BLow::Device,
    ) -> (TinyTransformer<BLow>, f32, f32) {
        let mask_flags: Vec<i64> = batch
            .loss_mask
            .iter()
            .map(|&enabled| i64::from(enabled))
            .collect();
        let mask_tensor = Tensor::<BLow, 1, Int>::from_data(
            TensorData::new(mask_flags, [batch.batch_size * batch.seq_len]),
            device_low,
        );

        let logits = low_model.forward(batch.input_ids.clone());
        let masked_loss = masked_cross_entropy(logits, batch.targets.clone(), &mask_tensor);

        // Read the unscaled loss BEFORE backward, exactly as
        // `train::train_step_guarded` does: a single non-finite or
        // way-too-large batch must never reach the optimizer.
        let loss_value: f32 = masked_loss.clone().into_scalar().elem();
        if !loss_value.is_finite() || loss_value > loss_cap {
            return (low_model, loss_value, f32::NAN);
        }

        let scaled_loss = masked_loss.mul_scalar(self.loss_scale);
        let grads_low_raw = scaled_loss.backward();
        let grads_low = GradientsParams::from_grads(grads_low_raw, &low_model);
        let grads_high = cast_gradients::<BLow, BHigh>(
            &low_model,
            grads_low,
            1.0 / self.loss_scale,
            &self.device_high,
        );

        let grad_norm =
            global_grad_norm::<BHigh, TinyTransformer<BHigh>>(&grads_high, &self.master);
        if !grad_norm.is_finite() || grad_norm > grad_norm_cap {
            return (low_model, loss_value, grad_norm);
        }

        self.master = self.optim.step(lr, self.master.clone(), grads_high);
        let synced = sync_low_precision::<BHigh, BLow>(&self.master, low_model, device_low);

        (synced, loss_value, grad_norm)
    }
}

/// Cross-backend, `ParamId`-preserving copy of `model` onto `To`, via an
/// in-memory full-precision record round trip.
///
/// This is the same mechanism `checkpoint.rs` already relies on for
/// cross-backend checkpoint loading (e.g. loading an ndarray-trained
/// checkpoint into a GPU eval backend): a Burn `Recorder`'s `Item`
/// representation stores each parameter's `ParamId` alongside its data at a
/// fixed on-the-wire precision ([`FullPrecisionSettings`], f32), and
/// `Module::load_record` overwrites both the id *and* the value of each
/// target parameter from that `Item` (`Param::transform_for_load`) --
/// unlike `Module::map`, which can only transform tensors in place on a
/// single backend. `config` is used only to allocate the (immediately
/// overwritten) target shell.
fn clone_across_backends<From, To>(
    model: &TinyTransformer<From>,
    config: &TinyTransformerConfig,
    device: &To::Device,
) -> GrugResult<TinyTransformer<To>>
where
    From: Backend,
    To: Backend,
{
    // Burn's lazily-initialized `Param`s (the default for freshly
    // `Config::init`-ed modules) each carry their *own* independent
    // uninitialized state: `Param::clone` on a not-yet-materialized
    // parameter clones the deferred-init recipe, not a value, so `model`
    // and `model.clone()` would each draw their *own* independent random
    // sample the first time anything reads them (`Param::val`'s doc:
    // "initializing one does not affect the other"). Force every
    // parameter to materialize *on `model` itself* first (`Param::val`
    // only needs `&self` -- it caches through interior mutability) so the
    // clone below captures the exact values `model` will actually use for
    // every future forward pass, not an independent re-roll.
    force_materialize(model);

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder
        .record(model.clone().into_record(), ())
        .map_err(|source| GrugError::MixedPrecisionCastFailed {
            message: format!("recording low-precision model: {source}"),
        })?;
    let record =
        recorder
            .load(bytes, device)
            .map_err(|source| GrugError::MixedPrecisionCastFailed {
                message: format!("loading cast model: {source}"),
            })?;
    Ok(config.init::<To>(device).load_record(record))
}

/// Force every float parameter in `model` to materialize in place (see
/// [`clone_across_backends`] for why this matters). No-op for parameters
/// that are already initialized.
fn force_materialize<B: Backend>(model: &TinyTransformer<B>) {
    struct ForceMaterialize;

    impl<B: Backend> ModuleVisitor<B> for ForceMaterialize {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let _ = param.val();
        }
    }

    model.visit(&mut ForceMaterialize);
}

/// Cast `grads_low` (registered as `BLow::InnerBackend` tensors, per
/// `GradientsParams::from_grads`) up to `BHigh::InnerBackend`, dividing out
/// `inv_loss_scale` (`1 / loss_scale`) along the way, keyed by the same
/// `ParamId`s `low_model` itself carries.
///
/// Visits `low_model` (rather than iterating `grads_low` directly) so every
/// registered gradient is paired with the dimensionality `D` its
/// `ModuleVisitor::visit_float::<D>` call is monomorphized for --
/// `GradientsParams` is dimension-erased in its public API.
fn cast_gradients<BLow, BHigh>(
    low_model: &TinyTransformer<BLow>,
    grads_low: GradientsParams,
    inv_loss_scale: f32,
    device_high: &BHigh::Device,
) -> GradientsParams
where
    BLow: AutodiffBackend,
    BHigh: AutodiffBackend,
{
    struct CastVisitor<'a, BLow: AutodiffBackend, BHigh: AutodiffBackend> {
        grads_low: GradientsParams,
        grads_high: GradientsParams,
        inv_loss_scale: f32,
        device_high: &'a BHigh::Device,
        _low: PhantomData<BLow>,
    }

    impl<BLow, BHigh> ModuleVisitor<BLow> for CastVisitor<'_, BLow, BHigh>
    where
        BLow: AutodiffBackend,
        BHigh: AutodiffBackend,
    {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<BLow, D>>) {
            let Some(grad) = self.grads_low.remove::<BLow::InnerBackend, D>(param.id) else {
                // No gradient for this parameter (e.g. it did not
                // participate in this batch's forward pass) -- nothing to
                // register, matches `GradientsParams::from_grads`'s own
                // behavior of simply omitting such parameters.
                return;
            };
            let data = grad
                .into_data()
                .convert::<<BHigh::InnerBackend as BackendTypes>::FloatElem>();
            let grad_high = Tensor::<BHigh::InnerBackend, D>::from_data(data, self.device_high)
                .mul_scalar(self.inv_loss_scale);
            self.grads_high
                .register::<BHigh::InnerBackend, D>(param.id, grad_high);
        }
    }

    let mut visitor = CastVisitor::<BLow, BHigh> {
        grads_low,
        grads_high: GradientsParams::new(),
        inv_loss_scale,
        device_high,
        _low: PhantomData,
    };
    low_model.visit(&mut visitor);
    visitor.grads_high
}

/// Write the master's current f32 weights down into `low_model`, casting
/// each parameter to `BLow`'s native float element and keeping `low_model`'s
/// own `ParamId`s (which, by the [`MixedPrecisionTrainer::link`] invariant,
/// already equal the master's).
///
/// Two passes: [`ModuleVisitor`] over `master` collects a `ParamId`-keyed
/// bag of already-cast, already-on-`device_low` values (reusing
/// [`GradientsParams`] purely as a generic container -- see the module
/// docs), then [`ModuleMapper`] over `low_model` writes each value back in
/// place, preserving `low_model`'s own ids and `require_grad` flags via
/// `Param::from_mapped_value`.
fn sync_low_precision<BHigh, BLow>(
    master: &TinyTransformer<BHigh>,
    low_model: TinyTransformer<BLow>,
    device_low: &BLow::Device,
) -> TinyTransformer<BLow>
where
    BHigh: AutodiffBackend,
    BLow: AutodiffBackend,
{
    struct CollectVisitor<'a, BHigh: AutodiffBackend, BLow: AutodiffBackend> {
        bag: GradientsParams,
        device_low: &'a BLow::Device,
        _high: PhantomData<BHigh>,
    }

    impl<BHigh, BLow> ModuleVisitor<BHigh> for CollectVisitor<'_, BHigh, BLow>
    where
        BHigh: AutodiffBackend,
        BLow: AutodiffBackend,
    {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<BHigh, D>>) {
            let data = param.val().into_data().convert::<BLow::FloatElem>();
            let value = Tensor::<BLow, D>::from_data(data, self.device_low);
            self.bag.register::<BLow, D>(param.id, value);
        }
    }

    let mut collector = CollectVisitor::<BHigh, BLow> {
        bag: GradientsParams::new(),
        device_low,
        _high: PhantomData,
    };
    master.visit(&mut collector);
    let bag = collector.bag;

    struct WriteMapper<'a> {
        bag: &'a GradientsParams,
    }

    impl<BLow: AutodiffBackend> ModuleMapper<BLow> for WriteMapper<'_> {
        fn map_float<const D: usize>(
            &mut self,
            param: Param<Tensor<BLow, D>>,
        ) -> Param<Tensor<BLow, D>> {
            let (id, old_value, mapper) = param.consume();
            let require_grad = old_value.is_require_grad();
            let new_value = match self.bag.get::<BLow, D>(id) {
                Some(value) => value.set_require_grad(require_grad),
                None => old_value,
            };
            Param::from_mapped_value(id, new_value, mapper)
        }
    }

    let mut writer = WriteMapper { bag: &bag };
    low_model.map(&mut writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuAutodiffBackend;
    use crate::tokenizer::{ByteTokenizer, VOCAB_SIZE};
    use crate::train::build_batch;
    use grug_data::Dataset;

    fn tiny_config() -> TinyTransformerConfig {
        TinyTransformerConfig::new(VOCAB_SIZE, 16, 2, 1, 32, 64)
    }

    /// Mechanics check for the master-weights round trip, run entirely on
    /// the CPU `NdArray<f32>` backend for both "low" and "high" (there is
    /// no real f16 backend on CPU, but the plumbing under test -- `link`
    /// preserving `ParamId`s, `train_step_guarded` updating the master, and
    /// syncing the result back into a low-precision model -- is exactly the
    /// same regardless of which concrete dtypes `BLow`/`BHigh` use).
    ///
    /// Proves: (1) a fresh `link` produces a master that is numerically
    /// identical to the model it was linked from, (2) one guarded step
    /// actually moves the master's weights, and (3) the synced
    /// low-precision model returned by that step exactly matches the new
    /// master -- i.e. "f32 master -> f16 params -> gradient step -> master
    /// updated, params equal cast master".
    #[test]
    fn master_weights_round_trip_and_sync() {
        let device = Default::default();
        let config = tiny_config();
        let low_model = config.init::<CpuAutodiffBackend>(&device);

        let tokenizer = ByteTokenizer::new();
        let dataset = Dataset::new(3).unwrap();
        let examples = dataset.train_examples(4).unwrap();
        let batch =
            build_batch::<CpuAutodiffBackend, _>(&tokenizer, &examples, 64, &device).unwrap();

        let mut trainer = MixedPrecisionTrainer::<CpuAutodiffBackend, CpuAutodiffBackend>::link(
            &low_model,
            &config,
            &device,
            DEFAULT_LOSS_SCALE,
        )
        .unwrap();

        let probe_ids = batch.input_ids.clone();
        let before_master = trainer
            .master()
            .forward(probe_ids.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let before_low = low_model
            .forward(probe_ids.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(
            before_master, before_low,
            "a fresh link must produce a master numerically identical to the source model"
        );

        let (synced_low, loss, grad_norm) = trainer.train_step_guarded(
            low_model,
            &batch,
            1e-2,
            f32::INFINITY,
            f32::INFINITY,
            &device,
        );
        assert!(loss.is_finite(), "loss went non-finite: {loss}");
        assert!(
            grad_norm.is_finite(),
            "grad_norm went non-finite: {grad_norm}"
        );

        let after_master = trainer
            .master()
            .forward(probe_ids.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_ne!(
            before_master, after_master,
            "master weights did not change after a guarded step"
        );

        let after_low = synced_low
            .forward(probe_ids)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(
            after_master, after_low,
            "low-precision model was not synced from the updated master"
        );
    }
}
