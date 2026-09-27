#!/usr/bin/env python3
"""One synthetic LoRA optimizer step on the pinned Gemma4Unified parent; no deployment."""
import argparse
import hashlib
import json
import time
import sys
from pathlib import Path

PARENT_REVISION = '5926caa4ec0cac5cbfadaf4077420520de1d5205'
PARENT_SHA256 = '5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d'
PARENT_CONFIG_SHA256 = '478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9'
TARGET = 'model.language_model.layers.47.self_attn.q_proj'


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python disables proof assertions; use an unoptimized interpreter')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download-receipt', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'schema': 'lilith/adapter-mechanics@1', 'status': 'running', 'quality_claim': False,
              'deployed_weights_changed': False, 'parent_revision': PARENT_REVISION,
              'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'parent_sha256': PARENT_SHA256, 'target': TARGET, 'rank': 2,
              'synthetic_tokens': 16, 'optimizer_steps': 1, 'use_cache': False,
              'reload_comparison_mode': 'eval', 'python_optimize': sys.flags.optimize, 'frozen_base_validation': 'requires_grad false; optimizer exclusion; all tensor mutation versions unchanged'}

    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')

    try:
        receipt = json.loads(args.download_receipt.read_text())
        assert receipt['revision'] == PARENT_REVISION
        assert receipt['repo'] == 'google/gemma-4-12B-it'
        assert Path(receipt['cache']).resolve() == args.parent.resolve()
        parent = next(a for a in receipt['artifacts'] if a['name'] == 'model.safetensors')
        assert parent['sha256'] == PARENT_SHA256
        assert (args.parent / 'model.safetensors').stat().st_size == parent['bytes']
        report['download_receipt_sha256'] = hashlib.sha256(args.download_receipt.read_bytes()).hexdigest()
        report['phase'] = 'verifying_parent'
        persist()
        verified = {}
        for artifact in receipt['artifacts']:
            path = args.parent / artifact['name']
            assert path.parent.resolve() == args.parent.resolve()
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                while block := handle.read(4 * 1024 * 1024):
                    digest.update(block)
            assert path.stat().st_size == artifact['bytes']
            assert digest.hexdigest() == artifact['sha256']
            verified[artifact['name']] = digest.hexdigest()
        assert 'config.json' in verified and 'model.safetensors' in verified
        assert verified['config.json'] == PARENT_CONFIG_SHA256
        assert verified['model.safetensors'] == PARENT_SHA256
        report['consumed_parent_artifacts_sha256'] = verified
        import torch
        import transformers
        import peft
        from transformers import Gemma4UnifiedForConditionalGeneration
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        report['versions'] = {'torch': torch.__version__, 'hip': torch.version.hip,
                              'transformers': transformers.__version__, 'peft': peft.__version__}
        assert torch.version.hip, 'Require actual ROCm torch, not an accidental CPU/CUDA build'
        assert torch.cuda.is_available()
        torch.manual_seed(73)
        torch.set_num_threads(2)
        report['device'] = torch.cuda.get_device_name(0)
        report['memory_before'] = dict(zip(['free', 'total'], torch.cuda.mem_get_info()))
        # Tiny independent BF16 forward/backward numerical control before the full model.
        left_cpu, right_cpu = torch.randn(32, 32), torch.randn(32, 32)
        left = left_cpu.to(device='cuda', dtype=torch.bfloat16).requires_grad_()
        right = right_cpu.to(device='cuda', dtype=torch.bfloat16)
        product = left @ right
        reference = left_cpu.to(torch.bfloat16).float() @ right_cpu.to(torch.bfloat16).float()
        relative_error = (product.float().cpu() - reference).norm() / reference.norm()
        product.sum().backward()
        grad_reference = torch.ones(32, 32) @ right_cpu.to(torch.bfloat16).float().T
        grad_error = (left.grad.float().cpu() - grad_reference).norm() / grad_reference.norm()
        assert relative_error.item() < 0.01 and grad_error.item() < 0.01
        report['bf16_control'] = {'forward_relative_l2': relative_error.item(), 'gradient_relative_l2': grad_error.item()}
        del left, right, product
        torch.cuda.empty_cache()
        report['phase'] = 'loading'
        persist()
        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            args.parent, local_files_only=True, dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, low_cpu_mem_usage=True, attn_implementation='eager')
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        assert TARGET in dict(model.named_modules())
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0,
                              target_modules=[TARGET], bias='none'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        parameters = dict(model.named_parameters())
        trainable = {name: param for name, param in parameters.items() if param.requires_grad}
        assert trainable and all('lora_' in name and TARGET in name for name in trainable)
        frozen = {name: param._version for name, param in parameters.items() if not param.requires_grad}
        before = {name: param.detach().cpu().clone() for name, param in trainable.items()}
        report['trainable_parameters'] = sum(p.numel() for p in trainable.values())
        report['trainable_names'] = list(trainable)
        inputs = torch.tensor([[2, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 106]], device='cuda')
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=0.001, weight_decay=0.0)
        model.train()
        report['phase'] = 'forward_backward'
        persist()
        result = model(input_ids=inputs, labels=inputs, use_cache=False)
        loss = result.loss
        assert torch.isfinite(loss)
        report['loss'] = loss.item()
        loss.backward()
        gradients = {name: param.grad for name, param in trainable.items()}
        assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients.values())
        report['gradient_norms'] = {name: grad.float().norm().item() for name, grad in gradients.items()}
        assert any(value > 0 for value in report['gradient_norms'].values())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del result, loss, gradients
        assert all(param._version == frozen[name] for name, param in parameters.items() if name in frozen)
        changed = [name for name, param in trainable.items() if not torch.equal(param.detach().cpu(), before[name])]
        assert changed
        assert all(torch.isfinite(param).all() for param in trainable.values())
        report['changed_adapter_tensors'] = changed
        report['frozen_tensor_versions_unchanged'] = True
        model.eval()
        with torch.no_grad():
            expected = model(input_ids=inputs, use_cache=False).logits.detach().cpu()
        assert torch.isfinite(expected).all()
        adapter_dir = args.output / 'adapter'
        model.save_pretrained(adapter_dir, safe_serialization=True)
        optimizer_path = args.output / 'optimizer.pt'
        torch.save(optimizer.state_dict(), optimizer_path)
        # Destroy the in-memory adapter so reload equality is not a vacuous same-model check.
        saved_adapter = {name: param.detach().cpu().clone() for name, param in trainable.items()}
        with torch.no_grad():
            for param in trainable.values():
                param.zero_()
            zeroed_logits = model(input_ids=inputs, use_cache=False).logits.detach().cpu()
        assert torch.isfinite(zeroed_logits).all()
        report['adapter_forward_max_abs_effect'] = (expected.float() - zeroed_logits.float()).abs().max().item()
        report['adapter_forward_effect_demonstrated'] = report['adapter_forward_max_abs_effect'] > 0
        del zeroed_logits
        state = load_file(str(adapter_dir / 'adapter_model.safetensors'))
        set_peft_model_state_dict(model, state)
        assert all(torch.equal(param.detach().cpu(), saved_adapter[name]) for name, param in trainable.items())
        report['adapter_tensors_exact_reload'] = True
        reloaded_optimizer = torch.optim.AdamW(list(trainable.values()), lr=0.001, weight_decay=0.0)
        reloaded_optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu', weights_only=True))
        assert reloaded_optimizer.state_dict()['param_groups'] == optimizer.state_dict()['param_groups']
        assert len(reloaded_optimizer.state) == len(optimizer.state)
        for parameter, values in optimizer.state.items():
            restored = reloaded_optimizer.state[parameter]
            assert values.keys() == restored.keys()
            for key, value in values.items():
                if torch.is_tensor(value):
                    assert torch.equal(value.detach().cpu(), restored[key].detach().cpu())
                else:
                    assert value == restored[key]
        report['optimizer_state_exact_reload'] = True
        model.eval()
        with torch.no_grad():
            actual = model(input_ids=inputs, use_cache=False).logits.detach().cpu()
        assert torch.isfinite(actual).all()
        error = (expected.float() - actual.float()).abs().max().item()
        report['reload_logits_max_abs_error'] = error
        assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
        assert all(param._version == frozen[name] for name, param in parameters.items() if name in frozen)
        report.update(status='passed', phase='finished', optimizer_state_entries=len(reloaded_optimizer.state),
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      adapter_sha256=hashlib.sha256((adapter_dir / 'adapter_model.safetensors').read_bytes()).hexdigest())
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
