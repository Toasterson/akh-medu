"""Instantiate exact Gemma config on meta only; no tensors or GPU context."""
import json
from pathlib import Path
import torch
import transformers
import peft
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
from peft import LoraConfig, get_peft_model

root = Path(__file__).parent
config = Gemma4UnifiedConfig.from_json_file(str(root / 'parent-candidate-config.json'))
target = 'model.language_model.layers.47.self_attn.q_proj'
with torch.device('meta'):
    model = Gemma4UnifiedForConditionalGeneration(config)
    module = dict(model.named_modules())[target]
    shape = list(module.weight.shape)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0, target_modules=[target], bias='none'))
trainable = {name: list(param.shape) for name, param in model.named_parameters() if param.requires_grad}
assert trainable and all('lora_' in name and target in name for name in trainable)
assert all(param.device.type == 'meta' for param in model.parameters())
report = {'status':'passed','device':'meta','architecture':type(model.base_model.model).__name__,
          'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
          'target':target,'target_shape':shape,'trainable_shapes':trainable,'training_executed':False}
(root / 'architecture-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
