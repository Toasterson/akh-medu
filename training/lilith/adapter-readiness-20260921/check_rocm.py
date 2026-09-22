import json
from pathlib import Path
import torch

result = {'torch':torch.__version__,'hip':torch.version.hip,'status':'running'}
try:
    if not torch.cuda.is_available():
        raise RuntimeError('ROCm device unavailable')
    result['device'] = torch.cuda.get_device_name(0)
    result['properties'] = str(torch.cuda.get_device_properties(0))
    x = torch.tensor([[1., 2.], [4., 8.]], device='cuda', dtype=torch.bfloat16, requires_grad=True)
    y = torch.tensor([[2., 4.], [8., 16.]], device='cuda', dtype=torch.bfloat16)
    out = x @ y
    expected = torch.tensor([[18., 36.], [72., 144.]])
    if not torch.equal(out.float().cpu(), expected):
        raise RuntimeError('Incorrect BF16 matrix product')
    out.sum().backward()
    expected_grad = torch.tensor([[6., 24.], [6., 24.]])
    if not torch.equal(x.grad.float().cpu(), expected_grad):
        raise RuntimeError('Incorrect BF16 gradient')
    result.update(status='passed', output=out.float().cpu().tolist(), gradient=x.grad.float().cpu().tolist())
finally:
    Path(__file__).with_name('rocm-control.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
