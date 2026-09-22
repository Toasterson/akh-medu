import sys,torch,json,gc,traceback
from pathlib import Path
sys.path.insert(0,'scripts/session-platform')
from lilith_adapter_mechanics import PARENT_SHA256,PARENT_CONFIG_SHA256,TARGET
from lilith_training_engine import GemmaTrainingEngine,assert_detached_kv
from transformers import Cache,DynamicCache
from transformers.cache_utils import DynamicLayer,DynamicSlidingWindowLayer
from torch.nn.attention import sdpa_kernel,SDPBackend
torch.set_num_threads(2)
e=GemmaTrainingEngine.load(Path.home()/'.local/share/akh-medu/lilith-training/parent-5926caa4',expected_parent_sha256=PARENT_SHA256,expected_config_sha256=PARENT_CONFIG_SHA256,expected_tokenizer_sha256='cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',target_module=TARGET)
m=e.model;d=m.device;ts=[2]+[100+i%40 for i in range(1100)]
print('READY',flush=True)
for line in sys.stdin:
 try:
  command=json.loads(line)
  if command.get('stop'):break
  exec(command['code'],globals())
  print('DONE',flush=True)
 except BaseException:traceback.print_exc();print('ERROR',flush=True)
