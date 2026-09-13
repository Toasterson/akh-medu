"""Compare two renderings of the same completed trajectory, without inference."""
import json

from command_registry import registry_tools
from compact_evidence import CompactEvidence,compact_messages
from run_binding_kata import load,write


def preflight(parent,out):
    from transformers import AutoTokenizer
    p=load(parent/'plan.json');tokenizer=AutoTokenizer.from_pretrained(p['model'],local_files_only=True)
    checks=[];phases={};tools=registry_tools(['fano','zuri'])
    def count(messages):
        # Transformers 5 returns BatchEncoding by default; count token IDs,
        # matching the MLX tokenizer wrapper rather than dictionary fields.
        return len(tokenizer.apply_chat_template(messages,tools=tools,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False))
    for phase in p['phases']:
        state=CompactEvidence();receipts=[];sizes=[]
        for line in (parent/phase/'trace.jsonl').read_text().splitlines():
            t=json.loads(line)
            if t.get('context_mode','history')!='history':raise ValueError('A full-history parent is required for this comparison')
            transformed=compact_messages(t['input'][:-1],state,t['input'][-1])
            original=count(t['input']);compact=count(transformed)
            checks.append(dict(phase=phase,turn=t['index'],passed=state.reconstruct(state.view())==receipts and original==t['prompt_tokens']))
            sizes.append(dict(turn=t['index'],history_tokens=original,compact_tokens=compact))
            state.observe(t['wire_request'],t['response']);receipts.append(dict(request=t['wire_request'],response=t['response']))
        checks.append(dict(phase=phase,turn='final',passed=state.reconstruct(state.view())==receipts))
        phases[phase]=dict(turns=sizes,history_sum=sum(x['history_tokens'] for x in sizes),compact_sum=sum(x['compact_tokens'] for x in sizes),
                           history_max=max(x['history_tokens'] for x in sizes),compact_max=max(x['compact_tokens'] for x in sizes))
    result=dict(passed=all(c['passed'] for c in checks),checks=checks,phases=phases,
                same_trajectory=True,inference=False,tools_executed=False,certification=False)
    write(out,result)
    if not result['passed']:raise ValueError('Compact evidence preflight failed')
    return result
