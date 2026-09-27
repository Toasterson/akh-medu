"""Audited complete trajectories and token-MC KLPO backward surrogate.

No model, critic or sampler is imported at module load. The estimator follows
KLPO revision 30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696, docs/algorithms.md.
Schema checks detect inconsistent records; they cannot prove an untrusted
collector used its claimed RNG or model. The local sampler helper records actual
full-support categorical draws and probabilities, not invented MC metadata.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def seal_record(record):
    row=copy.deepcopy(record)
    row.pop('record_sha256',None)
    row['context_sha256']=digest(canonical(row['context']))
    row['record_sha256']=digest(canonical(row))
    return row


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def validate_record(record, expected_policy=None, expected_verifier_sha256=None,
                    for_update=True, expected_task=None):
    """Validate immutable collection evidence. Failed/truncated records are retained.

    for_update=False permits nonterminal records for evaluation; it does not make
    their missing observations, invalid numerical values, or hashes acceptable.
    """
    try:
        row=copy.deepcopy(record)
        checksum=row.pop('record_sha256')
        if not _sha(checksum) or digest(canonical(row))!=checksum:
            raise ValueError('trajectory checksum mismatch')
        if row['schema']!='lilith/trajectory@1' or not isinstance(row['id'],str) or not row['id']:
            raise ValueError('unsupported or unidentified trajectory')
        if row['context_sha256']!=digest(canonical(row['context'])):
            raise ValueError('context/features checksum mismatch')
        task=row['task']
        if not all(isinstance(task[k],str) and task[k] for k in ('id','family','split')) or not _sha(task['prompt_sha256']):
            raise ValueError('task provenance incomplete')
        if expected_task is not None and task!=expected_task:
            raise ValueError('task provenance drift')
        if for_update and task['split'] not in ('train','train_candidate'):
            raise ValueError('holdout trajectory cannot update policy')
        policy=row['policy']
        if not all(_sha(policy[k]) for k in ('model_sha256','tokenizer_sha256','adapter_sha256')):
            raise ValueError('model/tokenizer/adapter hashes must be pinned')
        if not isinstance(policy['version'],str) or not policy['version']:
            raise ValueError('sampler policy version required')
        if expected_policy is not None and policy!=expected_policy:
            raise ValueError('sampler policy provenance drift')
        sampler=row['sampler']
        if (sampler['kind']!='categorical' or not _finite(sampler['temperature']) or sampler['temperature']<=0
            or sampler['top_k']!=0 or sampler['top_p']!=1.0 or sampler['full_support'] is not True
            or sampler['with_replacement'] is not True or sampler['aux_independent'] is not True):
            raise ValueError('positive-temperature full-support IID categorical sampling required')
        termination_ids=sampler['termination_token_ids']
        if (not isinstance(termination_ids,list) or not termination_ids
            or any(type(t) is not int or t<0 for t in termination_ids) or len(set(termination_ids))!=len(termination_ids)):
            raise ValueError('pinned allowed termination token IDs required')
        rngs=[sampler[k] for k in ('action_rng_id','aux_rng_id')]
        if any(not isinstance(k,str) or not k for k in rngs) or rngs[0]==rngs[1]:
            raise ValueError('distinct audited action and auxiliary RNG streams required')
        context=row['context']
        if not isinstance(context['prompt'],str) or not isinstance(context['features'],list) or not isinstance(context['lesson_ids'],list):
            raise ValueError('prompt/features/lesson context required')
        if digest(context['prompt'])!=task['prompt_sha256']:
            raise ValueError('task prompt/context binding mismatch')
        ids=row['token_ids'];roles=row['token_roles'];mask=row['policy_mask']
        q=row['sampler_logps'];aux_ids=row['mc_token_ids'];aux_q=row['mc_sampler_logps']
        if not isinstance(ids,list) or not ids or any(type(t) is not int or t<0 for t in ids):
            raise ValueError('actual token IDs required')
        size=len(ids)
        if any(not isinstance(v,list) or len(v)!=size for v in (roles,mask,q,aux_ids,aux_q)):
            raise ValueError('token records must have matching shapes')
        prompt_ids=context['prompt_token_ids']
        if (not isinstance(prompt_ids,list) or not prompt_ids or any(type(t) is not int or t<0 for t in prompt_ids)
            or ids[:len(prompt_ids)]!=prompt_ids or roles[:len(prompt_ids)]!=['prompt']*len(prompt_ids)):
            raise ValueError('actual prompt token prefix mismatch')
        if any(role=='prompt' for role in roles[len(prompt_ids):]):
            raise ValueError('prompt role outside prompt prefix')
        width=None
        for role,active,logq,draws,draw_logps in zip(roles,mask,q,aux_ids,aux_q):
            if role not in ('prompt','assistant','tool','padding') or type(active) is not bool or active!=(role=='assistant'):
                raise ValueError('train every generated policy token, never prompt/tool/padding')
            if not isinstance(draws,list) or not isinstance(draw_logps,list):
                raise ValueError('auxiliary records must be lists')
            if active:
                if not _finite(logq) or logq>0 or not draws or len(draws)!=len(draw_logps):
                    raise ValueError('missing/nonfinite sampler or auxiliary observations')
                if any(type(t) is not int or t<0 for t in draws) or any(not _finite(v) or v>0 for v in draw_logps):
                    raise ValueError('invalid auxiliary IDs/log probabilities')
                if width is not None and len(draws)!=width:
                    raise ValueError('fixed auxiliary sample count required')
                width=len(draws)
            elif logq is not None or draws or draw_logps:
                raise ValueError('nonpolicy tokens must not have sampler or auxiliary draws')
        completion=row['completion'];status=completion['status']
        if status not in ('completed','truncated','failed','cancelled') or not isinstance(completion['generated_text'],str):
            raise ValueError('invalid completion evidence')
        active_ids=[token for token,active in zip(ids,mask) if active]
        if status=='completed' and (not active_ids or type(completion['termination_token_id']) is not int
                                   or active_ids[-1]!=completion['termination_token_id']
                                   or completion['termination_token_id'] not in termination_ids):
            raise ValueError('complete trajectory must include actual policy termination token')
        if for_update and (status!='completed' or not active_ids):
            raise ValueError('incomplete/truncated trajectories are evaluation failures, not updates')
        reward=row['reward']
        if not _finite(reward['value']) or not _sha(reward['verifier_sha256']) or not isinstance(reward['evidence'],dict):
            raise ValueError('finite terminal reward and pinned verifier evidence required')
        if expected_verifier_sha256 is not None and reward['verifier_sha256']!=expected_verifier_sha256:
            raise ValueError('reward verifier provenance drift')
        if reward['evidence'].get('generated_text_sha256')!=digest(completion['generated_text']):
            raise ValueError('reward evidence is not bound to generated text')
        if type(reward['evidence'].get('passed')) is not bool or reward['value']!=float(reward['evidence']['passed']):
            raise ValueError('this probe uses independent binary verifier rewards only')
        if status!='completed' and (reward['value']!=0 or reward['evidence']['passed']):
            raise ValueError('incomplete evaluation cannot receive passing reward')
        return record
    except (KeyError,TypeError,OverflowError) as error:
        raise ValueError('malformed trajectory record: '+str(error)) from error


def claim_update(records, ledger_path, update_id, expected_policy=None, expected_verifier_sha256=None, expected_tasks=None):
    """Durably consume each complete training bank once; lost-response replay is safe.

    Claim before optimizer work. A crash afterward needs audit; it cannot silently
    reuse the bank. This function does not perform or retry an optimizer step.
    """
    if not isinstance(update_id,str) or not update_id or not records:
        raise ValueError('named update and nonempty complete training batch required')
    if not isinstance(expected_tasks,dict) or not expected_tasks:
        raise ValueError('authoritative task manifest required for update claim')
    for record in records:
        expected_task=expected_tasks.get(record.get('task',{}).get('id'))
        if expected_task is None:raise ValueError('trajectory task absent from authoritative manifest')
        validate_record(record,expected_policy,expected_verifier_sha256,expected_task=expected_task)
    ids=[record['id'] for record in records]
    if len(set(ids))!=len(ids):raise ValueError('duplicate trajectory in update')
    batch=digest(canonical(sorted(record['record_sha256'] for record in records)))
    path=Path(ledger_path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with sqlite3.connect(path) as db:
        path.chmod(0o600)
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS updates (id TEXT PRIMARY KEY, batch TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS banks (id TEXT PRIMARY KEY, checksum TEXT UNIQUE NOT NULL, update_id TEXT NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        previous=db.execute('SELECT batch FROM updates WHERE id=?',(update_id,)).fetchone()
        if previous:
            if previous[0]!=batch:raise ValueError('update ID reused with different batch')
            return dict(update_id=update_id,batch_sha256=batch,replayed=True,optimizer_replay_authorized=False)
        for record in records:
            if db.execute('SELECT 1 FROM banks WHERE id=? OR checksum=?',(record['id'],record['record_sha256'])).fetchone():
                raise ValueError('auxiliary trajectory bank already consumed')
        db.execute('INSERT INTO updates VALUES (?,?)',(update_id,batch))
        db.executemany('INSERT INTO banks VALUES (?,?,?)',[(r['id'],r['record_sha256'],update_id) for r in records])
    return dict(update_id=update_id,batch_sha256=batch,replayed=False,optimizer_replay_authorized=False)


def token_mc_loss(action_logp,sampler_action_logp,aux_logp,rewards,policy_mask,beta=0.1):
    """Scalar backward surrogate, summed over tokens and averaged over responses.

    Current action/auxiliary log probabilities must share the trainer graph.
    Historical sampler q and rewards never receive gradients. This tensor seam
    cannot validate token IDs or collection independence: validate records first.
    """
    import torch
    tensors=(action_logp,sampler_action_logp,aux_logp,rewards)
    if any(not isinstance(x,torch.Tensor) or not x.is_floating_point() for x in tensors):
        raise ValueError('floating torch tensors required')
    if (not isinstance(policy_mask,torch.Tensor) or policy_mask.dtype!=torch.bool
        or action_logp.ndim!=2 or policy_mask.shape!=action_logp.shape
        or sampler_action_logp.shape!=action_logp.shape or rewards.shape!=(action_logp.shape[0],)
        or aux_logp.ndim!=3 or aux_logp.shape[:2]!=action_logp.shape or aux_logp.shape[2]<1
        or action_logp.shape[0]<1 or action_logp.shape[1]<1):
        raise ValueError('expected action/mask[B,T], auxiliary[B,T,M>=1], reward[B]')
    if any(x.device!=action_logp.device for x in (*tensors,policy_mask)):
        raise ValueError('loss tensors must share a device')
    if not _finite(beta) or beta<=0:
        raise ValueError('finite positive beta required')
    if not bool(policy_mask.any(dim=1).all()):
        raise ValueError('every complete response needs policy tokens')
    if not bool(torch.isfinite(rewards).all()):raise ValueError('nonfinite reward')
    for tensor,mask in ((action_logp,policy_mask),(sampler_action_logp,policy_mask),(aux_logp,policy_mask.unsqueeze(-1).expand_as(aux_logp))):
        values=tensor[mask]
        if not bool(torch.isfinite(values).all()) or bool((values>0).any()):
            raise ValueError('active log probabilities must be finite and nonpositive')
    dtype=torch.float64 if any(x.dtype==torch.float64 for x in tensors) else torch.float32
    current=action_logp.to(dtype).masked_fill(~policy_mask,0)
    historical=sampler_action_logp.detach().to(dtype).masked_fill(~policy_mask,0)
    auxiliary=aux_logp.to(dtype).masked_fill(~policy_mask.unsqueeze(-1),0)
    coefficient=(rewards.detach().to(dtype).unsqueeze(-1)-beta*(current-historical)).detach()
    loss=-(coefficient*(current-auxiliary.mean(-1))).sum(-1).mean()
    if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
    return loss


def sample_policy_tokens(logits, action_generator, aux_generator, mc_samples=8, temperature=1.0):
    """One actual action plus independent IID auxiliary draws from identical q.

    logits and both generators must be on CPU. Float64 probabilities are checked
    for numerical full support; underflow is rejected, never repaired by clipping.
    Repeated auxiliary token values are preserved, including an action collision.
    """
    import torch
    if (not isinstance(logits,torch.Tensor) or logits.ndim!=1 or logits.numel()<2
        or logits.device.type!='cpu' or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())):
        raise ValueError('finite CPU full-vocabulary logits required')
    if type(mc_samples) is not int or mc_samples<1 or not _finite(temperature) or temperature<=0:
        raise ValueError('positive MC count and sampling temperature required')
    if (action_generator is aux_generator or action_generator.device.type!='cpu' or aux_generator.device.type!='cpu'
        or action_generator.initial_seed()==aux_generator.initial_seed()):
        raise ValueError('separate CPU generators with distinct seeds required')
    with torch.no_grad():
        probabilities=torch.softmax(logits.detach().double()/temperature,dim=-1)
        if not bool(torch.isfinite(probabilities).all()) or not bool((probabilities>0).all()):
            raise ValueError('numerically truncated sampler support; refuse collection')
        probabilities=probabilities/probabilities.sum()
        logq=probabilities.log()
        action=torch.multinomial(probabilities,1,replacement=True,generator=action_generator)
        auxiliary=torch.multinomial(probabilities,mc_samples,replacement=True,generator=aux_generator)
    return dict(token_id=action.item(),sampler_logp=logq[action].item(),
        mc_token_ids=auxiliary.tolist(),mc_sampler_logps=logq[auxiliary].tolist())
