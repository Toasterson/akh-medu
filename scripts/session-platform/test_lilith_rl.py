import copy
import itertools
import math
from pathlib import Path
import tempfile
import unittest

import lilith_rl as rl
try:
    import torch
except ImportError:
    torch=None


def record():
    row=dict(schema='lilith/trajectory@1',id='trajectory1',
        task=dict(id='task1',family='training_family',split='train',prompt_sha256=rl.digest('task')),
        policy=dict(model_sha256='a'*64,tokenizer_sha256='b'*64,adapter_sha256='c'*64,version='initial-adapter'),
        sampler=dict(kind='categorical',temperature=1.0,top_k=0,top_p=1.0,full_support=True,
            action_rng_id='actions-73',aux_rng_id='aux-74',with_replacement=True,aux_independent=True,termination_token_ids=[1]),
        context=dict(prompt='task',features=[['domain','integer']],lesson_ids=[],prompt_token_ids=[2,5]),
        token_ids=[2,5,7,1],token_roles=['prompt','prompt','assistant','assistant'],
        policy_mask=[False,False,True,True],sampler_logps=[None,None,-1.2,-2.3],
        mc_token_ids=[[],[],[7,7],[4,1]],mc_sampler_logps=[[],[],[-1.2,-1.2],[-3,-2.3]],
        completion=dict(status='completed',termination_token_id=1,generated_text='def f(a,b): return a'),
        reward=dict(value=1.0,verifier_sha256='d'*64,evidence=dict(passed=True,
            generated_text_sha256=rl.digest('def f(a,b): return a'))))
    return rl.seal_record(row)


class TrajectoryTests(unittest.TestCase):
    def test_complete_record_accepts_independent_duplicate_token_values(self):
        r=record();self.assertIs(rl.validate_record(r,r['policy'],'d'*64),r)

    def test_policy_verifier_context_and_checksum_drift_rejected(self):
        r=record()
        with self.assertRaisesRegex(ValueError,'provenance drift'):rl.validate_record(r,dict(r['policy'],version='other'))
        with self.assertRaisesRegex(ValueError,'verifier provenance'):rl.validate_record(r,expected_verifier_sha256='e'*64)
        r['context']['features'].append(['operation','changed'])
        with self.assertRaisesRegex(ValueError,'checksum'):rl.validate_record(r)
        r=record();r['sampler_logps'][-1]=-0.01
        with self.assertRaisesRegex(ValueError,'checksum'):rl.validate_record(r)

    def test_failed_truncated_records_retained_but_never_updated(self):
        for status in ('failed','truncated','cancelled'):
            r=record();r['completion']['status']=status;r['reward']['value']=0;r['reward']['evidence']['passed']=False
            r=rl.seal_record(r)
            self.assertEqual(rl.validate_record(r,for_update=False)['completion']['status'],status)
            with self.assertRaisesRegex(ValueError,'incomplete/truncated'):rl.validate_record(r)

    def test_prompt_tool_masks_and_dropped_policy_tokens_rejected(self):
        for index in (0,2):
            r=record();r['policy_mask'][index]=not r['policy_mask'][index];r=rl.seal_record(r)
            with self.assertRaisesRegex(ValueError,'policy token'):rl.validate_record(r)
        r=record();r['token_roles'][2]='tool';r=rl.seal_record(r)
        with self.assertRaises(ValueError):rl.validate_record(r)

    def test_unsupported_sampler_configs_and_rng_coupling_rejected(self):
        changes=[('temperature',0),('top_k',2),('top_p',.9),('with_replacement',False),
                 ('aux_independent',False),('aux_rng_id','actions-73')]
        for key,value in changes:
            with self.subTest(key=key):
                r=record();r['sampler'][key]=value;r=rl.seal_record(r)
                with self.assertRaises(ValueError):rl.validate_record(r)

    def test_missing_mc_termination_or_misaligned_tokens_rejected(self):
        for mutation in ('aux','shape','termination','prompt'):
            r=record()
            if mutation=='aux':r['mc_token_ids'][2]=[]
            elif mutation=='shape':r['policy_mask'].pop()
            elif mutation=='termination':r['completion']['termination_token_id']=99
            else:r['context']['prompt_token_ids']=[99,5]
            r=rl.seal_record(r)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):rl.validate_record(r)

    def test_last_arbitrary_token_cannot_be_relabelled_as_termination(self):
        r=record();r['token_ids'][-1]=99;r['completion']['termination_token_id']=99;r=rl.seal_record(r)
        with self.assertRaisesRegex(ValueError,'termination token'):rl.validate_record(r)

    def test_holdout_and_nonfinite_or_unbound_reward_rejected(self):
        r=record();r['task']['split']='development_holdout';r=rl.seal_record(r)
        rl.validate_record(r,for_update=False)
        with self.assertRaisesRegex(ValueError,'holdout'):rl.validate_record(r)
        r=record();r['reward']['value']=float('nan')
        with self.assertRaises(ValueError):rl.seal_record(r)
        r=record();r['completion']['generated_text']='different';r=rl.seal_record(r)
        with self.assertRaisesRegex(ValueError,'generated text'):rl.validate_record(r)
        r=record();r['reward']['value']=0;r=rl.seal_record(r)
        with self.assertRaisesRegex(ValueError,'binary verifier'):rl.validate_record(r)

    def test_claim_rejects_resealed_prompt_and_relabelled_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'banks.sqlite'
            original=record();expected={'task1':original['task']}
            changed=record();changed['context']['prompt']='different'
            changed['task']['prompt_sha256']=rl.digest('different');changed=rl.seal_record(changed)
            with self.assertRaisesRegex(ValueError,'task provenance'):
                rl.claim_update([changed],path,'update1',expected_tasks=expected)
            expected={'task1':dict(original['task'],split='development_holdout')}
            with self.assertRaisesRegex(ValueError,'task provenance'):
                rl.claim_update([original],path,'update2',expected_tasks=expected)
            with self.assertRaisesRegex(ValueError,'authoritative'):
                rl.claim_update([original],path,'update3')
            changed=record();changed['context']['prompt']='different';changed=rl.seal_record(changed)
            with self.assertRaisesRegex(ValueError,'binding mismatch'):rl.validate_record(changed)

    def test_one_use_bank_is_durable_and_response_loss_not_optimizer_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'banks.sqlite'
            first=rl.claim_update([record()],path,'update1',expected_tasks={'task1':record()['task']})
            second=rl.claim_update([record()],path,'update1',expected_tasks={'task1':record()['task']})
            self.assertFalse(first['replayed']);self.assertTrue(second['replayed'])
            self.assertFalse(second['optimizer_replay_authorized'])
            with self.assertRaisesRegex(ValueError,'already consumed'):rl.claim_update([record()],path,'update2',expected_tasks={'task1':record()['task']})
            changed=record();changed['id']='other';changed=rl.seal_record(changed)
            with self.assertRaisesRegex(ValueError,'different batch'):rl.claim_update([changed],path,'update1',expected_tasks={'task1':record()['task']})


@unittest.skipIf(torch is None,'CPU torch available in the isolated training venv')
class TokenMcTests(unittest.TestCase):
    def test_exhaustive_iid_expectation_matches_independent_full_vocab_gradient(self):
        torch.set_num_threads(1)
        q=torch.tensor([.2,.5,.3],dtype=torch.float64)
        base=torch.tensor([.3,-.4,.8],dtype=torch.float64)
        beta=.17;reward=.8;action=1
        logp=torch.log_softmax(base,dim=-1)
        coefficient=reward-beta*(logp[action]-q[action].log())
        onehot=torch.tensor([0.,1.,0.],dtype=torch.float64)
        expected_gradient=-coefficient*(onehot-q)
        expected_value=-coefficient*(logp[action]-(q*logp).sum())
        for count in (1,2):
            observed_gradient=torch.zeros_like(base);observed_value=0.
            for draws in itertools.product(range(3),repeat=count):
                logits=base.clone().requires_grad_();lp=torch.log_softmax(logits,dim=-1)
                loss=rl.token_mc_loss(lp[action].reshape(1,1),q[action].log().reshape(1,1),
                    lp[list(draws)].reshape(1,1,count),torch.tensor([reward],dtype=torch.float64),
                    torch.tensor([[True]]),beta)
                gradient=torch.autograd.grad(loss,logits)[0]
                probability=math.prod(q[j].item() for j in draws)
                observed_gradient+=probability*gradient;observed_value+=probability*loss.item()
            self.assertTrue(torch.allclose(observed_gradient,expected_gradient,atol=1e-12,rtol=1e-12))
            self.assertAlmostEqual(observed_value,expected_value.item(),places=12)

    def test_feedback_and_sampler_detached_but_both_policy_terms_differentiate(self):
        action=torch.tensor([[-1.,-2.]],requires_grad=True)
        q=torch.tensor([[-1.5,-2.5]],requires_grad=True)
        aux=torch.tensor([[[-3.,-4.],[-2.,-3.]]],requires_grad=True)
        reward=torch.tensor([1.],requires_grad=True)
        loss=rl.token_mc_loss(action,q,aux,reward,torch.tensor([[True,True]]),.2)
        loss.backward()
        self.assertIsNone(q.grad);self.assertIsNone(reward.grad)
        self.assertTrue(torch.allclose(action.grad,torch.tensor([[-.9,-.9]])))
        self.assertTrue(torch.allclose(aux.grad,torch.full_like(aux,.45)))

    def test_sum_tokens_mean_responses_not_length_normalized(self):
        a=torch.full((2,3),-1.,requires_grad=True);q=a.detach().clone()
        aux=torch.full((2,3,1),-2.,requires_grad=True)
        mask=torch.tensor([[True,True,True],[True,False,False]])
        loss=rl.token_mc_loss(a,q,aux,torch.ones(2),mask)
        self.assertEqual(loss.item(),-2.)
        loss.backward();self.assertEqual(a.grad.tolist(),[[-.5,-.5,-.5],[-.5,0.,0.]])

    def test_masked_nan_inf_do_not_train_prompt_or_tool_slots(self):
        a=torch.tensor([[-1.,float('nan'),float('inf')]],requires_grad=True)
        q=a.detach().clone();aux=torch.tensor([[[-2.],[float('nan')],[float('inf')]]],requires_grad=True)
        loss=rl.token_mc_loss(a,q,aux,torch.ones(1),torch.tensor([[True,False,False]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(a.grad.tolist(),[[-1.,0.,0.]])
        self.assertEqual(aux.grad.flatten().tolist(),[1.,0.,0.])

    def test_shape_mask_active_nan_and_probability_validation(self):
        a=torch.tensor([[-1.,-2.]])
        good=[a,a.clone(),a.unsqueeze(-1),torch.ones(1),torch.tensor([[True,True]])]
        replacements=[(0,torch.tensor([[float('nan'),-2.]])),(1,torch.tensor([[1.,-2.]])),
            (2,torch.empty(1,2,0)),(3,torch.ones(2)),(4,torch.tensor([[1,1]])),
            (4,torch.tensor([[False,False]]))]
        for index,value in replacements:
            args=list(good);args[index]=value
            with self.subTest(index=index),self.assertRaises(ValueError):rl.token_mc_loss(*args)

    def test_sampler_records_actual_q_and_keeps_iid_duplicates(self):
        action=torch.Generator(device='cpu').manual_seed(73)
        aux=torch.Generator(device='cpu').manual_seed(74)
        logits=torch.tensor([-.4,.7])
        result=rl.sample_policy_tokens(logits,action,aux,mc_samples=8,temperature=.8)
        q=torch.softmax(logits.double()/.8,dim=-1)
        self.assertAlmostEqual(result['sampler_logp'],q[result['token_id']].log().item(),places=12)
        self.assertEqual(len(result['mc_token_ids']),8)
        self.assertLess(len(set(result['mc_token_ids'])),8)
        for token,logq in zip(result['mc_token_ids'],result['mc_sampler_logps']):
            self.assertAlmostEqual(logq,q[token].log().item(),places=12)

    def test_sampler_rejects_greedy_same_rng_and_numeric_support_loss(self):
        a=torch.Generator().manual_seed(1);b=torch.Generator().manual_seed(2)
        for logits,first,second,temp in [(torch.ones(2),a,a,1),(torch.ones(2),a,b,0),
                                       (torch.tensor([0.,-1e20]),a,b,1)]:
            with self.assertRaises(ValueError):rl.sample_policy_tokens(logits,first,second,temperature=temp)


if __name__=='__main__':unittest.main()
