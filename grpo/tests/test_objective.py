import math
import unittest
import torch
from rltrain.objective import (LossConfig, outcome_advantages, response_weights,
                               response_mask, reference_loss, checkpoint_loss)
from rltrain.fused import fused_loss, _torch_stats, _torch_loss_derivative


def inputs(tokens=7, hidden=5, vocab=17, dtype=torch.float64, device="cpu", seed=11):
    g = torch.Generator(device=device).manual_seed(seed)
    h = (torch.randn(tokens, hidden, generator=g, dtype=dtype, device=device) * .2).requires_grad_()
    w = (torch.randn(vocab, hidden, generator=g, dtype=dtype, device=device) * .2).requires_grad_()
    y = torch.randint(vocab, (tokens,), generator=g, device=device)
    with torch.no_grad():
        lp = (h @ w.T).to(torch.float64 if dtype == torch.float64 else torch.float32).log_softmax(-1).gather(-1,y[:,None]).squeeze(-1)
    meta = lp.dtype
    offsets = torch.tensor([-.7,.7,-.1,.1,-.7,.7,0.],device=device,dtype=meta)
    old = lp + offsets.repeat(math.ceil(tokens/7))[:tokens]
    ref = lp + torch.linspace(-.3,.3,tokens,device=device,dtype=meta)
    adv = torch.linspace(-1.,1.,tokens,device=device,dtype=meta)
    q = torch.ones(tokens,device=device,dtype=meta) / tokens
    return h,w,y,old,ref,adv,q


class ObjectiveTests(unittest.TestCase):
    def test_group_advantage_sample_std(self):
        a = outcome_advantages(torch.tensor([0.,2.,4.,4.]),[0,0,1,1])
        torch.testing.assert_close(a[:2],torch.tensor([-1.,1.])/math.sqrt(2),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(a[2:],torch.zeros(2))

    def test_both_signs_of_clipped_policy_gradient(self):
        lp=torch.tensor([1.5,1.5,.5,.5],dtype=torch.float64).log()
        old=torch.zeros_like(lp);a=torch.tensor([1.,-1.,1.,-1.],dtype=lp.dtype)
        loss,g=_torch_loss_derivative(lp,old,lp,a,torch.ones_like(lp),LossConfig(beta=0.))
        self.assertAlmostEqual(float(loss),.6)
        torch.testing.assert_close(g,torch.tensor([0.,1.5,-.5,0.],dtype=lp.dtype))

    def test_group_order(self):
        a = outcome_advantages(torch.tensor([2.,0.,4.,4.]),["a","b","a","b"])
        self.assertLess(a[0],0);self.assertGreater(a[2],0)

    def test_singleton_rejected(self):
        with self.assertRaises(ValueError):outcome_advantages(torch.tensor([1.]),[0])

    def test_nonfinite_reward_rejected(self):
        with self.assertRaises(ValueError):outcome_advantages(torch.tensor([float("nan"),1.]),[0,0])

    def test_first_eos_included(self):
        x=torch.tensor([[3,9,9,9],[3,4,5,6]])
        expected=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
        torch.testing.assert_close(response_mask(x,9),expected)

    def test_lengths_exclude_padding(self):
        x=torch.tensor([[3,4,5],[9,9,9]])
        torch.testing.assert_close(response_mask(x,9,torch.tensor([2,0])),
                                   torch.tensor([[1,1,0],[0,0,0]],dtype=torch.bool))

    def test_sequence_and_token_normalization(self):
        m=torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
        torch.testing.assert_close(response_weights(m),torch.tensor([[.25,.25,0],[.5,0,0]]))
        torch.testing.assert_close(response_weights(m,"token"),m.float()/3)

    def test_empty_response_and_empty_batch(self):
        q=response_weights(torch.zeros(2,3,dtype=torch.bool))
        self.assertEqual(float(q.sum()),0)
        self.assertEqual(response_weights(torch.zeros(0,3,dtype=torch.bool)).numel(),0)

    def test_full_optimizer_batch_weights_survive_microbatch_split(self):
        m=torch.tensor([[1,1,1],[1,0,0],[1,1,0]],dtype=torch.bool)
        weights=response_weights(m)
        self.assertAlmostEqual(float(weights.sum()),1.,places=6)
        self.assertNotEqual(float(weights[:1].sum()),float(response_weights(m[:1]).sum()))

    def test_accumulated_loss_and_gradients_use_global_normalizer(self):
        data=list(inputs(tokens=9))
        mask=torch.tensor([[1,1,1],[1,0,0],[1,1,0]],dtype=torch.bool)
        data[-1]=response_weights(mask).reshape(-1).double()
        full=reference_loss(*data)
        expected=torch.autograd.grad(full,data[:2])
        partial=full.new_zeros(())
        for start in range(0,9,3):
            sl=slice(start,start+3)
            sliced=[data[0][sl],data[1]]+[t[sl] for t in data[2:]]
            partial=partial+fused_loss(*sliced,backend='torch')
        actual=torch.autograd.grad(partial,data[:2])
        torch.testing.assert_close(partial,full)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)

    def compare(self,config=LossConfig(),tokens=7,zero=False,scale=1.):
        data=list(inputs(tokens=tokens))
        if zero:data[-1].zero_()
        ref=reference_loss(*data,config)
        expected=torch.autograd.grad(ref*scale,data[:2])
        got=fused_loss(*data,config,backend="torch")
        actual=torch.autograd.grad(got*scale,data[:2])
        torch.testing.assert_close(got,ref,atol=1e-12,rtol=1e-12)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)

    def test_custom_forward_and_all_gradients(self):self.compare()
    def test_nondivisible_chunk_tail(self):self.compare(LossConfig(strategy="sequence",chunk_tokens=3),tokens=11)
    def test_chunk_larger_than_tokens(self):self.compare(LossConfig(strategy="sequence",chunk_tokens=999))
    def test_nondivisible_vocab_tail(self):self.compare(LossConfig(vocab_chunk=7),tokens=11)
    def test_without_kl(self):self.compare(LossConfig(beta=0.,chunk_tokens=2))
    def test_all_padding_zero_gradients(self):self.compare(zero=True)
    def test_backward_loss_scaling(self):self.compare(scale=17.)

    def test_finite_difference_gradcheck(self):
        data=inputs(tokens=3,hidden=2,vocab=5)
        fn=lambda h,w:fused_loss(h,w,*data[2:],LossConfig(chunk_tokens=2),backend="torch")
        self.assertTrue(torch.autograd.gradcheck(fn,data[:2],eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_fp64_three_adam_updates_match_reference(self):
        data=inputs()
        reference=[t.detach().clone().requires_grad_() for t in data[:2]]
        custom=[t.detach().clone().requires_grad_() for t in data[:2]]
        opt_a=torch.optim.AdamW(reference,lr=1e-3,weight_decay=0.,foreach=False)
        opt_b=torch.optim.AdamW(custom,lr=1e-3,weight_decay=0.,foreach=False)
        for _ in range(3):
            opt_a.zero_grad();opt_b.zero_grad()
            reference_loss(*reference,*data[2:]).backward()
            fused_loss(*custom,*data[2:],backend='torch').backward()
            opt_a.step();opt_b.step()
        for a,b,initial in zip(reference,custom,data[:2]):
            torch.testing.assert_close(a-initial,b-initial,atol=1e-12,rtol=1e-10)
            self.assertGreater(float((a-initial).norm()),0.)

    def test_frozen_head(self):
        data=list(inputs());data[1].requires_grad_(False)
        reference=reference_loss(*data);expected=torch.autograd.grad(reference,data[0])[0]
        loss=fused_loss(*data,backend="torch");actual=torch.autograd.grad(loss,data[0])[0]
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-11)

    def test_weight_only_gradient(self):
        data=list(inputs());data[0].requires_grad_(False)
        a=torch.autograd.grad(reference_loss(*data),data[1])[0]
        b=torch.autograd.grad(fused_loss(*data,backend="torch"),data[1])[0]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)

    def test_masked_nonfinite_metadata(self):
        data=list(inputs());data[-1][0]=0;data[3][0]=float('nan');data[4][0]=float('inf')
        a=reference_loss(*data);b=fused_loss(*data,backend="torch")
        self.assertTrue(torch.isfinite(a));torch.testing.assert_close(a,b)

    def test_checkpoint_baseline_matches_reference(self):
        data=inputs();a=reference_loss(*data);b=checkpoint_loss(*data,LossConfig(chunk_tokens=3))
        torch.testing.assert_close(a,b)
        for x,y in zip(torch.autograd.grad(a,data[:2]),torch.autograd.grad(b,data[:2])):
            torch.testing.assert_close(x,y)

    def test_metadata_grad_rejected(self):
        data=list(inputs());data[3].requires_grad_(True)
        with self.assertRaises(ValueError):fused_loss(*data)

    def test_shape_rejected(self):
        data=list(inputs());data[2]=data[2][:-1]
        with self.assertRaises(ValueError):fused_loss(*data)

    def test_invalid_target_rejected_before_kernel(self):
        data=list(inputs());data[2][0]=999
        with self.assertRaises(ValueError):fused_loss(*data)

    def test_negative_weight_rejected(self):
        data=list(inputs());data[-1][0]=-.1
        with self.assertRaises(ValueError):fused_loss(*data)

    def test_invalid_config(self):
        for kw in ({'epsilon':1.},{'beta':-1.},{'chunk_tokens':0}):
            with self.assertRaises(ValueError):LossConfig(**kw)


@unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
class KernelTests(unittest.TestCase):
    def compare(self,dtype,tokens=11,vocab=1031,chunk=4,beta=.04,strategy="vocab",frozen=False):
        data=inputs(tokens=tokens,hidden=32,vocab=vocab,dtype=dtype,device="cuda")
        if frozen:data[1].requires_grad_(False)
        variables=[x for x in data[:2] if x.requires_grad]
        config=LossConfig(beta=beta,chunk_tokens=chunk,vocab_chunk=8192,strategy=strategy)
        ref=reference_loss(*data,config);gr=torch.autograd.grad(ref,variables)
        got=fused_loss(*data,config,backend="triton");gg=torch.autograd.grad(got,variables)
        torch.testing.assert_close(got,ref,atol=2e-5,rtol=2e-5)
        for a,b in zip(gg,gr):
            torch.testing.assert_close(a.float(),b.float(),atol=1e-5 if dtype==torch.bfloat16 else 1e-7,
                                       rtol=.03 if dtype==torch.bfloat16 else 2e-4)

    def test_fp32(self):self.compare(torch.float32)
    def test_bf16(self):self.compare(torch.bfloat16)
    def test_large_vocabulary_partial_tile(self):self.compare(torch.bfloat16,tokens=7,vocab=151936)
    def test_no_kl(self):self.compare(torch.float32,beta=0.)
    def test_sequence_control(self):self.compare(torch.bfloat16,strategy="sequence")
    def test_frozen_large_head(self):self.compare(torch.bfloat16,vocab=33001,frozen=True)


if __name__=="__main__":unittest.main()
