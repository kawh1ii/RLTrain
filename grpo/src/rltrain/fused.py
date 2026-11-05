"""First-order custom autograd for the linear-head GRPO objective.

Vocabulary chunks bound temporary logits. Backward recomputes logits, writes
dlogits in place, then uses GEMMs. The [N,H] accumulator stays FP32 (FP64 for
finite-difference checks), and each [vocab-chunk,H] weight gradient is written
once. The older sequence-chunk implementation remains an explicit control.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from rltrain.objective import LossConfig, token_loss


def _torch_stats(logits, targets, old, ref, adv, weights, config):
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    x = logits.to(dtype)
    logz = x.logsumexp(-1)
    logp = x.gather(-1, targets[:, None]).squeeze(-1) - logz
    loss,derivative=_torch_loss_derivative(logp,old,ref,adv,weights,config)
    return loss,logz,derivative


def _torch_loss_derivative(logp,old,ref,adv,weights,config):
    loss = token_loss(logp, old, ref, adv, weights, config).sum()
    valid = weights > 0
    logp = torch.where(valid, logp, torch.zeros_like(logp))
    old = torch.where(valid, old, torch.zeros_like(old))
    ref = torch.where(valid, ref, torch.zeros_like(ref))
    adv = torch.where(valid, adv, torch.zeros_like(adv))
    ratio = (logp - old).exp()
    clipped = ((adv >= 0) & (ratio > 1 + config.epsilon)) | ((adv < 0) & (ratio < 1 - config.epsilon))
    derivative = torch.where(clipped, torch.zeros_like(ratio), -adv * ratio)
    if config.beta:
        derivative += config.beta * (1 - (ref - logp).exp())
    derivative = torch.where(valid, derivative * weights, torch.zeros_like(derivative))
    return loss, derivative


def _vocab_forward(hidden,weight,targets,old,ref,adv,weights,config,backend):
    dtype=torch.float64 if hidden.dtype==torch.float64 else torch.float32
    n=hidden.shape[0]
    maximum=torch.full((n,),-float("inf"),device=hidden.device,dtype=dtype)
    total=torch.zeros_like(maximum);selected=torch.zeros_like(maximum)
    for start in range(0,weight.shape[0],config.vocab_chunk):
        logits=F.linear(hidden,weight[start:start+config.vocab_chunk])
        if backend=="triton":
            from rltrain.kernels import update_vocab_stats
            update_vocab_stats(logits,targets,maximum,total,selected,start)
        else:
            x=logits.to(dtype)
            local=x.amax(-1)
            combined=torch.maximum(maximum,local)
            total=total*(maximum-combined).exp()+(x-combined[:,None]).exp().sum(-1)
            maximum=combined
            local_y=targets-start
            value=x.gather(-1,local_y.clamp(0,x.shape[1]-1)[:,None]).squeeze(-1)
            selected=torch.where((local_y>=0)&(local_y<x.shape[1]),value,selected)
    if backend=="triton":
        from rltrain.kernels import finish_vocab_loss
        return finish_vocab_loss(maximum,total,selected,old,ref,adv,weights,config)
    logz=maximum+total.log()
    loss,g=_torch_loss_derivative(selected-logz,old,ref,adv,weights,config)
    return loss,logz,g


def _vocab_backward(hidden,weight,targets,logz,derivative,scale,config,backend,need_h,need_w):
    dtype=torch.float64 if hidden.dtype==torch.float64 else torch.float32
    # Accumulate the SMALL [N,H] gradient, rather than repeatedly read/writing
    # a full FP32 [V,H] gradient. Each vocabulary row of dW is written once.
    gh=torch.zeros(hidden.shape,device=hidden.device,dtype=dtype) if need_h else None
    gw=torch.empty_like(weight) if need_w else None
    for start in range(0,weight.shape[0],config.vocab_chunk):
        sl=slice(start,start+config.vocab_chunk)
        logits=F.linear(hidden,weight[sl])
        if backend=="triton":
            from rltrain.kernels import logits_gradient_inplace
            grad=logits_gradient_inplace(logits,targets,logz,derivative,scale,start)
        else:
            grad=-(logits.to(dtype)-logz[:,None]).exp()
            local_y=targets-start
            valid=(local_y>=0)&(local_y<logits.shape[1])
            grad[torch.arange(len(grad),device=grad.device),local_y.clamp(0,logits.shape[1]-1)]+=valid
            grad=(grad*(derivative*scale)[:,None]).to(hidden.dtype)
        if need_w:
            torch.mm(grad.T,hidden,out=gw[sl])
        if need_h:
            if hidden.is_cuda and hidden.dtype==torch.bfloat16:
                torch.addmm(gh,grad,weight[sl],out_dtype=torch.float32,out=gh)
            else:
                gh.addmm_(grad.to(dtype),weight[sl].to(dtype))
    return gh.to(hidden.dtype) if gh is not None else None,gw


def validate_shapes(hidden, weight, targets, old, ref, advantages, weights):
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("expected hidden[N,H] and weight[V,H]")
    n = hidden.shape[0]
    if n < 1 or weight.shape[0] < 2:
        raise ValueError("empty input or invalid vocabulary")
    if hidden.dtype != weight.dtype or hidden.device != weight.device:
        raise ValueError("linear tensors must share dtype/device")
    if hidden.dtype not in (torch.float64, torch.float32, torch.bfloat16):
        raise ValueError("supported dtypes: FP64 reference, FP32, BF16")
    if targets.dtype != torch.int64:
        raise ValueError("targets must be int64")
    for name, tensor in zip(("targets", "old", "ref", "advantages", "weights"),
                            (targets, old, ref, advantages, weights)):
        if tensor.shape != (n,) or tensor.device != hidden.device:
            raise ValueError(name + " shape/device mismatch")
        if tensor.requires_grad:
            raise ValueError(name + " must be detached rollout metadata")
    expected = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    if any(t.dtype != expected for t in (old, ref, advantages, weights)):
        raise ValueError("metadata must be FP32 (FP64 for gradcheck)")


def validate_values(hidden,weight,targets,old,ref,advantages,weights):
    """Checked public path. Frozen-workload benchmarks call once before timing."""
    if not bool(((targets>=0)&(targets<weight.shape[0])).all()):
        raise ValueError("target outside vocabulary")
    if not bool((torch.isfinite(weights)&(weights>=0)).all()):
        raise ValueError("invalid token weights")
    valid=weights>0
    for tensor in (old,ref,advantages):
        if not bool(torch.isfinite(tensor[valid]).all()):
            raise ValueError("nonfinite active rollout metadata")
    if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("nonfinite linear input")


class _LinearGRPO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, old, ref, advantages, weights, config, backend):
        n = hidden.shape[0]
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        logz = torch.empty(n, dtype=dtype, device=hidden.device)
        derivative = torch.empty_like(logz)
        loss = torch.zeros((), dtype=dtype, device=hidden.device)
        if config.strategy=="vocab":
            loss,logz,derivative=_vocab_forward(hidden,weight,targets,old,ref,advantages,weights,config,backend)
        for start in range(0, n if config.strategy=="sequence" else 0, config.chunk_tokens):
            sl = slice(start, start + config.chunk_tokens)
            logits = F.linear(hidden[sl], weight)
            if backend == "triton":
                from rltrain.kernels import loss_stats
                value, z, g = loss_stats(logits, targets[sl], old[sl], ref[sl],
                                          advantages[sl], weights[sl], config)
            else:
                value, z, g = _torch_stats(logits, targets[sl], old[sl], ref[sl],
                                            advantages[sl], weights[sl], config)
            loss.add_(value)
            logz[sl] = z
            derivative[sl] = g
        ctx.save_for_backward(hidden, weight, targets, logz, derivative)
        ctx.config = config
        ctx.backend = backend
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, scale):
        hidden, weight, targets, logz, derivative = ctx.saved_tensors
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        need_h, need_w = ctx.needs_input_grad[:2]
        if ctx.config.strategy=="vocab":
            grad_h,grad_w=_vocab_backward(hidden,weight,targets,logz,derivative,scale,
                                          ctx.config,ctx.backend,need_h,need_w)
            return grad_h,grad_w,None,None,None,None,None,None,None
        grad_h = torch.empty_like(hidden) if need_h else None
        grad_w = torch.zeros(weight.shape, dtype=dtype, device=weight.device) if need_w else None
        for start in range(0, hidden.shape[0], ctx.config.chunk_tokens):
            sl = slice(start, start + ctx.config.chunk_tokens)
            logits = F.linear(hidden[sl], weight)
            if ctx.backend == "triton":
                from rltrain.kernels import logits_gradient_inplace
                grad_logits = logits_gradient_inplace(logits, targets[sl], logz[sl], derivative[sl], scale)
            else:
                probabilities = (logits.to(dtype) - logz[sl, None]).exp()
                grad = -probabilities
                grad[torch.arange(len(grad), device=grad.device), targets[sl]] += 1
                grad_logits = (grad * (derivative[sl] * scale)[:, None]).to(hidden.dtype)
            if need_h:
                grad_h[sl] = grad_logits @ weight
            if need_w:
                if hidden.is_cuda and hidden.dtype == torch.bfloat16:
                    # CUDA supports FP32 output from BF16 inputs without
                    # materializing FP32 copies of the vocabulary matrix.
                    torch.addmm(grad_w, grad_logits.T, hidden[sl], out_dtype=torch.float32, out=grad_w)
                else:
                    grad_w.addmm_(grad_logits.T.to(dtype), hidden[sl].to(dtype))
        return grad_h, grad_w.to(weight.dtype) if need_w else None, None, None, None, None, None, None, None


def fused_loss(hidden, weight, targets, old, ref, advantages, weights,
               config=LossConfig(), backend="auto",check_values=True):
    validate_shapes(hidden, weight, targets, old, ref, advantages, weights)
    if check_values:validate_values(hidden,weight,targets,old,ref,advantages,weights)
    if backend == "auto":
        backend = "triton" if hidden.is_cuda and hidden.dtype != torch.float64 else "torch"
    if backend not in ("torch", "triton"):
        raise ValueError("unknown backend")
    if backend == "triton" and (not hidden.is_cuda or hidden.dtype == torch.float64):
        raise ValueError("Triton supports CUDA FP32/BF16 only")
    # The kernel contracts are contiguous row-major, including selected metadata.
    return _LinearGRPO.apply(hidden.contiguous(), weight.contiguous(), targets.contiguous(),
                             old.contiguous(), ref.contiguous(), advantages.contiguous(),
                             weights.contiguous(), config, backend)
