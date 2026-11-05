"""Independent, differentiable GRPO reference and response semantics.

Explicit objective: token-level clipped importance ratio + k3 KL estimator,
then either sequence mean or token mean. No dual clipping, entropy bonus,
reference-policy gradients, or rollout-IS correction are silently enabled.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class LossConfig:
    epsilon: float = 0.2
    beta: float = 0.04
    chunk_tokens: int = 64
    strategy: str = "vocab"
    vocab_chunk: int = 32768

    def __post_init__(self):
        if not 0 < self.epsilon < 1 or not 0 <= self.beta < float("inf"):
            raise ValueError("invalid clipping or KL coefficient")
        if not isinstance(self.chunk_tokens, int) or self.chunk_tokens < 1:
            raise ValueError("chunk_tokens must be positive")
        if self.strategy not in ("sequence", "vocab") or self.vocab_chunk < 1:
            raise ValueError("invalid chunk strategy")


def outcome_advantages(rewards, groups, epsilon=1e-6):
    """Sample std within groups. Singleton groups are rejected explicitly."""
    if rewards.ndim != 1 or len(groups) != rewards.numel():
        raise ValueError("rewards/groups shape mismatch")
    if not torch.isfinite(rewards).all():
        raise ValueError("nonfinite reward")
    out = torch.zeros_like(rewards)
    buckets = {}
    for index, group in enumerate(groups):
        buckets.setdefault(group, []).append(index)
    for indices in buckets.values():
        if len(indices) < 2:
            raise ValueError("GRPO needs at least two outcomes per group")
        score = rewards[indices]
        out[indices] = (score - score.mean()) / (score.std(unbiased=True) + epsilon)
    return out.detach()


def response_weights(mask, reduction="sequence"):
    """Return additive per-token weights; empty responses contribute zero.

    For gradient accumulation, construct weights for the FULL optimizer batch
    first, then slice them. Renormalizing every microbatch changes the objective.
    """
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be a [batch,response] bool tensor")
    m = mask.float()
    if reduction == "sequence":
        return m / m.sum(-1, keepdim=True).clamp(min=1) / max(1, mask.shape[0])
    if reduction == "token":
        return m / m.sum().clamp(min=1)
    raise ValueError("unknown reduction")


def response_mask(token_ids, eos_id, lengths=None):
    """Include the FIRST EOS in loss; exclude padding and all later EOS."""
    if token_ids.ndim != 2:
        raise ValueError("expected [batch,response] IDs")
    eos = token_ids.eq(eos_id)
    mask = (eos.cumsum(-1) - eos.long()).eq(0)
    if lengths is not None:
        if lengths.shape != token_ids.shape[:1]:
            raise ValueError("invalid lengths")
        mask &= torch.arange(token_ids.shape[1], device=token_ids.device)[None, :] < lengths[:, None]
    return mask


def token_loss(logp, old, ref, advantages, weights, config):
    valid = weights > 0
    # Mask metadata BEFORE exponentiation, so arbitrary padded metadata cannot
    # produce NaN*0. Values at active tokens must be finite.
    lp = torch.where(valid, logp, torch.zeros_like(logp))
    old = torch.where(valid, old, torch.zeros_like(old))
    ref = torch.where(valid, ref, torch.zeros_like(ref))
    adv = torch.where(valid, advantages, torch.zeros_like(advantages))
    ratio = (lp - old).exp()
    clipped = ratio.clamp(1 - config.epsilon, 1 + config.epsilon)
    pg = -torch.minimum(ratio * adv, clipped * adv)
    delta = ref - lp
    kl = delta.exp() - delta - 1 if config.beta else torch.zeros_like(lp)
    return (pg + config.beta * kl) * weights


def reference_loss(hidden, weight, targets, old, ref, advantages, weights,
                   config=LossConfig()):
    logits = F.linear(hidden, weight)
    # Float64 remains float64 for finite-difference gradient checking.
    logits = logits if logits.dtype == torch.float64 else logits.float()
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
    return token_loss(logp, old, ref, advantages, weights, config).sum()


def checkpoint_loss(hidden, weight, targets, old, ref, advantages, weights,
                    config=LossConfig()):
    """Strong pure-PyTorch sequence-chunk/recompute baseline, same weights."""
    loss = hidden.new_zeros((), dtype=torch.float64 if hidden.dtype == torch.float64 else torch.float32)
    for start in range(0, hidden.shape[0], config.chunk_tokens):
        sl = slice(start, start + config.chunk_tokens)
        def chunk(h, w, y, o, r, a, q):
            return reference_loss(h, w, y, o, r, a, q, config)
        loss = loss + checkpoint(chunk, hidden[sl], weight, targets[sl], old[sl],
                                 ref[sl], advantages[sl], weights[sl], use_reentrant=False)
    return loss
