"""Own Triton reductions for a token chunk's large-vocabulary linear head.

GEMMs remain PyTorch/cuBLAS. Split row reductions avoid a 262144-element
single-program softmax for a 151936-token vocabulary. No full FP32 probability
or log-softmax matrix is materialized or saved across forward/backward.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial_stats(X, M, S, V: tl.constexpr, PARTS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    cols = part * TILE + tl.arange(0, TILE)
    x = tl.load(X + row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
    m = tl.max(x, 0)
    s = tl.sum(tl.exp(x - m), 0)
    tl.store(M + row * PARTS + part, m)
    tl.store(S + row * PARTS + part, s)


@triton.jit
def _loss_and_derivative(X, M, S, Y, OLD, REF, ADV, WEIGHTS, LSE, G, LOSS,
                         V: tl.constexpr, PARTS: tl.constexpr,
                         REDUCE: tl.constexpr, EPS: tl.constexpr, BETA: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, REDUCE)
    m = tl.load(M + row * PARTS + cols, cols < PARTS, other=-float("inf"))
    s = tl.load(S + row * PARTS + cols, cols < PARTS, other=0.0)
    maximum = tl.max(m, 0)
    normalizer = maximum + tl.log(tl.sum(s * tl.exp(m - maximum), 0))
    y = tl.load(Y + row)
    selected = tl.load(X + row * V + y).to(tl.float32)
    q = tl.load(WEIGHTS + row)
    valid = q > 0
    logp = tl.where(valid, selected - normalizer, 0.0)
    old = tl.where(valid, tl.load(OLD + row), 0.0)
    a = tl.where(valid, tl.load(ADV + row), 0.0)
    ratio = tl.exp(logp - old)
    clipped = tl.minimum(tl.maximum(ratio, 1.0 - EPS), 1.0 + EPS)
    pg = -tl.minimum(ratio * a, clipped * a)
    clipped_branch = ((a >= 0) & (ratio > 1.0 + EPS)) | ((a < 0) & (ratio < 1.0 - EPS))
    derivative = tl.where(clipped_branch, 0.0, -a * ratio)
    loss = pg
    if BETA != 0.0:
        ref = tl.where(valid, tl.load(REF + row), 0.0)
        delta = ref - logp
        exp_delta = tl.exp(delta)
        loss += BETA * (exp_delta - delta - 1.0)
        derivative += BETA * (1.0 - exp_delta)
    tl.store(LSE + row, normalizer)
    tl.store(G + row, tl.where(valid, derivative * q, 0.0))
    tl.store(LOSS + row, tl.where(valid, loss * q, 0.0))


@triton.jit
def _logits_gradient_inplace(X, Y, LSE, G, SCALE, V: tl.constexpr, TILE: tl.constexpr,
                            VOCAB_START):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    x = tl.load(X + row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
    y = tl.load(Y + row)
    logz = tl.load(LSE + row)
    g = tl.load(G + row) * tl.load(SCALE)
    value = g * ((cols + VOCAB_START == y).to(tl.float32) - tl.exp(x - logz))
    value = tl.where(g == 0.0, 0.0, value)
    tl.store(X + row * V + cols, value, cols < V)


def loss_stats(logits, targets, old, ref, adv, weights, config):
    rows, vocab = logits.shape
    parts = triton.cdiv(vocab, 4096)
    maxima = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    logz = torch.empty(rows, device=logits.device, dtype=torch.float32)
    derivative = torch.empty_like(logz)
    losses = torch.empty_like(logz)
    _partial_stats[(rows, parts)](logits, maxima, sums, vocab, parts, 4096)
    _loss_and_derivative[(rows,)](logits, maxima, sums, targets, old, ref, adv, weights,
                                  logz, derivative, losses, vocab, parts,
                                  triton.next_power_of_2(parts), config.epsilon, config.beta)
    return losses.sum(), logz, derivative


def logits_gradient_inplace(logits, targets, logz, derivative, scale, vocab_start=0):
    rows, vocab = logits.shape
    _logits_gradient_inplace[(rows, triton.cdiv(vocab, 4096))](
        logits, targets, logz, derivative, scale, vocab, 4096, vocab_start)
    return logits


@triton.jit
def _merge_vocab_stats(X, M, S, Y, RUN_M, RUN_S, SELECTED,
                       V: tl.constexpr, PARTS: tl.constexpr, REDUCE: tl.constexpr,
                       VOCAB_START):
    row=tl.program_id(0).to(tl.int64)
    cols=tl.arange(0,REDUCE)
    m=tl.load(M+row*PARTS+cols,cols<PARTS,other=-float("inf"))
    s=tl.load(S+row*PARTS+cols,cols<PARTS,other=0.0)
    local_max=tl.max(m,0)
    local_sum=tl.sum(s*tl.exp(m-local_max),0)
    previous_max=tl.load(RUN_M+row)
    previous_sum=tl.load(RUN_S+row)
    maximum=tl.maximum(previous_max,local_max)
    total=previous_sum*tl.exp(previous_max-maximum)+local_sum*tl.exp(local_max-maximum)
    tl.store(RUN_M+row,maximum)
    tl.store(RUN_S+row,total)
    y=tl.load(Y+row)-VOCAB_START
    selected=tl.load(X+row*V+y,(y>=0)&(y<V),other=0.0).to(tl.float32)
    tl.store(SELECTED+row,selected,(y>=0)&(y<V))


@triton.jit
def _finish_vocab_loss(M, S, SELECTED, OLD, REF, ADV, WEIGHTS, LSE, G, LOSS,
                        N: tl.constexpr, BLOCK: tl.constexpr,
                        EPS: tl.constexpr, BETA: tl.constexpr):
    rows=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=rows<N
    z=tl.load(M+rows,mask,other=0.0)+tl.log(tl.load(S+rows,mask,other=1.0))
    q=tl.load(WEIGHTS+rows,mask,other=0.0)
    valid=mask&(q>0)
    lp=tl.where(valid,tl.load(SELECTED+rows,mask,other=0.0)-z,0.0)
    old=tl.where(valid,tl.load(OLD+rows,mask,other=0.0),0.0)
    a=tl.where(valid,tl.load(ADV+rows,mask,other=0.0),0.0)
    ratio=tl.exp(lp-old)
    clipped=tl.minimum(tl.maximum(ratio,1-EPS),1+EPS)
    loss=-tl.minimum(ratio*a,clipped*a)
    clip_branch=((a>=0)&(ratio>1+EPS))|((a<0)&(ratio<1-EPS))
    derivative=tl.where(clip_branch,0.,-a*ratio)
    if BETA!=0.0:
        delta=tl.where(valid,tl.load(REF+rows,mask,other=0.0),0.0)-lp
        exp_delta=tl.exp(delta)
        loss+=BETA*(exp_delta-delta-1.)
        derivative+=BETA*(1.-exp_delta)
    tl.store(LSE+rows,z,mask)
    tl.store(G+rows,tl.where(valid,derivative*q,0.),mask)
    tl.store(LOSS+rows,tl.where(valid,loss*q,0.),mask)


def update_vocab_stats(logits,targets,maximum,total,selected,start):
    rows,vocab=logits.shape
    parts=triton.cdiv(vocab,4096)
    m=torch.empty((rows,parts),device=logits.device,dtype=torch.float32)
    s=torch.empty_like(m)
    _partial_stats[(rows,parts)](logits,m,s,vocab,parts,4096)
    _merge_vocab_stats[(rows,)](logits,m,s,targets,maximum,total,selected,vocab,parts,
                               triton.next_power_of_2(parts),start)


def finish_vocab_loss(maximum,total,selected,old,ref,adv,weights,config):
    n=len(selected)
    z=torch.empty_like(selected);g=torch.empty_like(selected);loss=torch.empty_like(selected)
    _finish_vocab_loss[(triton.cdiv(n,256),)](maximum,total,selected,old,ref,adv,weights,
                                             z,g,loss,n,256,config.epsilon,config.beta)
    return loss.sum(),z,g
