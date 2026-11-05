"""Paired forward+backward wall time and allocated-memory peak, no optimizer.

Dense, compiled, checkpoint and manual gradients are separate controls. Liger
is pinned externally and receives the same GRPO reduction and coefficients.
Never report a forward-only speed comparison against its fused fwd+bwd API.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
import torch

from rltrain.baselines import prepare_method,liger_class_and_provenance
from rltrain.objective import LossConfig, response_weights
from rltrain.fused import validate_values

ROOT=Path(__file__).resolve().parents[1]


def tensor_hash(t):
    return hashlib.sha256(memoryview(t.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def make_data(a):
    torch.manual_seed(a.seed)
    n=a.batch*a.response
    h=(torch.randn(n,a.hidden,device="cuda",dtype=torch.bfloat16)*.2).requires_grad_()
    w=(torch.randn(a.vocab,a.hidden,device="cuda",dtype=torch.bfloat16)*.02).requires_grad_(not a.frozen_head)
    y=torch.randint(a.vocab,(n,),device="cuda")
    lp=[]
    with torch.no_grad():
        for start in range(0,n,64):
            x=(h[start:start+64]@w.T).float()
            labels=y[start:start+64]
            lp.append(x.gather(-1,labels[:,None]).squeeze(-1)-x.logsumexp(-1))
    lp=torch.cat(lp)
    # Offsets straddle clipping branches and stay away from their kinks.
    offsets=torch.tensor([-.7,-.3,-.1,.1,.3,.7],device="cuda").repeat((n+5)//6)[:n]
    old=(lp+offsets).detach()
    ref=(lp+torch.linspace(-.3,.3,n,device="cuda")).detach()
    seq_adv=torch.linspace(-1.,1.,a.batch,device="cuda")
    adv=seq_adv[:,None].expand(a.batch,a.response).reshape(-1).contiguous()
    mask=torch.ones(a.batch,a.response,dtype=torch.bool,device="cuda")
    weights=response_weights(mask).reshape(-1)
    return h,w,y,old,ref,adv,weights


def gradients(data,fn):
    h,w=data[:2]
    h.grad=None;w.grad=None
    loss=fn()
    loss.backward()
    result={"loss":float(loss.detach()),"hidden":h.grad.detach().float().cpu(),
            "weight":w.grad.detach().float().cpu() if w.requires_grad else None}
    h.grad=None;w.grad=None
    return result


def grad_metrics(a,b):
    a=a.reshape(-1);b=b.reshape(-1)
    aa=bb=ab=dd=maximum=0.
    for start in range(0,a.numel(),1_000_000):
        x=a[start:start+1_000_000].double();y=b[start:start+1_000_000].double();d=x-y
        aa+=float(x.square().sum());bb+=float(y.square().sum());ab+=float((x*y).sum());dd+=float(d.square().sum())
        maximum=max(maximum,float(d.abs().max()))
    return {"relative_l2":(dd/max(bb,1e-30))**.5,
            "cosine":ab/max((aa*bb)**.5,1e-30),"max_absolute":maximum}


def run(a):
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    config=LossConfig(chunk_tokens=a.chunk,strategy=a.strategy,vocab_chunk=a.vocab_chunk)
    methods=a.methods.split(",")
    methods=methods[a.seed%len(methods):]+methods[:a.seed%len(methods)]
    data=make_data(a)
    validate_values(*data)
    inputs_hash={name:tensor_hash(t) for name,t in zip(('hidden','weight','targets','old','reference','advantages','weights'),data)}
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/"src").rglob("*.py")) if not p.name.startswith("._")}
    result={"utc":datetime.now(timezone.utc).isoformat(),"args":vars(a),
            "environment":{"torch":torch.__version__,"triton":importlib.metadata.version("triton"),
                           "gpu":torch.cuda.get_device_name(0),"float32_matmul_precision":"high"},
            "source_hashes":hashes,"input_hashes":inputs_hash,"methods":{}}
    if 'liger' in methods:result['external_baseline']=liger_class_and_provenance()[1]
    output.write_text(json.dumps(result,indent=2))
    reference=gradients(data,prepare_method("dense",data,config,a.batch,a.response))
    for method in methods:
        print("METHOD",method,flush=True)
        fn=prepare_method(method,data,config,a.batch,a.response)
        candidate=gradients(data,fn)
        accuracy={"loss_absolute":abs(candidate["loss"]-reference["loss"]),
                  "loss":candidate["loss"],"reference_loss":reference["loss"]}
        for field in ("hidden","weight"):
            if candidate[field] is not None:
                accuracy[field]=grad_metrics(candidate[field],reference[field])
        accuracy["passed"]=accuracy["loss_absolute"] <= 2e-4
        for field in ("hidden","weight"):
            if field in accuracy:
                accuracy["passed"] &= accuracy[field]["relative_l2"]<=.01 and accuracy[field]["cosine"]>=.9999
        del candidate
        if not accuracy["passed"]:
            result["methods"][method]={"accuracy":accuracy,"status":"accuracy_failed"}
            output.write_text(json.dumps(result,indent=2))
            raise AssertionError((method,accuracy))
        for _ in range(a.warmup):
            data[0].grad=None;data[1].grad=None
            loss=fn();loss.backward()
        del loss
        data[0].grad=None;data[1].grad=None
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        before=torch.cuda.memory_allocated()
        walls=[];gpu_times=[];peaks=[];reserved=[]
        for _ in range(a.repetitions):
            data[0].grad=None;data[1].grad=None
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();start.record()
            loss=fn();loss.backward()
            end.record();torch.cuda.synchronize()
            walls.append((time.perf_counter()-wall)*1000);gpu_times.append(start.elapsed_time(end))
            peaks.append(torch.cuda.max_memory_allocated());reserved.append(torch.cuda.max_memory_reserved())
            del loss
        result["methods"][method]={"status":"passed","accuracy":accuracy,
                  "wall_ms":walls,"cuda_event_ms":gpu_times,
                  "median_wall_ms":statistics.median(walls),
                  "allocated_before_bytes":before,"peak_allocated_bytes":max(peaks),
                  "peak_reserved_bytes":max(reserved),"all_peak_allocated_bytes":peaks}
        output.write_text(json.dumps(result,indent=2))
        print(json.dumps({"method":method,"median_wall_ms":statistics.median(walls),
                          "peak_allocated_bytes":max(peaks)}),flush=True)
        data[0].grad=None;data[1].grad=None
    result["completed"]=True
    after={name:tensor_hash(t) for name,t in zip(('hidden','weight','targets','old','reference','advantages','weights'),data)}
    if after!=inputs_hash:raise ValueError('benchmark inputs mutated by a backend')
    result['inputs_unchanged']=True
    output.write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--response",type=int,default=128)
    p.add_argument("--hidden",type=int,default=896)
    p.add_argument("--vocab",type=int,default=151936)
    p.add_argument("--chunk",type=int,default=64)
    p.add_argument("--strategy",choices=["sequence","vocab"],default="vocab")
    p.add_argument("--vocab-chunk",type=int,default=32768)
    p.add_argument("--seed",type=int,default=81)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--repetitions",type=int,default=7)
    p.add_argument("--methods",default="dense,compiled,checkpoint,manual_torch,fused,liger")
    p.add_argument("--frozen-head",action="store_true")
    run(p.parse_args())
