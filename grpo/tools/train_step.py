"""Full-model FP32-master/BF16-autocast GRPO updates on frozen real rollouts.

This measures forward, head loss, backward and AdamW together. Generation is
separate and is NOT counted as an end-to-end RL throughput improvement.
Every backend starts the same model and consumes the same frozen PPO epochs.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
from transformers import AutoModelForCausalLM
from rltrain.batch import tensorize,response_hidden,PolicyVersions
from rltrain.objective import LossConfig
from rltrain.baselines import prepare_method,liger_class_and_provenance

ROOT=Path(__file__).resolve().parents[1]


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run(a,capture_callback=None):
    if not 0<=a.warmup<a.steps or a.microbatch<1:raise ValueError('invalid step/microbatch counts')
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.manual_seed(a.seed)
    path=Path(a.data)
    source=json.loads((path/'trajectories.json').read_text())
    if not source['nonzero_advantages']:raise ValueError('no policy-gradient signal in this frozen batch')
    values=torch.load(path/'logprobs.pt',map_location='cpu',weights_only=True)
    assert hashlib.sha256((path/'logprobs.pt').read_bytes()).hexdigest()==source['logprobs_sha256']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    batch=tensorize(source['records'],tokenizer.eos_token_id,device='cuda')
    old=values['old_logps'].to('cuda');ref=values['ref_logps'].to('cuda')
    old_hash=tensor_hash(old);ref_hash=tensor_hash(ref)
    versions=PolicyVersions(source['rollout_policy_version'],source['old_logprob_policy_version'],source['reference_policy_version'])
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.float32,attn_implementation='sdpa').to('cuda')
    # Qwen has no dropout in this configuration. Evaluation mode makes this
    # explicit while still enabling all gradients and optimizer updates.
    model.eval()
    if model.get_output_embeddings().bias is not None:
        raise ValueError('only bias-free Qwen linear heads are implemented')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,
                               weight_decay=0.,foreach=False,fused=True)
    config=LossConfig(vocab_chunk=a.vocab_chunk,chunk_tokens=a.chunk,strategy=a.strategy)
    b,r=batch['targets'].shape
    bindings=[]
    for start in range(0,b,a.microbatch):
        sl=slice(start,min(start+a.microbatch,b));m=sl.stop-sl.start
        adv=batch['advantages'][sl,None].expand(m,r).reshape(-1).contiguous()
        data=[None,None,batch['targets'][sl].reshape(-1),old[sl].reshape(-1),ref[sl].reshape(-1),
              adv,batch['weights'][sl].reshape(-1)]
        fn=prepare_method(a.method,data,config,m,r,normalization_scale=m/b)
        bindings.append((sl,data,fn))
    result={'utc':datetime.now(timezone.utc).isoformat(),'args':vars(a),
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((ROOT/'src').rglob('*.py')) if not p.name.startswith('._')},
            'data_sha256':hashlib.sha256((path/'trajectories.json').read_bytes()).hexdigest(),
            'old_logps_sha256':old_hash,'reference_logps_sha256':ref_hash,
            'precision':{'parameters':'float32','autocast':'bfloat16','head':'bfloat16','matmul':'high',
                         'compiled_backward_autocast':'off'},
            'optimizer':{'name':'AdamW','lr':1e-6,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'foreach':False,'fused':True},
            'backbone_gradient_checkpointing':False,'dropout_mode':'eval_with_gradients',
            'batch_size':b,'response_width':r,'active_response_tokens':source['active_response_tokens'],
            'nonzero_advantages':source['nonzero_advantages'],'steps':[]}
    if a.method=='liger':result['external_baseline']=liger_class_and_provenance()[1]
    output.write_text(json.dumps(result,indent=2))
    if capture_callback is not None:
        capture_callback('initial_parameters',model)
    for step in range(a.steps):
        versions.validate(step,max_epoch_lag=a.steps-1)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();first.record()
        total_loss=torch.zeros((),device='cuda')
        for sl,data,fn in bindings:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output_model=model.model(input_ids=batch['input_ids'][sl],
                                           attention_mask=batch['attention_mask'][sl],use_cache=False)
                h=response_hidden(output_model.last_hidden_state,batch['positions'][sl])
                data[0]=h.reshape(-1,h.shape[-1]).to(torch.bfloat16)
                data[1]=model.get_output_embeddings().weight.to(torch.bfloat16)
                loss=fn()
            loss.backward()
            total_loss+=loss.detach()
            data[0]=None;data[1]=None
            del output_model,h,loss
        # Timing validation snapshots and norm scans would bias a backend.
        # They are performed only in separate accuracy runs, not step timings.
        if a.save_gradients and step==0:
            torch.cuda.synchronize()
            snapshot={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
            torch.save(snapshot,a.save_gradients)
            del snapshot
        if capture_callback is not None and step==0:
            capture_callback('gradients',model)
        optimizer.step()
        last.record();torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        value=float(total_loss)
        if not torch.isfinite(total_loss):raise ValueError('nonfinite training loss')
        row={'step':step,'phase':'warmup' if step<a.warmup else 'measured','loss':value,
             'wall_ms':elapsed,'cuda_event_ms':first.elapsed_time(last),
             'allocated_before_bytes':before,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'actor_version_before':step,'actor_version_after':step+1,
             'old_policy_version':versions.old_logprob,'reference_policy_version':versions.reference,
             'timing_usable':not bool(a.save_gradients) and capture_callback is None}
        result['steps'].append(row);output.write_text(json.dumps(result,indent=2))
        print(json.dumps(row),flush=True)
    if tensor_hash(old)!=old_hash or tensor_hash(ref)!=ref_hash:
        raise ValueError('frozen old/reference log-probs mutated during updates')
    measured=[s for s in result['steps'] if s['phase']=='measured']
    result['median_wall_ms']=statistics.median(s['wall_ms'] for s in measured) if measured else None
    result['peak_allocated_bytes']=max(s['peak_allocated_bytes'] for s in measured) if measured else None
    if a.save_parameters:
        torch.save({name:p.detach().cpu().clone() for name,p in model.named_parameters()},a.save_parameters)
    if capture_callback is not None:
        capture_callback('parameters',model)
    result['frozen_logprobs_unchanged']=True;result['completed']=True
    output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['dense','compiled','checkpoint','manual_torch','fused','liger'],required=True)
    p.add_argument('--microbatch',type=int,default=8);p.add_argument('--steps',type=int,default=4)
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--seed',type=int,default=86)
    p.add_argument('--strategy',choices=['vocab','sequence'],default='vocab')
    p.add_argument('--vocab-chunk',type=int,default=32768);p.add_argument('--chunk',type=int,default=64)
    p.add_argument('--save-gradients');p.add_argument('--save-parameters')
    run(p.parse_args())
