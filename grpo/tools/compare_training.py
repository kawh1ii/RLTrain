"""Compare ALL actual model parameter gradients/updates, without huge files.

Snapshots stay in CPU RAM while processing backends sequentially. The report
retains per-parameter errors, shapes and hashes; GPU timings from these accuracy
runs are explicitly unusable because CPU snapshot/comparison time is included.
"""
import argparse
from argparse import Namespace
import gc
import hashlib
import json
from pathlib import Path
import torch
from train_step import run

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True)
p.add_argument('--output',required=True);p.add_argument('--microbatch',type=int,default=8)
p.add_argument('--methods',default='dense,compiled,fused,liger');p.add_argument('--steps',type=int,default=3)
a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
if a.methods.split(',')[0]!='dense':raise ValueError('dense reference must run first')
if (root/'comparison.json').exists():raise FileExistsError(root)
baseline={};all_metrics={}


def compare_tensor(value,reference):
    # Chunk the large lm_head tensor to keep CPU temporary memory bounded.
    x=value.reshape(-1);y=reference.reshape(-1)
    aa=bb=ab=dd=0.;maximum=0.
    for start in range(0,x.numel(),1_000_000):
        vx=x[start:start+1_000_000].double();vy=y[start:start+1_000_000].double()
        diff=vx-vy
        aa+=float(vx.square().sum());bb+=float(vy.square().sum());ab+=float((vx*vy).sum());dd+=float(diff.square().sum())
        maximum=max(maximum,float(diff.abs().max()))
    return {'candidate_squared_norm':aa,'reference_squared_norm':bb,'dot':ab,
            'difference_squared_norm':dd,'max_absolute':maximum,
            'relative_l2':(dd/max(bb,1e-30))**.5,
            'cosine':ab/max((aa*bb)**.5,1e-30)}


for method in a.methods.split(','):
    records={}
    def capture(kind,model):
        metrics={};snapshot={};fingerprints={}
        for name,param in model.named_parameters():
            value=param.grad if kind=='gradients' else param
            if value is None:continue
            cpu=value.detach().cpu()
            # Compare the UPDATE itself, not just final weights dominated by
            # the pretrained initialization. A no-op optimizer must fail.
            if kind=='parameters':cpu=cpu-baseline['initial_parameters'][name]
            if method=='dense':
                snapshot[name]=cpu.clone()
                fingerprints[name]={'shape':list(cpu.shape),'sha256':hashlib.sha256(memoryview(cpu.contiguous().numpy())).hexdigest()}
            else:
                if name not in baseline[kind]:raise ValueError('parameter set changed')
                metrics[name]={'shape':list(cpu.shape),**compare_tensor(cpu,baseline[kind][name])}
        if method=='dense':
            baseline[kind]=snapshot
            total=sum(v.numel() for v in snapshot.values())
            norm=sum(float(v.double().square().sum()) for v in snapshot.values())**.5
            records[kind]={'parameters':len(snapshot),'elements':total,'norm':norm,'fingerprints':fingerprints,
                           'quantity':'optimizer_update' if kind=='parameters' else kind}
        else:
            if set(metrics)!=set(baseline[kind]):
                raise ValueError('missing gradient/parameter coverage: '+kind)
            sums={field:sum(v[field] for v in metrics.values()) for field in
                  ['candidate_squared_norm','reference_squared_norm','dot','difference_squared_norm']}
            rel=(sums['difference_squared_norm']/max(sums['reference_squared_norm'],1e-30))**.5
            cos=sums['dot']/max((sums['candidate_squared_norm']*sums['reference_squared_norm'])**.5,1e-30)
            if kind=='initial_parameters':passed=sums['difference_squared_norm']==0.
            elif kind=='gradients':passed=rel<.03 and cos>.999
            else:passed=rel<.1 and cos>.995
            records[kind]={'relative_l2':rel,'cosine':cos,'per_parameter':metrics,'passed':passed,
                           'quantity':'optimizer_update' if kind=='parameters' else kind,
                           'required_gate':kind!='parameters',
                           'nonzero_update':sums['candidate_squared_norm']>0 if kind=='parameters' else None}
    run(Namespace(model=a.model,data=a.data,output=str(root/(method+'.json')),method=method,
                  microbatch=a.microbatch,steps=a.steps,warmup=0,seed=86,strategy='vocab',
                  vocab_chunk=32768,chunk=64,save_gradients=None,save_parameters=None),capture_callback=capture)
    all_metrics[method]=records
    (root/'comparison.json').write_text(json.dumps({'methods':all_metrics,'completed':False},indent=2))
    if method!='dense' and not all(v['passed'] for k,v in records.items() if k!='parameters'):
        raise AssertionError((method,{k:(v['relative_l2'],v['cosine']) for k,v in records.items()}))
    if method!='dense' and not records['parameters']['nonzero_update']:
        raise ValueError('candidate optimizer did not perform an update')
    if method=='dense' and not records['gradients']['norm']>0:raise ValueError('no nonzero training gradient')
    if method=='dense' and not records['parameters']['norm']>0:raise ValueError('optimizer did not update parameters')
    gc.collect();torch.cuda.empty_cache()
(root/'comparison.json').write_text(json.dumps({'methods':all_metrics,'completed':True,
    'all_required_checks_passed':True,
    'all_bf16_update_tolerances_passed':all(s['parameters']['passed'] for m,s in all_metrics.items() if m!='dense'),
    'update_tolerance':{'relative_l2':.1,'cosine':.995,
                        'role':'diagnostic after eager-vs-compiled BF16 control failed; unchanged original threshold'},
    'scope':'FP64 formula/update equivalence is a unit gate; real mixed-precision gradients use declared tolerance; multi-step BF16 updates are characterized, not claimed equivalent'},indent=2))
print('REQUIRED_MODEL_CHECKS_PASSED; BF16_UPDATE_DRIFT_REPORTED',flush=True)
