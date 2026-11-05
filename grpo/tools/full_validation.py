"""Single-GPU, frozen-source validation. Sequential stages, no silent failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True)
p.add_argument('--liger',required=True,help='pinned checkout/src directory')
p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['OMP_NUM_THREADS']='4';env['TOKENIZERS_PARALLELISM']='false'
env['RLTRAIN_LIGER_PATH']=str(Path(a.liger).resolve())
env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
data=Path(a.data).resolve()
source=json.loads((data/'trajectories.json').read_text())
if source['nonzero_advantages']<=0:raise ValueError('frozen data contains no policy-gradient signal')
fingerprint={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
             for folder in ('src','tools','tests','baselines') for f in sorted((ROOT/folder).rglob('*'))
             if f.is_file() and '__pycache__' not in str(f) and not f.name.startswith('._')}
fingerprint['pyproject.toml']=hashlib.sha256((ROOT/'pyproject.toml').read_bytes()).hexdigest()
freeze=out/'source-fingerprint.json'
if freeze.exists() and json.loads(freeze.read_text())!=fingerprint:
    raise ValueError('source changed; preserve old data and use a fresh output directory')
freeze.write_text(json.dumps(fingerprint,indent=2))
protocol={'model':a.model,'data':str(data),'data_sha256':hashlib.sha256((data/'trajectories.json').read_bytes()).hexdigest(),
          'micro_seeds':[82,83,84],'model_repeats':[86,87,88],
          'fixed_strategy':'vocab','fixed_vocab_chunk':32768,
          'primary_micro_shapes':[[4,128,896,151936],[8,256,896,151936],[8,512,896,151936]],
          'micro_methods':['dense','compiled','checkpoint','manual_torch','fused','liger'],
          'model_methods':['dense','compiled','fused','liger'],
          'model_microbatch':8,'model_steps':4,'warmup_steps':1,
          'note':'fixed trajectory batch and pretrained initialization; repeated model runs are timing replicates, not new datasets'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))


def stage(name,cmd):
    done=out/(name+'.done')
    command=[sys.executable]+cmd
    if done.exists():
        existing=json.loads(done.read_text())
        if existing['source_fingerprint']!=fingerprint or existing['command']!=command:
            raise ValueError('stage marker differs: '+name)
        print('SKIP',name,flush=True);return
    print('STAGE',name,flush=True)
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Refuse a source mutation between stages, even if tests happened to pass.
    for file,digest in fingerprint.items():
        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest()!=digest:
            raise ValueError('source changed during validation: '+file)
    done.write_text(json.dumps({'command':command,'source_fingerprint':fingerprint},indent=2))


stage('unit-tests',['-m','unittest','discover','-s',str(ROOT/'tests'),'-v'])
stage('model-accuracy',[str(ROOT/'tools/compare_training.py'),'--model',a.model,'--data',str(data),
                        '--output',str(out/'accuracy'),'--steps','3'])
for seed in (82,83,84):
    for b,r,h,v in protocol['primary_micro_shapes']:
        name=f'loss-n{b*r}-h{h}-s{seed}'
        stage(name,[str(ROOT/'tools/benchmark_loss.py'),'--output',str(out/(name+'.json')),
                    '--batch',str(b),'--response',str(r),'--hidden',str(h),'--vocab',str(v),'--seed',str(seed)])
for h,frozen in ((2048,False),(896,True)):
    name=f'loss-extension-h{h}-frozen{int(frozen)}'
    cmd=[str(ROOT/'tools/benchmark_loss.py'),'--output',str(out/(name+'.json')),
         '--batch','4' if h==2048 else '8','--response','128' if h==2048 else '256',
         '--hidden',str(h),'--seed','84']
    if frozen:cmd.append('--frozen-head')
    stage(name,cmd)
for strategy in ('sequence','vocab'):
    name='ablation-'+strategy
    stage(name,[str(ROOT/'tools/benchmark_loss.py'),'--output',str(out/(name+'.json')),
                '--seed','81','--strategy',strategy,'--methods','dense,compiled,manual_torch,fused'])
for seed in (86,87,88):
    methods=protocol['model_methods'];offset=seed%len(methods)
    for method in methods[offset:]+methods[:offset]:
        name=f'train-{method}-s{seed}'
        stage(name,[str(ROOT/'tools/train_step.py'),'--model',a.model,'--data',str(data),
                    '--output',str(out/(name+'.json')),'--method',method,'--seed',str(seed)])
for method in ('dense','fused'):
    name='train-micro4-'+method
    stage(name,[str(ROOT/'tools/train_step.py'),'--model',a.model,'--data',str(data),
                '--output',str(out/(name+'.json')),'--method',method,'--microbatch','4'])
print('ALL_STAGES_COMPLETED',flush=True)
