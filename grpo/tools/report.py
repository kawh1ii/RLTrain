"""Recompute final tables and audit ALL frozen stage records without a GPU."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

PROJECT=Path(__file__).resolve().parents[1]


def report(root):
    freeze=json.loads((root/'source-fingerprint.json').read_text())
    for file,digest in freeze.items():
        if hashlib.sha256((PROJECT/file).read_bytes()).hexdigest()!=digest:
            raise ValueError('current release differs from measured source: '+file)
    markers=list(root.glob('*.done'))
    if len(markers)!=29:raise ValueError(f'expected 29 completed stages, found {len(markers)}')
    for marker in markers:
        if json.loads(marker.read_text())['source_fingerprint']!=freeze:
            raise ValueError('stage source mismatch: '+str(marker))
    unit=(root/'unit-tests.log').read_text()
    match=re.search(r'Ran (\d+) tests',unit)
    if not match or '\nOK\n' not in unit or 'skipped' in unit:
        raise ValueError('unit suite did not fully pass')
    accuracy=json.loads((root/'accuracy/comparison.json').read_text())
    if not accuracy.get('completed'):raise ValueError('model accuracy incomplete')
    for mode,stats in accuracy['methods'].items():
        if mode!='dense' and not all(item['passed'] for kind,item in stats.items() if kind!='parameters'):
            raise ValueError('model gradient/update check failed: '+mode)
        if mode!='dense' and not stats['parameters']['nonzero_update']:
            raise ValueError('no actual optimizer update')
    runtime={name:digest for name,digest in freeze.items() if name.startswith('src/') and name.endswith('.py')}
    raw_files=list(root.glob('loss-*.json'))+list(root.glob('ablation-*.json'))+list(root.glob('train-*.json'))+list((root/'accuracy').glob('*.json'))
    raw_files=[p for p in raw_files if p.name!='comparison.json']
    for file in raw_files:
        d=json.loads(file.read_text())
        if not d.get('completed') or d['source_hashes']!=runtime:
            raise ValueError('incomplete run or runtime fingerprint mismatch: '+str(file))
        if 'methods' in d:
            if not d.get('inputs_unchanged'):raise ValueError('mutated operator inputs')
            for mode,item in d['methods'].items():
                if item['status']!='passed' or not item['accuracy']['passed']:
                    raise ValueError('operator correctness failed: '+str(file))
                if len(item['wall_ms'])!=d['args']['repetitions']:
                    raise ValueError('missing timing samples')
                if any(not math.isfinite(t) or t<=0 for t in item['wall_ms']):
                    raise ValueError('invalid timer')
        else:
            if not d.get('frozen_logprobs_unchanged'):raise ValueError('old/reference policy mutated')
            if len(d['steps'])!=d['args']['steps']:raise ValueError('missing model steps')
            for i,step in enumerate(d['steps']):
                if step['actor_version_before']!=i or step['old_policy_version']!=0:
                    raise ValueError('policy-version mismatch')
    groups=defaultdict(lambda:defaultdict(list));training=defaultdict(list)
    for p in sorted(root.glob('loss-*.json')):
        d=json.loads(p.read_text());a=d['args']
        key=f"N={a['batch']*a['response']}, H={a['hidden']}, V={a['vocab']}, frozen={a['frozen_head']}"
        for mode,item in d['methods'].items():
            groups[key][mode].append({'seed':a['seed'],'wall_ms':statistics.median(item['wall_ms']),
                                      'peak_bytes':item['peak_allocated_bytes'],'file':p.name})
    for p in sorted(root.glob('train-*.json')):
        d=json.loads(p.read_text());samples=[x for x in d['steps'] if x['phase']=='measured']
        if not samples or not all(x['timing_usable'] for x in samples):raise ValueError('accuracy run used as timing')
        training[f"{d['args']['method']}/micro{d['args']['microbatch']}"].append({
            'seed':d['args']['seed'],'wall_ms':statistics.median(x['wall_ms'] for x in samples),
            'peak_bytes':max(x['peak_allocated_bytes'] for x in samples),'file':p.name})
    summary={'unit_tests':int(match.group(1)),'completed_stages':len(markers),
             'runtime_hashes_match':True,'operator_groups':dict(groups),'model_groups':dict(training),
             'model_accuracy':accuracy,'raw_run_files':len(raw_files),
             'operator_method_measurements':sum(len(json.loads(p.read_text())['methods']) for p in
                  list(root.glob('loss-*.json'))+list(root.glob('ablation-*.json'))),
             'training_steps_recorded':sum(len(json.loads(p.read_text())['steps']) for p in root.glob('train-*.json')),
             'training_steps_measured':sum(sum(x['phase']=='measured' for x in json.loads(p.read_text())['steps']) for p in root.glob('train-*.json'))}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Frozen validation results','',
           'Operator rows: per-run median of all measured forward+backward calls, then median across runs. Peak = median of each run\'s maximum allocated tensor bytes. GiB uses 2^30 bytes. No samples or slow runs removed.',
           '', '| shape | mode | runs | wall ms | peak GiB |','|---|---|---:|---:|---:|']
    for key,modes in groups.items():
        for mode,runs in modes.items():
            lines.append(f"| {key} | {mode} | {len(runs)} | {statistics.median(r['wall_ms'] for r in runs):.3f} | {statistics.median(r['peak_bytes'] for r in runs)/2**30:.3f} |")
    lines+=['','## Full model training steps','',
            'Fixed pretrained initialization, real frozen trajectories, FP32 master / BF16 autocast, fused AdamW. First optimizer step excluded from steady-state medians; its data is retained. Generation is outside this timer.',
            '', '| mode / microbatch | runs | wall ms | peak GiB |','|---|---:|---:|---:|']
    for mode,runs in training.items():
        lines.append(f"| {mode} | {len(runs)} | {statistics.median(r['wall_ms'] for r in runs):.3f} | {statistics.median(r['peak_bytes'] for r in runs)/2**30:.3f} |")
    lines+=['','## Real-model numerical checks','',
            'Compare all initial parameters exactly, all first-step gradients, and actual three-step optimizer UPDATE vectors (final minus initial). Accuracy runs never supply performance numbers.',
            'Mixed-precision UPDATE tolerance failures are retained as diagnostics, including the eager-vs-compiled control. This table does not establish equivalent BF16 training trajectories.',
            '', '| mode | gradient relative L2 | gradient cosine | update relative L2 | update cosine | update within original tolerance |','|---|---:|---:|---:|---:|---|']
    for mode,stats in accuracy['methods'].items():
        if mode=='dense':continue
        g=stats['gradients'];u=stats['parameters']
        lines.append(f"| {mode} | {g['relative_l2']:.6g} | {g['cosine']:.8f} | {u['relative_l2']:.6g} | {u['cosine']:.8f} | {u['passed']} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    inventory={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in raw_files+markers+[root/'unit-tests.log',root/'source-fingerprint.json',root/'protocol.json']}
    (root/'evidence-manifest.json').write_text(json.dumps(inventory,indent=2))
    print(json.dumps({k:summary[k] for k in ['unit_tests','completed_stages','raw_run_files',
                                           'operator_method_measurements','training_steps_recorded','training_steps_measured']},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    report(Path(p.parse_args().root).resolve())
