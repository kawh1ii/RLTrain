"""Additional paired-input and evidence-integrity audit; Python standard library only."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from report import report
R=ROOT/'results/final'
summary=report(R)
def read(name):return json.loads((R/name).read_text())
assert read('ablation-sequence.json')['input_hashes']==read('ablation-vocab.json')['input_hashes']
assert read('loss-extension-h896-frozen1.json')['input_hashes']==read('loss-n2048-h896-s84.json')['input_hashes']
runs=[json.loads(p.read_text()) for p in list(R.glob('train-*.json'))+list((R/'accuracy').glob('*.json')) if p.name!='comparison.json']
for key in ('data_sha256','old_logps_sha256','reference_logps_sha256'):
    assert len({d[key] for d in runs})==1,key
payload=ROOT/'data/rollouts256/trajectories.json'
assert hashlib.sha256(payload.read_bytes()).hexdigest()==runs[0]['data_sha256']
comparison=read('accuracy/comparison.json')
assert comparison['all_required_checks_passed']
assert not comparison['all_bf16_update_tolerances_passed']
initial=comparison['methods']['dense']['initial_parameters']['fingerprints']
for mode,d in comparison['methods'].items():
    if mode=='dense':continue
    assert d['initial_parameters']['passed']
    assert d['gradients']['passed']
    assert not d['parameters']['passed']
    assert not d['parameters']['required_gate']
    assert d['parameters']['nonzero_update']
for relative,digest in json.loads((R/'evidence-manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
print('PASS: paired ablations, identical frozen trajectories/logprobs, initial/gradient gates, retained BF16 update failures, evidence hashes')
