"""Identical objective controls, including a pinned external implementation."""
import os
import sys
import functools
import hashlib
import importlib
import json
from pathlib import Path
import torch
import torch._functorch.config as aot_config
from rltrain.objective import reference_loss, checkpoint_loss
from rltrain.fused import fused_loss

# Training uses AMP for forward and runs backward outside AMP. Match the
# documented AOTAutograd contract rather than its same_as_forward default.
aot_config.backward_pass_autocast="off"


@functools.lru_cache(maxsize=1)
def liger_class_and_provenance():
    path=os.environ.get("RLTRAIN_LIGER_PATH")
    if not path:raise ValueError("set RLTRAIN_LIGER_PATH to pinned checkout/src")
    source=Path(path).resolve()
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    manifest_path=Path(__file__).resolve().parents[2]/'baselines/liger-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for name,digest in manifest['source_hashes'].items():
        file=source.parent/name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError('external baseline source mismatch: '+name)
    module=importlib.import_module('liger_kernel.chunked_loss.grpo_loss')
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError('imported another Liger installation')
    info={'repository':manifest['repository'],'commit':manifest['commit'],
          'verified_source_files':len(manifest['source_hashes']),
          'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'loaded_module':module.__file__}
    return module.LigerFusedLinearGRPOLoss,info


def method_loss(method, data, config, batch, response):
    if method == "dense":
        return reference_loss(*data, config)
    if method == "checkpoint":
        return checkpoint_loss(*data, config)
    if method == "manual_torch":
        return fused_loss(*data, config, backend="torch",check_values=False)
    if method == "fused":
        return fused_loss(*data, config, backend="triton",check_values=False)
    if method == "compiled":
        return compiled_reference(*data, config)
    if method == "liger":
        return prepare_method(method,data,config,batch,response)()
    raise ValueError("unknown baseline: "+method)


# Compiles once per shape/config and includes compiled backward. Creation is
# lazy in torch.compile; compilation/warmup is excluded for EVERY method.
compiled_reference = torch.compile(reference_loss, fullgraph=True)


def prepare_method(method,data,config,batch,response,normalization_scale=1.0):
    """Prepare immutable metadata/module once, outside any timed operation."""
    if method != "liger":
        return lambda:method_loss(method,data,config,batch,response)
    LigerFusedLinearGRPOLoss,_=liger_class_and_provenance()
    mask=(data[-1].view(batch,response)>0)
    seq_adv=data[-2].view(batch,response)[:,0]
    expected=mask.float()/mask.sum(-1,keepdim=True).clamp(min=1)/batch*normalization_scale
    torch.testing.assert_close(data[-1].view(batch,response),expected)
    torch.testing.assert_close(data[-2].view(batch,response),seq_adv[:,None].expand(-1,response))
    module=LigerFusedLinearGRPOLoss(beta=config.beta,compiled=True,chunk_size=1,
                loss_type="grpo",use_ref_model=True,
                epsilon_low=config.epsilon,epsilon_high=config.epsilon)
    def call():
        h,w,y,old,ref,_,_=data
        return module(h.view(batch,response,-1),w,y.view(batch,response),mask,seq_adv,
                      ref_per_token_logps=ref.view(batch,response),
                      old_per_token_logps=old.view(batch,response))[0]*normalization_scale
    return call
