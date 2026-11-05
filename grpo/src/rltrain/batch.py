"""Frozen-rollout contracts and causal response positions for model-step tests."""
from dataclasses import dataclass
import torch
from rltrain.objective import outcome_advantages,response_weights


@dataclass(frozen=True)
class PolicyVersions:
    rollout: int
    old_logprob: int
    reference: int

    def validate(self,current,max_epoch_lag):
        if min(self.rollout,self.old_logprob,self.reference,current,max_epoch_lag)<0:
            raise ValueError("negative policy version")
        if self.rollout!=self.old_logprob:
            raise ValueError("old log-probs belong to another rollout checkpoint")
        if not 0<=current-self.rollout<=max_epoch_lag:
            raise ValueError("current policy outside declared frozen-batch PPO epochs")


def tensorize(records,pad_id,device="cpu"):
    if not records:raise ValueError("no trajectories")
    prompts=[r['prompt_token_ids'] for r in records]
    responses=[r['response_token_ids'] for r in records]
    if any(not p for p in prompts) or any(not r for r in responses):
        raise ValueError("empty prompt/response")
    b=len(records);s=max(len(p)+len(r) for p,r in zip(prompts,responses));t=max(map(len,responses))
    ids=torch.full((b,s),pad_id,dtype=torch.int64,device=device)
    attention=torch.zeros(b,s,dtype=torch.bool,device=device)
    labels=torch.full((b,t),pad_id,dtype=torch.int64,device=device)
    mask=torch.zeros(b,t,dtype=torch.bool,device=device)
    prompt_lengths=torch.tensor(list(map(len,prompts)),dtype=torch.int64,device=device)
    for i,(p,r) in enumerate(zip(prompts,responses)):
        seq=torch.tensor(p+r,dtype=torch.int64,device=device)
        ids[i,:len(seq)]=seq;attention[i,:len(seq)]=True
        labels[i,:len(r)]=torch.tensor(r,dtype=torch.int64,device=device);mask[i,:len(r)]=True
    positions=prompt_lengths[:,None]+torch.arange(t,device=device)[None,:]-1
    # Padded response positions may exceed the padded input width; they are
    # harmless only after clamping AND applying their zero loss weights.
    positions=positions.clamp(0,s-1)
    rewards=torch.tensor([r['reward'] for r in records],dtype=torch.float32,device=device)
    advantages=outcome_advantages(rewards,[r['group_id'] for r in records])
    return {'input_ids':ids,'attention_mask':attention,'targets':labels,'response_mask':mask,
            'positions':positions,'advantages':advantages,'weights':response_weights(mask),
            'rewards':rewards,'prompt_lengths':prompt_lengths}


def response_hidden(hidden,positions):
    batch=torch.arange(hidden.shape[0],device=hidden.device)[:,None]
    return hidden[batch,positions]
