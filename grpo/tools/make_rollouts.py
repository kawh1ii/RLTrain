"""Create real sampled responses and fixed old/reference log-probs once.

This is a systems workload, not a convergence experiment. Teacher-forced
old-policy log-probs are recomputed on checkpoint zero; they are not claimed
bit-identical to cached generation kernels' logits.
"""
import argparse
from decimal import Decimal,InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rltrain.batch import tensorize,response_hidden

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--prompts',required=True)
p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=85)
p.add_argument('--n',type=int,default=4);p.add_argument('--tokens',type=int,default=128)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if (out/'trajectories.json').exists():raise FileExistsError(out)
torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.manual_seed(a.seed)
tokenizer=AutoTokenizer.from_pretrained(a.model,padding_side='left')
tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
sources=json.loads(Path(a.prompts).read_text())['records']
prompt_ids=[r['prompt_token_ids'] for r in sources for _ in range(a.n)]
maximum=max(map(len,prompt_ids))
ids=torch.full((len(prompt_ids),maximum),tokenizer.pad_token_id,device='cuda',dtype=torch.int64)
mask=torch.zeros_like(ids)
for i,row in enumerate(prompt_ids):ids[i,-len(row):]=torch.tensor(row,device='cuda');mask[i,-len(row):]=1
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    generated=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=a.tokens,
        do_sample=True,temperature=1.0,top_p=1.0,top_k=0,repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
records=[]
for i,sequence in enumerate(generated[:,maximum:].tolist()):
    if tokenizer.eos_token_id in sequence:sequence=sequence[:sequence.index(tokenizer.eos_token_id)+1]
    source=sources[i//a.n];text=tokenizer.decode(sequence,skip_special_tokens=True)
    match=re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)',text)
    correct=False
    if match:
        try:correct=Decimal(match.group(1).replace(',',''))==Decimal(source['gold'])
        except InvalidOperation:pass
    reward=float(correct)-.001*len(sequence)
    records.append({'group_id':i//a.n,'sample':i%a.n,'source_row':source['source_row'],
                    'prompt_token_ids':prompt_ids[i],'response_token_ids':sequence,
                    'response':text,'correct':correct,'reward':reward,
                    'reward_components':{'numeric_correctness':float(correct),'length_penalty':-.001*len(sequence)}})
batch=tensorize(records,tokenizer.pad_token_id,device='cuda')
old=torch.zeros_like(batch['weights'])
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for start in range(0,len(records),4):
        sl=slice(start,start+4)
        hidden=model.model(input_ids=batch['input_ids'][sl],attention_mask=batch['attention_mask'][sl],use_cache=False).last_hidden_state
        h=response_hidden(hidden,batch['positions'][sl]).reshape(-1,hidden.shape[-1])
        y=batch['targets'][sl].reshape(-1)
        values=[]
        for offset in range(0,len(h),64):
            logits=(h[offset:offset+64]@model.get_output_embeddings().weight.T).float()
            values.append(logits.gather(-1,y[offset:offset+64,None]).squeeze(-1)-logits.logsumexp(-1))
        old[sl]=torch.cat(values).view_as(old[sl])
old=torch.where(batch['response_mask'],old,torch.zeros_like(old))
torch.save({'old_logps':old.cpu(),'ref_logps':old.cpu().clone()},out/'logprobs.pt')
data={'model':a.model,'seed':a.seed,'n':a.n,'max_new_tokens':a.tokens,
      'sampling':{'temperature':1.0,'top_p':1.0,'top_k':0},
      'parameter_dtype':'float32','compute_autocast':'bfloat16',
      'float32_matmul_precision':'high',
      'rollout_policy_version':0,'old_logprob_policy_version':0,'reference_policy_version':0,
      'reward':'exact #### numeric correctness minus 0.001 per response token',
      'prompts_sha256':hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest(),
      'logprobs_sha256':hashlib.sha256((out/'logprobs.pt').read_bytes()).hexdigest(),
      'old_logprob_source':'teacher-forced checkpoint zero, right padded, not cached-generation bit equivalence',
      'records':records,'group_advantages':batch['advantages'].cpu().tolist(),
      'active_response_tokens':int(batch['response_mask'].sum()),
      'nonzero_advantages':int((batch['advantages']!=0).sum()),
      'correct_responses':sum(r['correct'] for r in records)}
(out/'trajectories.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:data[k] for k in ['active_response_tokens','nonzero_advantages','correct_responses']},indent=2))
