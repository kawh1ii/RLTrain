"""Pinned, licensed GSM8K subset; never silently truncate prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT='3101c7d5072418e28b9008a6636bde82a006892c'
BASE=f'https://raw.githubusercontent.com/openai/grade-school-math/{COMMIT}/'
INSTRUCTION='Solve the problem step by step. End your answer with #### followed by the final numeric answer.\n\n'
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--count',type=int,default=4);p.add_argument('--max-prompt',type=int,default=256)
a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
raw=urllib.request.urlopen(BASE+'grade_school_math/data/train.jsonl',timeout=60).read()
license_=urllib.request.urlopen(BASE+'LICENSE',timeout=30).read()
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(a.model)
selected=[]
for index,line in enumerate(raw.splitlines()):
    source=json.loads(line)
    question=INSTRUCTION+source['question']
    ids=tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=True,
                                      add_generation_prompt=True,return_dict=False)
    if not isinstance(ids,list) or not all(isinstance(v,int) for v in ids):
        raise TypeError('expected a flat token-ID list from chat template')
    if len(ids)>a.max_prompt:continue
    selected.append({'source_row':index,'question':source['question'],'prompt':question,
                     'prompt_token_ids':ids,'gold':source['answer'].split('####')[-1].strip().replace(',','')})
    if len(selected)==a.count:break
if len(selected)!=a.count:raise RuntimeError('not enough prompts within length limit')
(root/'GSM8K_LICENSE').write_bytes(license_)
data={'source':BASE+'grade_school_math/data/train.jsonl','upstream_commit':COMMIT,
      'full_source_sha256':hashlib.sha256(raw).hexdigest(),'split':'train','license':'MIT',
      'selection':'first rows fitting token limit, no answer-based selection','max_prompt':a.max_prompt,
      'records':selected}
(root/'prompts.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'prompts':len(selected),'token_lengths':[len(r['prompt_token_ids']) for r in selected]},indent=2))
