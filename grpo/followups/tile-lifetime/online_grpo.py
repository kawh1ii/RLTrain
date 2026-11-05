"""Exercise grouped on-policy rollout and GRPO updates with verifiable rewards."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rltrain.batch import PolicyVersions, response_hidden, tensorize
from rltrain.fused import fused_loss
from rltrain.objective import LossConfig


def tasks(seed, count):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        if i % 4 == 0:
            a, b = rng.randrange(25, 100), rng.randrange(25, 100)
            question, answer = f"{a} + {b}", a + b
        elif i % 4 == 1:
            a, b = rng.randrange(140, 350), rng.randrange(21, 135)
            question, answer = f"{a} - {b}", a - b
        elif i % 4 == 2:
            a, b = rng.randrange(12, 35), rng.randrange(3, 10)
            question, answer = f"{a} * {b}", a * b
        else:
            a, b = rng.randrange(121, 550), rng.randrange(7, 23)
            question, answer = f"{a} modulo {b}", a % b
        rows.append((question, answer))
    return rows


def parse_answer(text):
    clean = text.strip()
    whole = re.fullmatch(r"([+-]?[0-9]+)\s*\.?", clean)
    if whole:
        return int(whole.group(1))
    # Grade only an explicitly identified final integer, with optional
    # Markdown/LaTeX wrappers. Intermediate operands do not qualify.
    tail = re.search(
        r"(?:final\s+answer\s*[:=]|answer\s*[:=]|\bis\b|=)"
        r"\s*(?:(?:\\\(|\\\[|\\boxed\{|\$|\*\*)\s*)*([+-]?[0-9]+)"
        r"\s*(?:(?:\\\)|\\\]|\}|\$|\*\*|[.!])\s*)*$",
        clean, re.I,
    )
    return int(tail.group(1)) if tail else None


def sample(model, tokenizer, problems, group_size, max_new, seed, greedy=False):
    prompts = []
    for problem, _ in problems:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Compute {problem}. Put the result at the end as Final answer: NUMBER."}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        prompts.extend([prompt] * group_size)
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), tokenizer.pad_token_id, device="cuda")
    mask = torch.zeros_like(ids)
    for i, row in enumerate(prompts):
        ids[i, -len(row):] = torch.tensor(row, device="cuda")
        mask[i, -len(row):] = 1

    torch.manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    settings = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, **settings,
        )
    torch.cuda.synchronize()
    generation_s = time.perf_counter() - start

    rows = []
    for i, completion in enumerate(generated[:, width:].tolist()):
        if tokenizer.eos_token_id in completion:
            completion = completion[:completion.index(tokenizer.eos_token_id) + 1]
        text = tokenizer.decode(completion, skip_special_tokens=True)
        question, answer = problems[i // group_size]
        parsed = parse_answer(text)
        rows.append({
            "group_id": i // group_size, "prompt_token_ids": prompts[i],
            "response_token_ids": completion, "response": text,
            "problem": question, "answer": answer, "parsed_answer": parsed, "reward": int(parsed == answer),
        })
    return rows, generation_s


def response_scores(model, batch, microbatch):
    results = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(batch["targets"]), microbatch):
            sl = slice(start, start + microbatch)
            hidden = model.model(
                input_ids=batch["input_ids"][sl], attention_mask=batch["attention_mask"][sl],
                use_cache=False,
            ).last_hidden_state
            selected = response_hidden(hidden, batch["positions"][sl])
            flat = selected.reshape(-1, selected.shape[-1])
            target = batch["targets"][sl].reshape(-1)
            parts = []
            weight = model.get_output_embeddings().weight
            for first in range(0, len(flat), 64):
                chunk = slice(first, first + 64)
                logits = (flat[chunk] @ weight.T).float()
                parts.append(logits.gather(-1, target[chunk, None]).squeeze(-1) - logits.logsumexp(-1))
            results.append(torch.cat(parts).reshape_as(batch["targets"][sl]))
        return torch.where(batch["response_mask"], torch.cat(results), 0.0)


def update(model, optimizer, batch, old, ref, microbatch):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    total = torch.zeros((), device="cuda")
    config = LossConfig()
    for offset in range(0, len(old), microbatch):
        sl = slice(offset, offset + microbatch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(
                input_ids=batch["input_ids"][sl], attention_mask=batch["attention_mask"][sl],
                use_cache=False,
            ).last_hidden_state
            h = response_hidden(hidden, batch["positions"][sl]).flatten(0, 1).to(torch.bfloat16)
            weight = model.get_output_embeddings().weight.to(torch.bfloat16)
            advantage = batch["advantages"][sl, None].expand_as(old[sl]).reshape(-1).contiguous()
            loss = fused_loss(h, weight, batch["targets"][sl].reshape(-1), old[sl].reshape(-1),
                              ref[sl].reshape(-1), advantage, batch["weights"][sl].reshape(-1),
                              config, check_values=False)
        loss.backward()
        total += loss.detach()
    probe = model.model.layers[0].self_attn.q_proj.weight[:64, :64]
    before = probe.detach().clone()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    delta = float((probe.detach() - before).norm())
    value = float(total)
    if not torch.isfinite(total) or not delta > 0:
        raise RuntimeError(f"Update is not finite and nonzero: loss={value}, probe_delta={delta}")
    return {"loss": value, "update_probe_l2": delta, "update_s": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=196)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(dtype=torch.float32, attn_implementation="sdpa")
    actor = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to("cuda").eval()
    ref = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to("cuda").eval()
    ref.requires_grad_(False)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6, weight_decay=0, fused=True)
    assert parse_answer("Final answer: -12.") == -12
    assert parse_answer(r"The result is \(148\).") == 148
    assert parse_answer("15 + 2 = 17, so we should now calculate") is None
    train = tasks(917, 12)
    holdout = tasks(4530, 16)
    assert not set(train) & set(holdout)
    max_new = 96
    result = {
        "protocol": {"train_tasks": train, "holdout_tasks": holdout, "group_size": 4,
                     "max_new_tokens": max_new, "optimizer_lr": 1e-6, "microbatch": 4,
                     "reward": "exact parsed final integer, 0 or 1", "sampling_seed": args.seed},
        "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        "model_manifest_sha256": hashlib.sha256((Path(args.model) / "download-manifest.json").read_bytes()).hexdigest(),
        "rounds": [],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    result["holdout_before"], _ = sample(actor, tokenizer, holdout, 1, max_new, args.seed, greedy=True)
    save()
    version = 0
    for i in range(3):
        records, generation_s = sample(actor, tokenizer, train[4*i:4*i+4], 4, max_new, args.seed+i)
        batch = tensorize(records, tokenizer.pad_token_id, device="cuda")
        varying = sum(len({r["reward"] for r in records if r["group_id"] == g}) > 1 for g in range(4))
        row = {"round": i, "rollout_version": version, "generation_s": generation_s,
               "correct_responses": sum(r["reward"] for r in records), "mixed_reward_groups": varying,
               "active_response_tokens": int(batch["response_mask"].sum()),
               "records": records, "advantages": batch["advantages"].tolist()}
        if varying:
            PolicyVersions(version, version, 0).validate(version, max_epoch_lag=0)
            t0 = time.perf_counter()
            old = response_scores(actor, batch, 4).clone()
            ref_logps = response_scores(ref, batch, 4).clone()
            torch.cuda.synchronize()
            if not (old.isfinite().all() and ref_logps.isfinite().all()):
                raise RuntimeError("Nonfinite rollout log probabilities")
            row["logprob_s"] = time.perf_counter() - t0
            row.update(update(actor, optimizer, batch, old, ref_logps, 4))
            version += 1
        row["actor_version_after"] = version
        result["rounds"].append(row)
        save()
        print(json.dumps({k: v for k, v in row.items() if k not in ("records", "advantages")}), flush=True)
    result["holdout_after"], _ = sample(actor, tokenizer, holdout, 1, max_new, args.seed, greedy=True)
    result["optimizer_updates"] = version
    result["completed"] = version >= 2
    save()
    if not result["completed"]:
        raise RuntimeError("Fewer than two informative policy updates")


if __name__ == "__main__":
    main()
