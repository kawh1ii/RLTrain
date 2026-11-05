"""Check that releasing a Triton tile leaves loss and both gradients identical."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark_loss import make_data
from rltrain.objective import LossConfig


def load(name, root):
    spec = spec_from_file_location(name, root / "src/rltrain/fused.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()
    a, b = load("fused_original", args.baseline), load("fused_lifetime", args.candidate)
    for n in (1024, 2048, 4096):
        for seed in (197, 198, 199):
            data = make_data(SimpleNamespace(batch=4, response=n // 4, hidden=896,
                                             vocab=151936, seed=seed, frozen_head=False))
            cfg = LossConfig(strategy="vocab", vocab_chunk=32768)
            original = a.fused_loss(*data, cfg, backend="triton")
            expected = torch.autograd.grad(original, data[:2])
            candidate = b.fused_loss(*data, cfg, backend="triton")
            observed = torch.autograd.grad(candidate, data[:2])
            if not torch.equal(original, candidate) or not all(
                    torch.equal(x, y) for x, y in zip(expected, observed)):
                raise AssertionError(f"Output or gradients changed at N={n}, seed={seed}")
            print("EXACT_MATCH", n, seed, flush=True)


if __name__ == "__main__":
    main()
