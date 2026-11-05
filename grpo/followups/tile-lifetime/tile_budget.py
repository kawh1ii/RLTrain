"""Choose a BF16 Triton vocabulary tile within the tile-buffer budget.

The budget covers one logits tile and two FP32 partial reduction buffers.
It excludes weights, gradients, persistent row statistics, model activations,
and library workspace. Use with the explicit tile lifetime patch.
"""
import argparse


def tile_bytes(tokens, width):
    parts = (width + 4095) // 4096
    return tokens * (2 * width + 8 * parts)


def choose_width(tokens, vocab, budget_bytes):
    if min(tokens, vocab, budget_bytes) < 1:
        raise ValueError("Token count, vocabulary size and budget must be positive")
    for width in (32768, 16384, 8192, 4096):
        width = min(width, vocab)
        if tile_bytes(tokens, width) <= budget_bytes:
            return width
    raise ValueError("Tile buffer budget cannot hold the smallest supported tile")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--budget-mib", type=int, default=80)
    a = p.parse_args()
    print(choose_width(a.tokens, a.vocab, a.budget_mib * 2**20))
