"""Check paired inputs, numerical gates, and the policy-version sequence."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def benchmark(path):
    data = read(path)
    assert data["completed"] and data["inputs_unchanged"], path
    assert data["environment"]["torch"] == "2.8.0+cu128", path
    for row in data["methods"].values():
        assert row["status"] == "passed" and row["accuracy"]["passed"], path
        assert len(row["wall_ms"]) == 11, path
    return data


raw = ROOT / "grpo-cu128"
assert len(list((raw / "sweep").glob("*.json"))) == 24
assert len(list((raw / "liveness").glob("*.json"))) == 27
for response in (128, 1024):
    for seed in (93, 94, 95):
        runs = [benchmark(raw / "sweep" / f"r{response}-s{seed}-tile{tile}.json")
                for tile in (8192, 16384, 32768, 65536)]
        assert all(x["input_hashes"] == runs[0]["input_hashes"] for x in runs)

print("Tokens    Old ms/GiB       Release ms/GiB     Release + budget ms/GiB")
for n, tile in ((1024, 32768), (2048, 16384), (4096, 8192)):
    groups = [[], [], []]
    for seed in (197, 198, 199):
        runs = [benchmark(raw / "liveness" / f"n{n}-s{seed}-{method}.json")
                for method in ("original", "lifetime", "budget")]
        assert all(x["input_hashes"] == runs[0]["input_hashes"] for x in runs)
        assert [x["args"]["vocab_chunk"] for x in runs] == [32768, 32768, tile]
        original, candidate, capped = [x["source_hashes"] for x in runs]
        assert set(original) == set(candidate)
        assert [k for k in original if original[k] != candidate[k]] == ["src/rltrain/fused.py"]
        assert candidate == capped
        for group, run in zip(groups, runs):
            group.append(run["methods"]["fused"])
    values = [(statistics.median(x["median_wall_ms"] for x in group),
               statistics.median(x["peak_allocated_bytes"] for x in group) / 2**30)
              for group in groups]
    print(f"{n:<9} " + "    ".join(f"{t:.3f}/{m:.3f}" for t, m in values))

paired = []
for seed in (197, 198, 199):
    run = benchmark(raw / "final-paired" / f"s{seed}.json")
    reference = benchmark(raw / "liveness" / f"n4096-s{seed}-budget.json")
    assert run["source_hashes"] == reference["source_hashes"]
    assert run["input_hashes"] == reference["input_hashes"]
    assert run["args"]["vocab_chunk"] == 8192
    paired.append(run["methods"])
for method in ("dense", "compiled", "fused"):
    wall = statistics.median(run[method]["median_wall_ms"] for run in paired)
    gib = statistics.median(run[method]["peak_allocated_bytes"] for run in paired) / 2**30
    print(f"Matched 4096-token control {method}: {wall:.3f} ms, {gib:.3f} GiB")

expected = {f"EXACT_MATCH {n} {s}" for n in (1024, 2048, 4096) for s in (197, 198, 199)}
assert set((ROOT / "rl-lifetime-exact.log").read_text().splitlines()) == expected
assert "Ran 42 tests" in (ROOT / "rl-liveness-units.log").read_text()

for seed in (201, 211, 221):
    data = read(raw / "online-v2" / f"s{seed}.json")
    assert data["completed"] and data["optimizer_updates"] == len(data["rounds"]) == 3
    for version, row in enumerate(data["rounds"]):
        assert row["rollout_version"] == version and row["actor_version_after"] == version + 1
        assert len(row["records"]) == 16 and row["mixed_reward_groups"] > 0
        assert row["update_probe_l2"] > 0
        for r in row["records"]:
            assert r["reward"] == int(r["parsed_answer"] == r["answer"])
    before = sum(x["reward"] for x in data["holdout_before"])
    after = sum(x["reward"] for x in data["holdout_after"])
    print(f"Seed {seed}: {data['optimizer_updates']} updates, greedy holdout {before}/16 → {after}/16")

print("PASS: matched GPU cases, exact lifetime gradients and online actor versions")
