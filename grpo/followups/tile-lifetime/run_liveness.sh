#!/usr/bin/env bash
set -euo pipefail
baseline=$(realpath "$1")
candidate=$(realpath "$2")
out=$(realpath -m "$3")
py=$(realpath -s "$4")
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$out"
export PYTHONDONTWRITEBYTECODE=1
for seed in 197 198 199; do
    for response in 256 512 1024; do
        tokens=$((4 * response))
        tile=$("$py" "$here/tile_budget.py" --tokens "$tokens" --budget-mib 80)
        order=(original lifetime budget)
        if (( seed % 2 == 0 )); then order=(budget lifetime original); fi
        for method in "${order[@]}"; do
            repo=$candidate
            width=32768
            if [[ $method == original ]]; then repo=$baseline; fi
            if [[ $method == budget ]]; then width=$tile; fi
            name="n${tokens}-s${seed}-${method}"
            (
                cd "$repo"
                PYTHONPATH=src "$py" tools/benchmark_loss.py --output "$out/$name.json" \
                    --batch 4 --response "$response" --seed "$seed" --vocab-chunk "$width" \
                    --methods fused --warmup 3 --repetitions 11
            ) > "$out/$name.log" 2>&1
            echo "DONE $name"
        done
    done
done
