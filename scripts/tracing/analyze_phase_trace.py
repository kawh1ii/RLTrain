#!/usr/bin/env python3

import os
import json
import statistics
from pathlib import Path
from datetime import datetime


RESULT_DIR = Path(
    os.environ["RLTRAIN_RESULT_DIR"]
)

RUN_ID = os.environ[
    "RLTRAIN_RUN_ID"
]

OUTPUT_JSON = Path(
    os.environ["RLTRAIN_OUTPUT_JSON"]
)

REPORT_MD = Path(
    os.environ["RLTRAIN_REPORT_MD"]
)


ARCHES = {
    "DISAGG_6PLUS2": {
        "trace":
            RESULT_DIR
            /
            f"RLTRAIN_STAGE_9_5_2_DISAGG_TRACE_{RUN_ID}.jsonl",

        "phases": [
            "generate_async",
            "gen",
            "reward",
            "adv",
            "update_actor",
            "sync_rollout_weights",
        ],
    },

    "COLOCATED_8GPU": {
        "trace":
            RESULT_DIR
            /
            f"RLTRAIN_STAGE_9_5_2_COLOCATED_TRACE_{RUN_ID}.jsonl",

        "phases": [
            "gen",
            "old_log_prob",
            "adv",
            "update_actor",
            "update_weights",
        ],
    },
}


def merge_intervals(
    intervals,
):
    if not intervals:
        return []

    values = sorted(
        intervals
    )

    merged = [
        list(
            values[0]
        )
    ]

    for start, end in values[1:]:

        if start <= merged[-1][1]:

            merged[-1][1] = max(
                merged[-1][1],
                end,
            )

        else:

            merged.append(
                [
                    start,
                    end,
                ]
            )

    return [
        (
            start,
            end,
        )
        for start, end
        in merged
    ]


def union_ns(
    intervals,
):
    return sum(
        end - start
        for start, end
        in merge_intervals(
            intervals
        )
    )


def clip(
    interval,
    window,
):
    start = max(
        interval[0],
        window[0],
    )

    end = min(
        interval[1],
        window[1],
    )

    if end <= start:
        return None

    return (
        start,
        end,
    )


def overlap_ns(
    a,
    b,
):
    aa = merge_intervals(a)
    bb = merge_intervals(b)

    i = 0
    j = 0
    total = 0

    while (
        i < len(aa)
        and
        j < len(bb)
    ):

        start = max(
            aa[i][0],
            bb[j][0],
        )

        end = min(
            aa[i][1],
            bb[j][1],
        )

        if end > start:
            total += end - start

        if aa[i][1] <= bb[j][1]:
            i += 1
        else:
            j += 1

    return total


def complement(
    covered,
    window,
):
    merged = merge_intervals(
        [
            clipped
            for interval in covered
            if (
                clipped := clip(
                    interval,
                    window,
                )
            )
            is not None
        ]
    )

    gaps = []

    cursor = window[0]

    for start, end in merged:

        if start > cursor:
            gaps.append(
                (
                    cursor,
                    start,
                )
            )

        cursor = max(
            cursor,
            end,
        )

    if cursor < window[1]:

        gaps.append(
            (
                cursor,
                window[1],
            )
        )

    return gaps


def load_events(
    path,
):
    events = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(
                    line
                )

            except Exception:
                continue

            events.append(
                event
            )

    return events


def build_intervals(
    events,
):
    begins = {}
    intervals = []

    for event in events:

        kind = event.get(
            "event",
            "",
        )

        ctx = event.get(
            "ctx"
        )

        if not ctx:
            continue

        if kind.endswith(
            "_BEGIN"
        ):

            begins[
                ctx
            ] = event

            continue

        if not kind.endswith(
            "_END"
        ):

            continue

        begin = begins.get(
            ctx
        )

        if begin is None:
            continue

        start = int(
            begin[
                "ts_mono_ns"
            ]
        )

        end = int(
            event[
                "ts_mono_ns"
            ]
        )

        if end < start:
            continue

        intervals.append(
            {
                "kind":
                    kind[
                        :-4
                    ],

                "name":
                    begin.get(
                        "name"
                    ),

                "ctx":
                    ctx,

                "pid":
                    int(
                        begin.get(
                            "pid",
                            -1,
                        )
                    ),

                "start_ns":
                    start,

                "end_ns":
                    end,

                "duration_s":
                    (
                        end
                        -
                        start
                    )
                    /
                    1e9,
            }
        )

    return intervals


def intervals_for(
    intervals,
    *,
    kind=None,
    name=None,
    contains=None,
    window=None,
):
    result = []

    for item in intervals:

        if (
            kind is not None
            and
            item[
                "kind"
            ]
            != kind
        ):
            continue

        if (
            name is not None
            and
            item[
                "name"
            ]
            != name
        ):
            continue

        if (
            contains is not None
            and
            contains
            not in str(
                item[
                    "name"
                ]
            )
        ):
            continue

        interval = (
            item[
                "start_ns"
            ],
            item[
                "end_ns"
            ],
        )

        if window is not None:

            interval = clip(
                interval,
                window,
            )

            if interval is None:
                continue

        result.append(
            interval
        )

    return result


def analyze_arch(
    arch,
    path,
    phase_names,
):
    events = load_events(
        path
    )

    intervals = build_intervals(
        events
    )

    #
    # Find the process that owns the main
    # step timers.
    #

    steps_by_pid = {}

    for item in intervals:

        if (
            item[
                "kind"
            ]
            ==
            "TIMER"
            and
            item[
                "name"
            ]
            ==
            "step"
        ):

            steps_by_pid.setdefault(
                item[
                    "pid"
                ],
                [],
            ).append(
                item
            )

    if not steps_by_pid:

        raise RuntimeError(
            f"No step timers found for {arch}"
        )

    driver_pid = max(
        steps_by_pid,
        key=lambda pid:
            len(
                steps_by_pid[
                    pid
                ]
            ),
    )

    step_intervals = sorted(
        steps_by_pid[
            driver_pid
        ],
        key=lambda item:
            item[
                "start_ns"
            ],
    )

    if len(
        step_intervals
    ) < 4:

        raise RuntimeError(
            f"{arch}: expected at least "
            f"4 step timers, got "
            f"{len(step_intervals)}"
        )

    #
    # Four-step trace:
    # Step 1-2 warm-up.
    # Step 3-4 measured.
    #

    measured_steps = (
        step_intervals[
            -2:
        ]
    )

    step_results = []

    for measured_index, step_item in enumerate(
        measured_steps,
        start=3,
    ):

        window = (
            step_item[
                "start_ns"
            ],
            step_item[
                "end_ns"
            ],
        )

        step_duration_ns = (
            window[1]
            -
            window[0]
        )

        phase_map = {}

        all_phase_intervals = []

        for phase in phase_names:

            values = intervals_for(
                intervals,
                kind="TIMER",
                name=phase,
                window=window,
            )

            phase_map[
                phase
            ] = values

            all_phase_intervals.extend(
                values
            )

        covered_ns = union_ns(
            all_phase_intervals
        )

        gaps = complement(
            all_phase_intervals,
            window,
        )

        gap_rows = []

        tq_all = intervals_for(
            intervals,
            kind="TQ",
            window=window,
        )

        sleep_all = intervals_for(
            intervals,
            kind="METHOD",
            contains="sleep_replicas",
            window=window,
        )

        method_update_all = intervals_for(
            intervals,
            kind="METHOD",
            contains=".update_weights",
            window=window,
        )

        for start, end in gaps:

            gap = (
                start,
                end,
            )

            tq_ns = overlap_ns(
                [
                    gap
                ],
                tq_all,
            )

            sleep_ns = overlap_ns(
                [
                    gap
                ],
                sleep_all,
            )

            update_ns = overlap_ns(
                [
                    gap
                ],
                method_update_all,
            )

            gap_rows.append(
                {
                    "start_offset_s":
                        (
                            start
                            -
                            window[0]
                        )
                        /
                        1e9,

                    "end_offset_s":
                        (
                            end
                            -
                            window[0]
                        )
                        /
                        1e9,

                    "duration_s":
                        (
                            end
                            -
                            start
                        )
                        /
                        1e9,

                    "tq_overlap_s":
                        tq_ns
                        /
                        1e9,

                    "sleep_replicas_overlap_s":
                        sleep_ns
                        /
                        1e9,

                    "method_update_weights_overlap_s":
                        update_ns
                        /
                        1e9,
                }
            )

        def phase_union(
            phase,
        ):
            return union_ns(
                phase_map.get(
                    phase,
                    []
                )
            )

        generation_phase = (
            "generate_async"
            if arch
            ==
            "DISAGG_6PLUS2"
            else
            "gen"
        )

        generation = phase_map.get(
            generation_phase,
            []
        )

        actor = phase_map.get(
            "update_actor",
            []
        )

        weight_phase = (
            "sync_rollout_weights"
            if arch
            ==
            "DISAGG_6PLUS2"
            else
            "update_weights"
        )

        weight = phase_map.get(
            weight_phase,
            []
        )

        gen_actor_overlap_ns = overlap_ns(
            generation,
            actor,
        )

        gen_weight_overlap_ns = overlap_ns(
            generation,
            weight,
        )

        actor_weight_overlap_ns = overlap_ns(
            actor,
            weight,
        )

        largest_gap = (
            max(
                gap_rows,
                key=lambda row:
                    row[
                        "duration_s"
                    ],
            )
            if gap_rows
            else None
        )

        step_results.append(
            {
                "step_index":
                    measured_index,

                "driver_pid":
                    driver_pid,

                "step_duration_s":
                    step_duration_ns
                    /
                    1e9,

                "phase_union_coverage_s":
                    covered_ns
                    /
                    1e9,

                "phase_union_coverage_fraction":
                    covered_ns
                    /
                    step_duration_ns,

                "uncovered_wall_s":
                    (
                        step_duration_ns
                        -
                        covered_ns
                    )
                    /
                    1e9,

                "generation_phase":
                    generation_phase,

                "generation_union_s":
                    phase_union(
                        generation_phase
                    )
                    /
                    1e9,

                "actor_update_union_s":
                    phase_union(
                        "update_actor"
                    )
                    /
                    1e9,

                "weight_phase":
                    weight_phase,

                "weight_union_s":
                    phase_union(
                        weight_phase
                    )
                    /
                    1e9,

                "generation_actor_overlap_s":
                    gen_actor_overlap_ns
                    /
                    1e9,

                "generation_weight_overlap_s":
                    gen_weight_overlap_ns
                    /
                    1e9,

                "actor_weight_overlap_s":
                    actor_weight_overlap_ns
                    /
                    1e9,

                "gaps":
                    gap_rows,

                "largest_uncovered_gap":
                    largest_gap,
            }
        )

    def means(
        key,
    ):
        return statistics.mean(
            row[key]
            for row
            in step_results
        )

    result = {
        "trace_file":
            str(
                path
            ),

        "event_count":
            len(
                events
            ),

        "interval_count":
            len(
                intervals
            ),

        "driver_pid":
            driver_pid,

        "total_complete_step_timers":
            len(
                step_intervals
            ),

        "measured_steps":
            step_results,

        "means": {
            "step_duration_s":
                means(
                    "step_duration_s"
                ),

            "phase_union_coverage_s":
                means(
                    "phase_union_coverage_s"
                ),

            "phase_union_coverage_fraction":
                means(
                    "phase_union_coverage_fraction"
                ),

            "uncovered_wall_s":
                means(
                    "uncovered_wall_s"
                ),

            "generation_union_s":
                means(
                    "generation_union_s"
                ),

            "actor_update_union_s":
                means(
                    "actor_update_union_s"
                ),

            "weight_union_s":
                means(
                    "weight_union_s"
                ),

            "generation_actor_overlap_s":
                means(
                    "generation_actor_overlap_s"
                ),

            "generation_weight_overlap_s":
                means(
                    "generation_weight_overlap_s"
                ),

            "actor_weight_overlap_s":
                means(
                    "actor_weight_overlap_s"
                ),
        },
    }

    return result


results = {}


for arch, config in ARCHES.items():

    path = config[
        "trace"
    ]

    if not path.exists():

        raise RuntimeError(
            "Trace missing: "
            + str(
                path
            )
        )

    results[
        arch
    ] = analyze_arch(
        arch,
        path,
        config[
            "phases"
        ],
    )


A = results[
    "DISAGG_6PLUS2"
]

B = results[
    "COLOCATED_8GPU"
]


disagg_overlap = (
    A[
        "means"
    ][
        "generation_actor_overlap_s"
    ]
)

disagg_step = (
    A[
        "means"
    ][
        "step_duration_s"
    ]
)

disagg_overlap_fraction = (
    disagg_overlap
    /
    disagg_step
)


colocated_uncovered = (
    B[
        "means"
    ][
        "uncovered_wall_s"
    ]
)

colocated_step = (
    B[
        "means"
    ][
        "step_duration_s"
    ]
)

colocated_uncovered_fraction = (
    colocated_uncovered
    /
    colocated_step
)


colocated_largest_gaps = [
    row[
        "largest_uncovered_gap"
    ]
    for row
    in B[
        "measured_steps"
    ]
    if row[
        "largest_uncovered_gap"
    ]
    is not None
]


largest_colocated_gap = (
    max(
        colocated_largest_gaps,
        key=lambda row:
            row[
                "duration_s"
            ],
    )
    if colocated_largest_gaps
    else None
)


classification = (
    "TARGETED_PHASE_TRACE_COMPLETE_"
    "DIRECT_INTERVAL_OVERLAP_AND_"
    "UNACCOUNTED_WALL_OBSERVED"
)


verdict = (
    "RLTRAIN_STAGE_9_5_2_TARGETED_PHASE_EVENT_TRACE_"
    "COMPLETE_READY_FOR_RLTRAIN_STAGE_9_5_3"
)


decision = {
    "schema_version":
        1,

    "project":
        "RLTrain",

    "milestone":
        "RLTrain-stage-9.5.2",

    "timestamp":
        datetime.now()
        .astimezone()
        .isoformat(),

    "mode":
        "targeted_event_trace",

    "results":
        results,

    "summary": {
        "disagg_generation_actor_overlap_mean_s":
            disagg_overlap,

        "disagg_generation_actor_overlap_fraction_of_step":
            disagg_overlap_fraction,

        "disagg_phase_union_coverage_fraction":
            A[
                "means"
            ][
                "phase_union_coverage_fraction"
            ],

        "disagg_uncovered_wall_mean_s":
            A[
                "means"
            ][
                "uncovered_wall_s"
            ],

        "colocated_phase_union_coverage_fraction":
            B[
                "means"
            ][
                "phase_union_coverage_fraction"
            ],

        "colocated_uncovered_wall_mean_s":
            colocated_uncovered,

        "colocated_uncovered_wall_fraction_of_step":
            colocated_uncovered_fraction,

        "largest_colocated_uncovered_gap":
            largest_colocated_gap,
    },

    "classification":
        classification,

    "scope": {
        "process_level_replicates":
            1,

        "measured_steps_per_architecture":
            2,

        "formal_performance_claim":
            False,

        "formal_statistical_claim":
            False,

        "causal_mechanism_claim":
            False,

        "direct_interval_overlap_observation":
            True,

        "direct_uncovered_wall_observation":
            True,
    },

    "next_action":
        (
            "MECHANISM_ADJUDICATION_AND_"
            "DECIDE_IF_NSYS_IS_REQUIRED"
        ),

    "verdict":
        verdict,

    "next_milestone":
        "RLTrain-stage-9.5.3",
}


OUTPUT_JSON.write_text(
    json.dumps(
        decision,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "========================================================================================================================"
)

print(
    "RLTrain-stage-9.5.2 TARGETED TRACE ANALYSIS"
)

print(
    "========================================================================================================================"
)

print()

print(
    "DISAGG_STEP_MEAN_S = "
    + str(
        A[
            "means"
        ][
            "step_duration_s"
        ]
    )
)

print(
    "DISAGG_GENERATION_MEAN_S = "
    + str(
        A[
            "means"
        ][
            "generation_union_s"
        ]
    )
)

print(
    "DISAGG_ACTOR_UPDATE_MEAN_S = "
    + str(
        A[
            "means"
        ][
            "actor_update_union_s"
        ]
    )
)

print(
    "DISAGG_GENERATION_ACTOR_OVERLAP_MEAN_S = "
    + str(
        disagg_overlap
    )
)

print(
    "DISAGG_GENERATION_ACTOR_OVERLAP_FRACTION_OF_STEP = "
    + str(
        disagg_overlap_fraction
    )
)

print(
    "DISAGG_PHASE_UNION_COVERAGE_FRACTION = "
    + str(
        A[
            "means"
        ][
            "phase_union_coverage_fraction"
        ]
    )
)

print(
    "DISAGG_UNCOVERED_WALL_MEAN_S = "
    + str(
        A[
            "means"
        ][
            "uncovered_wall_s"
        ]
    )
)

print()

print(
    "COLOCATED_STEP_MEAN_S = "
    + str(
        B[
            "means"
        ][
            "step_duration_s"
        ]
    )
)

print(
    "COLOCATED_GENERATION_MEAN_S = "
    + str(
        B[
            "means"
        ][
            "generation_union_s"
        ]
    )
)

print(
    "COLOCATED_ACTOR_UPDATE_MEAN_S = "
    + str(
        B[
            "means"
        ][
            "actor_update_union_s"
        ]
    )
)

print(
    "COLOCATED_PHASE_UNION_COVERAGE_FRACTION = "
    + str(
        B[
            "means"
        ][
            "phase_union_coverage_fraction"
        ]
    )
)

print(
    "COLOCATED_UNCOVERED_WALL_MEAN_S = "
    + str(
        colocated_uncovered
    )
)

print(
    "COLOCATED_UNCOVERED_WALL_FRACTION_OF_STEP = "
    + str(
        colocated_uncovered_fraction
    )
)

print()

print(
    "COLOCATED_LARGEST_UNCOVERED_GAP = "
    + str(
        largest_colocated_gap
    )
)

print()

print(
    "DIRECT_INTERVAL_OVERLAP_OBSERVATION = TRUE"
)

print(
    "DIRECT_UNCOVERED_WALL_OBSERVATION = TRUE"
)

print(
    "CAUSAL_MECHANISM_CLAIM = FALSE"
)

print(
    "FORMAL_PERFORMANCE_CLAIM = FALSE"
)

print()

print(
    "CLASSIFICATION = "
    + classification
)

print(
    "NEXT_ACTION = "
    + decision[
        "next_action"
    ]
)

print(
    "FINAL_VERDICT = "
    + verdict
)

print(
    "JSON = "
    + str(
        OUTPUT_JSON
    )
)


report = [
    "# RLTrain-stage-9.5.2 — Targeted Phase Event Trace",
    "",
    f"Verdict: `{verdict}`",
    "",
    "## Disaggregated",
    "",
    (
        f"- Mean measured step: "
        f"{A['means']['step_duration_s']:.6f} s"
    ),
    (
        f"- Generation / actor direct overlap: "
        f"{disagg_overlap:.6f} s"
    ),
    (
        f"- Generation / actor overlap fraction of step: "
        f"{100.0 * disagg_overlap_fraction:.2f}%"
    ),
    (
        f"- Phase-union coverage: "
        f"{100.0 * A['means']['phase_union_coverage_fraction']:.2f}%"
    ),
    "",
    "## Colocated",
    "",
    (
        f"- Mean measured step: "
        f"{B['means']['step_duration_s']:.6f} s"
    ),
    (
        f"- Phase-union coverage: "
        f"{100.0 * B['means']['phase_union_coverage_fraction']:.2f}%"
    ),
    (
        f"- Uncovered wall: "
        f"{colocated_uncovered:.6f} s"
    ),
    "",
    "## Boundary",
    "",
    "- One traced process-level run per architecture",
    "- Two measured post-warmup steps per architecture",
    "- Formal performance claim: false",
    "- Causal mechanism claim: false",
]


REPORT_MD.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8",
)
