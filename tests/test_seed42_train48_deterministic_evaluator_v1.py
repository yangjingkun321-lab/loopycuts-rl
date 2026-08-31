from __future__ import annotations

import sys

from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from evaluation.run_seed42_train48_deterministic_v1 import (
    EXPECTED_ACTOR_PARAMETER_COUNT,
    load_frozen_actor,
    select_operational_models,
    summarize_records,
)


# ----------------------------------------------------------------------
# Actor-only loading must instantiate only the actor architecture.
# ----------------------------------------------------------------------

actor, artifact = (
    load_frozen_actor()
)

parameter_count = sum(
    parameter.numel()
    for parameter
    in actor.parameters()
    if parameter.requires_grad
)

assert (
    parameter_count
    ==
    EXPECTED_ACTOR_PARAMETER_COUNT
)

assert int(
    artifact[
        "seed"
    ]
) == 42

print(
    "PASS: direct LoopyCutsActorV1 "
    "strict-loads frozen actor-only artifact"
)


# ----------------------------------------------------------------------
# Operational set is exactly Train49 minus boat.
# ----------------------------------------------------------------------

models = [
    SimpleNamespace(
        model="boat"
    )
]

models.extend(
    SimpleNamespace(
        model=f"model_{index:02d}"
    )
    for index
    in range(
        48
    )
)

selected = (
    select_operational_models(
        models
    )
)

assert len(
    selected
) == 48

assert all(
    model.model != "boat"
    for model
    in selected
)

assert [
    model.model
    for model
    in selected
] == sorted(
    model.model
    for model
    in selected
)

print(
    "PASS: operational model selection "
    "is exactly Train49 minus boat"
)


# ----------------------------------------------------------------------
# Aggregation semantics.
# ----------------------------------------------------------------------

records = [
    {
        "outcome":
            "FULL_HEX",

        "quality": {
            "d_c": 1.0,
            "q_fidelity": 0.8,
            "total_polys": 100,
            "nonhex": 0,
        },

        "utility":
            0.8,
    },

    {
        "outcome":
            "NON_FULL_HEX",

        "quality": {
            "d_c": 0.5,
            "q_fidelity": 0.6,
            "total_polys": 100,
            "nonhex": 20,
        },

        "utility":
            0.3,
    },

    {
        "outcome":
            "RESOURCE_ABORT",

        "quality":
            None,

        "utility":
            None,
    },

    {
        "outcome":
            "FINALIZATION_CRASH",

        "quality":
            None,

        "utility":
            None,
    },
]

summary = summarize_records(
    records
)

assert (
    summary[
        "completed_model_records"
    ]
    ==
    4
)

assert (
    summary[
        "outcomes"
    ]
    ==
    {
        "FINALIZATION_CRASH": 1,
        "FULL_HEX": 1,
        "NON_FULL_HEX": 1,
        "RESOURCE_ABORT": 1,
    }
)

assert (
    summary[
        "completed_finalization_count"
    ]
    ==
    2
)

assert (
    summary[
        "aggregate_nonhex_fraction"
    ]
    ==
    0.1
)

assert (
    summary[
        "mean_d_c"
    ]
    ==
    0.75
)

assert (
    summary[
        "mean_q_fidelity"
    ]
    ==
    0.7
)

assert (
    summary[
        "mean_utility"
    ]
    ==
    0.55
)

assert (
    summary[
        "complete"
    ]
    is False
)

print(
    "PASS: Train48 summary aggregation "
    "handles success/resource/crash outcomes"
)

print(
    "PASS: deterministic evaluator unit contract"
)
