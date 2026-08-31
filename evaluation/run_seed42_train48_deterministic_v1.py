from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import traceback

from collections import Counter
from pathlib import Path

import numpy as np
import torch


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


from bridge.cpp_client import (
    RLServerProcessError,
    RLServerResourceAbort,
)

from bridge.resource_guard_v1 import (
    GIB,
    ResourceGuardPolicyV1,
)

from envs.loopycuts_env import (
    LoopyCutsEnv,
)

from evaluation.deterministic_actor_v1 import (
    DETERMINISTIC_ACTOR_VERSION,
    select_deterministic_actor_action,
)

from networks.loopycuts_actor_critic_v1 import (
    LoopyCutsActorV1,
    count_trainable_parameters,
)

from observation.builder import (
    MAX_LOOPS,
)

from training.formal_training_input_provenance_v1 import (
    compute_train49_inputs,
)

from training.formal_training_v1 import (
    DEFAULT_DATASET_MANIFEST,
    configure_formal_training_runtime,
    load_formal_stage2_models,
    set_formal_training_seed,
)

from training.protocol_v1 import (
    PROJECT_STAGE2_RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_WARNING_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_HOLD_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_EMERGENCY_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_REARM_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_QUALITY_SWAP_ABORT_GIB,
)

from training.run_formal_training_v1 import (
    wait_for_formal_resource_rearm,
)


EVALUATOR_VERSION = (
    "loopycuts_seed42_train48_deterministic_evaluator_v1"
)

RESULT_SCHEMA_VERSION = (
    "loopycuts_seed42_train48_model_result_v1"
)

RUN_MANIFEST_SCHEMA_VERSION = (
    "loopycuts_seed42_train48_run_manifest_v1"
)

SUMMARY_SCHEMA_VERSION = (
    "loopycuts_seed42_train48_summary_v1"
)


SEED = 42

TRAINING_DATASET_NAME = "Train49"

OPERATIONAL_DATASET_NAME = "Train48"

EXCLUDED_MODEL = "boat"


EVALUATION_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "evaluation_v5_seed42"
)


ACTOR_PATH = (
    EVALUATION_ROOT
    /
    "actor"
    /
    "seed42_actor_v5.pt"
)


EVALUATION_EXECUTABLE = (
    EVALUATION_ROOT
    /
    "frozen_evaluation_inputs"
    /
    (
        "volumetric_cutter_eval_"
        "53e4ec4c137e9c959abcb2ceac91b039"
        "02a17f9577457970de0e506150f34875"
    )
)


EVALUATION_PROVENANCE = (
    EVALUATION_ROOT
    /
    "evaluation_provenance_v1.json"
)


QUALITY_REF_SET_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "quality_refs_train49_v1"
)

QUALITY_REF_SHA256SUMS = (
    QUALITY_REF_SET_ROOT
    /
    "SHA256SUMS.txt"
)


DEFAULT_RUN_ROOT = (
    EVALUATION_ROOT
    /
    "train48_deterministic_v1"
)


EXPECTED_ACTOR_SHA256 = (
    "6486910b923818c9197bf2a69a41d5f"
    "dbd3161470968467e6ec60711d9fac86a"
)

EXPECTED_EVALUATION_EXECUTABLE_SHA256 = (
    "53e4ec4c137e9c959abcb2ceac91b039"
    "02a17f9577457970de0e506150f34875"
)

EXPECTED_EVALUATION_PROVENANCE_SHA256 = (
    "55b5fba4fc028af1b114cece3a76a7a1"
    "5d6d2742f124ac0c545ddbe4c61c6b3e"
)

EXPECTED_TRAIN49_INPUT_AGGREGATE_SHA256 = (
    "a1e68312f05457e2f3ecb92e7b59fa9"
    "3facbc57850833b4f8931a7143f55d42d"
)

EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY = (
    "484d08dc5bad32dd4dab5721251969609"
    "0dcf1b6ad6d973cd818d4a4b512b8a0"
)

EXPECTED_ACTOR_PARAMETER_COUNT = (
    184_577
)


# ======================================================================
# Basic immutable-artifact helpers.
# ======================================================================

def sha256_file(
    path: Path,
) -> str:

    path = Path(
        path
    )

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def git_head(
    repo: Path,
) -> str:

    return subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def git_is_clean(
    repo: Path,
) -> bool:

    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    )

    return (
        output.strip()
        ==
        ""
    )


def to_jsonable(
    value,
):

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                to_jsonable(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            to_jsonable(
                item
            )
            for item
            in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return (
            value.tolist()
        )

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    return value


def atomic_write_json(
    path: Path,
    payload,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name
        +
        ".tmp"
    )

    tmp.write_text(
        json.dumps(
            to_jsonable(
                payload
            ),
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def write_once_json(
    path: Path,
    payload,
):

    path = Path(
        path
    )

    if path.exists():
        raise RuntimeError(
            "Refusing to overwrite "
            f"existing evidence: {path}"
        )

    atomic_write_json(
        path,
        payload,
    )


# ======================================================================
# Frozen actor.
# ======================================================================

def load_frozen_actor():

    if not ACTOR_PATH.is_file():
        raise FileNotFoundError(
            ACTOR_PATH
        )

    actual_sha = sha256_file(
        ACTOR_PATH
    )

    if (
        actual_sha
        !=
        EXPECTED_ACTOR_SHA256
    ):
        raise RuntimeError(
            "Actor SHA256 mismatch"
        )

    artifact = torch.load(
        str(
            ACTOR_PATH
        ),
        map_location="cpu",
        weights_only=False,
    )

    if (
        artifact.get(
            "schema_version"
        )
        !=
        "loopycuts_actor_only_v5_v1"
    ):
        raise RuntimeError(
            "Actor artifact schema mismatch"
        )

    if int(
        artifact.get(
            "seed",
            -1,
        )
    ) != SEED:
        raise RuntimeError(
            "Actor seed mismatch"
        )

    if (
        artifact.get(
            "deterministic_eval"
        )
        is not True
    ):
        raise RuntimeError(
            "Actor artifact is not "
            "frozen for deterministic evaluation"
        )

    if float(
        artifact.get(
            "exploration_epsilon",
            -1.0,
        )
    ) != 0.0:
        raise RuntimeError(
            "Actor artifact epsilon is not zero"
        )

    if (
        "actor_state_dict"
        not in artifact
    ):
        raise RuntimeError(
            "Actor artifact lacks actor_state_dict"
        )

    #
    # Important:
    #
    # Instantiate ONLY the Actor.
    # Do not allocate Critic-1 or Critic-2 during evaluation.
    #
    actor = (
        LoopyCutsActorV1()
        .to(
            "cpu"
        )
    )

    actor.load_state_dict(
        artifact[
            "actor_state_dict"
        ],
        strict=True,
    )

    actor.eval()

    parameter_count = (
        count_trainable_parameters(
            actor
        )
    )

    if (
        parameter_count
        !=
        EXPECTED_ACTOR_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Actor parameter-count mismatch: "
            f"{parameter_count}"
        )

    return (
        actor,
        artifact,
    )


# ======================================================================
# Frozen Train49 -> operational Train48.
# ======================================================================

def select_operational_models(
    models,
):

    models = tuple(
        models
    )

    if len(
        models
    ) != 49:
        raise RuntimeError(
            "Expected original Train49 "
            f"but received {len(models)} models"
        )

    boat_models = [
        model
        for model in models
        if (
            model.model
            ==
            EXCLUDED_MODEL
        )
    ]

    if len(
        boat_models
    ) != 1:
        raise RuntimeError(
            "Train49 must contain exactly "
            "one boat model"
        )

    operational = tuple(
        sorted(
            (
                model
                for model in models
                if (
                    model.model
                    !=
                    EXCLUDED_MODEL
                )
            ),
            key=lambda item:
                item.model,
        )
    )

    if len(
        operational
    ) != 48:
        raise RuntimeError(
            "Operational Train48 model "
            "count mismatch"
        )

    if any(
        model.model == EXCLUDED_MODEL
        for model in operational
    ):
        raise RuntimeError(
            "boat leaked into operational Train48"
        )

    return operational


# ======================================================================
# ResourceGuard.
# ======================================================================

def make_evaluation_resource_guard():

    return ResourceGuardPolicyV1(
        warning_swap_used_bytes=
            (
                PROJECT_STAGE2_RESOURCE_GUARD_WARNING_SWAP_GIB
                *
                GIB
            ),

        abort_swap_used_bytes=
            (
                PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SWAP_GIB
                *
                GIB
            ),

        emergency_swap_used_bytes=
            (
                PROJECT_STAGE2_RESOURCE_GUARD_EMERGENCY_SWAP_GIB
                *
                GIB
            ),

        rearm_swap_used_bytes=
            (
                PROJECT_STAGE2_RESOURCE_GUARD_REARM_SWAP_GIB
                *
                GIB
            ),

        abort_hold_seconds=
            PROJECT_STAGE2_RESOURCE_GUARD_ABORT_HOLD_SECONDS,
    )


def resource_abort_from_exception(
    exc: RLServerResourceAbort,
):

    snapshot = exc.snapshot

    cpp_rss = 0
    cpp_swap = 0

    if snapshot.cpp_memory is not None:

        cpp_rss = int(
            snapshot.cpp_memory.rss_bytes
        )

        cpp_swap = int(
            snapshot.cpp_memory.swap_bytes
        )

    return {
        "outcome":
            "RESOURCE_ABORT",

        "phase":
            str(
                exc.phase
            ),

        "guard_state":
            str(
                exc.guard_state
            ),

        "return_code":
            (
                None
                if exc.return_code is None
                else
                int(
                    exc.return_code
                )
            ),

        "swap_used_bytes":
            int(
                snapshot.swap_used_bytes
            ),

        "swap_total_bytes":
            int(
                snapshot.swap_total_bytes
            ),

        "swap_free_bytes":
            int(
                snapshot.swap_free_bytes
            ),

        "mem_available_bytes":
            int(
                snapshot.mem_available_bytes
            ),

        "python_rss_bytes":
            int(
                snapshot.python_memory.rss_bytes
            ),

        "python_swap_bytes":
            int(
                snapshot.python_memory.swap_bytes
            ),

        "cpp_rss_bytes":
            cpp_rss,

        "cpp_swap_bytes":
            cpp_swap,
    }


# ======================================================================
# Evaluation records.
# ======================================================================

def legal_mask_sha256(
    mask,
):

    mask = np.asarray(
        mask,
        dtype=np.bool_,
    )

    if mask.shape != (
        MAX_LOOPS,
    ):
        raise RuntimeError(
            "Unexpected legal-mask shape"
        )

    return hashlib.sha256(
        mask.astype(
            np.uint8
        ).tobytes()
    ).hexdigest()


def geometry_manifest(
    root: Path,
):

    root = Path(
        root
    )

    if not root.is_dir():
        return []

    records = []

    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
    ):

        records.append(
            {
                "relative_path":
                    str(
                        path.relative_to(
                            root
                        )
                    ),

                "bytes":
                    int(
                        path.stat().st_size
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    return records


def next_attempt_directory(
    run_root: Path,
    model_name: str,
):

    model_root = (
        Path(
            run_root
        )
        /
        "models"
        /
        model_name
    )

    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    index = 1

    while True:

        candidate = (
            model_root
            /
            f"attempt_{index:03d}"
        )

        if not candidate.exists():

            candidate.mkdir(
                parents=False,
                exist_ok=False,
            )

            return (
                candidate,
                index,
            )

        index += 1


def base_model_result(
    *,
    model,
    runner_commit: str,
    attempt_index: int,
    attempt_directory: Path,
):

    return {
        "schema_version":
            RESULT_SCHEMA_VERSION,

        "evaluator_version":
            EVALUATOR_VERSION,

        "seed":
            SEED,

        "training_protocol_dataset":
            TRAINING_DATASET_NAME,

        "operational_evaluation_dataset":
            OPERATIONAL_DATASET_NAME,

        "operational_exclusion":
            EXCLUDED_MODEL,

        "model":
            model.model,

        "complexity_stratum":
            int(
                model.complexity_stratum
            ),

        "mesh_file":
            str(
                model.mesh_file
            ),

        "loop_file":
            str(
                model.loop_file
            ),

        "quality_ref_file":
            str(
                model.quality_ref_file
            ),

        "runner_git_commit":
            runner_commit,

        "actor_sha256":
            EXPECTED_ACTOR_SHA256,

        "evaluation_executable_sha256":
            EXPECTED_EVALUATION_EXECUTABLE_SHA256,

        "deterministic":
            True,

        "exploration_epsilon":
            0.0,

        "attempt_index":
            int(
                attempt_index
            ),

        "attempt_directory":
            str(
                attempt_directory
            ),
    }


def run_one_model(
    *,
    actor,
    model,
    runner_commit: str,
    attempt_index: int,
    attempt_directory: Path,
):

    geometry_dir = (
        attempt_directory
        /
        "geometry"
    )

    geometry_dir.mkdir(
        parents=False,
        exist_ok=False,
    )

    result = base_model_result(
        model=model,
        runner_commit=runner_commit,
        attempt_index=attempt_index,
        attempt_directory=attempt_directory,
    )

    steps = []

    env = LoopyCutsEnv(
        executable=
            EVALUATION_EXECUTABLE,

        mesh_file=
            model.mesh_file,

        loop_file=
            model.loop_file,

        echo_logs=
            False,

        resource_guard_policy=
            make_evaluation_resource_guard(),

        resource_guard_sample_interval_seconds=
            PROJECT_STAGE2_RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,

        finalize_eval_swap_abort_bytes=
            (
                PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_QUALITY_SWAP_ABORT_GIB
                *
                GIB
            ),
    )

    try:

        (
            observation,
            info,
        ) = env.reset(
            seed=SEED
        )

        initial_state = dict(
            info[
                "state"
            ]
        )

        result[
            "initial_state"
        ] = initial_state

        # ----------------------------------------------------------
        # One deterministic policy trajectory.
        # ----------------------------------------------------------

        while not bool(
            info[
                "state"
            ][
                "terminal"
            ]
        ):

            if len(
                steps
            ) >= MAX_LOOPS:
                raise RuntimeError(
                    "Episode exceeded MAX_LOOPS "
                    "without reaching C++ terminal"
                )

            legal_actions = tuple(
                int(value)
                for value
                in env.legal_actions
            )

            mask = np.asarray(
                observation[
                    "mask"
                ],
                dtype=np.bool_,
            )

            mask_actions = tuple(
                int(value)
                for value
                in np.flatnonzero(
                    mask
                ).tolist()
            )

            if (
                mask_actions
                !=
                legal_actions
            ):
                raise RuntimeError(
                    "Observation mask differs "
                    "from authoritative C++ ACTIONS"
                )

            if not legal_actions:
                raise RuntimeError(
                    "Non-terminal state has "
                    "no legal C++ ACTIONS"
                )

            decision = (
                select_deterministic_actor_action(
                    actor,
                    observation,
                )
            )

            action = int(
                decision[
                    "action"
                ]
            )

            if action not in legal_actions:
                raise RuntimeError(
                    "Actor-only inference selected "
                    "an illegal C++ action"
                )

            state_before = dict(
                env.current_state
                or
                {}
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                next_info,
            ) = env.step(
                action
            )

            if truncated:
                raise RuntimeError(
                    "Unexpected truncated=True "
                    "during formal evaluation"
                )

            step_record = {
                "step_index":
                    len(
                        steps
                    )
                    +
                    1,

                "action":
                    action,

                "selected_logit":
                    float(
                        decision[
                            "selected_logit"
                        ]
                    ),

                "legal_action_count":
                    len(
                        legal_actions
                    ),

                "legal_actions":
                    list(
                        legal_actions
                    ),

                "legal_mask_sha256":
                    legal_mask_sha256(
                        mask
                    ),

                "state_before":
                    state_before,

                "state_after":
                    next_info.get(
                        "state"
                    ),

                "step_result":
                    next_info.get(
                        "step_result"
                    ),

                "transition_metrics":
                    next_info.get(
                        "transition_metrics"
                    ),

                "selection_reward":
                    float(
                        reward
                    ),

                "reward_breakdown":
                    next_info.get(
                        "reward_breakdown"
                    ),

                "terminated":
                    bool(
                        terminated
                    ),
            }

            if (
                "resource_abort"
                in next_info
            ):
                step_record[
                    "resource_abort"
                ] = next_info[
                    "resource_abort"
                ]

            steps.append(
                to_jsonable(
                    step_record
                )
            )

            observation = (
                next_observation
            )

            info = (
                next_info
            )

            # ------------------------------------------------------
            # STEP-time ResourceAbort is already converted by
            # LoopyCutsEnv into a terminal transition.
            # Do NOT attempt finalization afterwards.
            # ------------------------------------------------------

            if (
                "resource_abort"
                in next_info
            ):

                result.update(
                    {
                        "outcome":
                            "RESOURCE_ABORT",

                        "resource_abort":
                            to_jsonable(
                                next_info[
                                    "resource_abort"
                                ]
                            ),

                        "steps":
                            steps,

                        "action_sequence":
                            [
                                int(
                                    item[
                                        "action"
                                    ]
                                )
                                for item
                                in steps
                            ],

                        "selection_terminal_state":
                            to_jsonable(
                                next_info[
                                    "state"
                                ]
                            ),

                        "final_result":
                            None,

                        "quality":
                            None,

                        "utility":
                            None,

                        "geometry_files":
                            geometry_manifest(
                                geometry_dir
                            ),
                    }
                )

                return result

            if terminated:
                break


        # ----------------------------------------------------------
        # Normal C++ terminal state.
        #
        # Run geometry + quality exactly once.
        # ----------------------------------------------------------

        selection_terminal_state = dict(
            info[
                "state"
            ]
        )

        try:

            (
                final_result,
                quality,
                final_state,
            ) = env.client.finalize_quality_export(
                model.quality_ref_file,
                geometry_dir,
            )

        except RLServerResourceAbort as exc:

            result.update(
                {
                    "outcome":
                        "RESOURCE_ABORT",

                    "resource_abort":
                        resource_abort_from_exception(
                            exc
                        ),

                    "steps":
                        steps,

                    "action_sequence":
                        [
                            int(
                                item[
                                    "action"
                                ]
                            )
                            for item
                            in steps
                        ],

                    "selection_terminal_state":
                        selection_terminal_state,

                    "final_result":
                        None,

                    "quality":
                        None,

                    "utility":
                        None,

                    "geometry_files":
                        geometry_manifest(
                            geometry_dir
                        ),
                }
            )

            return result

        except RLServerProcessError as exc:

            #
            # Frozen V5 semantics:
            #
            # Only a genuine nonzero C++ process exit during
            # finalization is classified as FINALIZATION_CRASH.
            #
            if (
                exc.return_code
                is None
                or
                int(
                    exc.return_code
                )
                ==
                0
            ):
                raise

            if (
                str(
                    exc.phase
                )
                !=
                "FINALIZE_QUALITY_EXPORT"
            ):
                raise

            result.update(
                {
                    "outcome":
                        "FINALIZATION_CRASH",

                    "finalization_crash":
                        {
                            "phase":
                                str(
                                    exc.phase
                                ),

                            "return_code":
                                int(
                                    exc.return_code
                                ),

                            "signal_number":
                                exc.signal_number,

                            "signal_name":
                                exc.signal_name,

                            "output_tail":
                                list(
                                    exc.lines[
                                        -100:
                                    ]
                                ),
                        },

                    "steps":
                        steps,

                    "action_sequence":
                        [
                            int(
                                item[
                                    "action"
                                ]
                            )
                            for item
                            in steps
                        ],

                    "selection_terminal_state":
                        selection_terminal_state,

                    "final_result":
                        None,

                    "quality":
                        None,

                    "utility":
                        None,

                    "geometry_files":
                        geometry_manifest(
                            geometry_dir
                        ),
                }
            )

            return result


        # ----------------------------------------------------------
        # Successful finalization contract.
        # ----------------------------------------------------------

        if (
            quality.get(
                "model"
            )
            !=
            model.model
        ):
            raise RuntimeError(
                "Quality record model mismatch"
            )

        if (
            int(
                quality[
                    "hex"
                ]
            )
            !=
            int(
                final_result[
                    "hex"
                ]
            )
        ):
            raise RuntimeError(
                "FINAL_RESULT/quality hex mismatch"
            )

        if (
            int(
                quality[
                    "total_polys"
                ]
            )
            !=
            int(
                final_result[
                    "total_polys"
                ]
            )
        ):
            raise RuntimeError(
                "FINAL_RESULT/quality total_polys mismatch"
            )

        expected_nonhex = (
            int(
                final_result[
                    "total_polys"
                ]
            )
            -
            int(
                final_result[
                    "hex"
                ]
            )
        )

        if (
            int(
                quality[
                    "nonhex"
                ]
            )
            !=
            expected_nonhex
        ):
            raise RuntimeError(
                "Quality nonhex mismatch"
            )

        full_hex = int(
            final_result[
                "full_hex"
            ]
        )

        if full_hex not in (
            0,
            1,
        ):
            raise RuntimeError(
                "Invalid full_hex flag"
            )

        geometry_files = (
            geometry_manifest(
                geometry_dir
            )
        )

        if len(
            geometry_files
        ) < 3:
            raise RuntimeError(
                "Successful finalization exported "
                "fewer than three geometry files"
            )

        hex_mesh_files = [
            item
            for item
            in geometry_files
            if (
                item[
                    "relative_path"
                ].endswith(
                    "_hex.mesh"
                )
            )
        ]

        if (
            full_hex
            and
            len(
                hex_mesh_files
            )
            !=
            1
        ):
            raise RuntimeError(
                "FULL_HEX result lacks exactly "
                "one exported _hex.mesh"
            )

        utility = (
            float(
                quality[
                    "d_c"
                ]
            )
            *
            float(
                quality[
                    "q_fidelity"
                ]
            )
        )

        result.update(
            {
                "outcome":
                    (
                        "FULL_HEX"
                        if full_hex
                        else
                        "NON_FULL_HEX"
                    ),

                "steps":
                    steps,

                "action_sequence":
                    [
                        int(
                            item[
                                "action"
                            ]
                        )
                        for item
                        in steps
                    ],

                "selection_terminal_state":
                    selection_terminal_state,

                "final_result":
                    to_jsonable(
                        final_result
                    ),

                "quality":
                    to_jsonable(
                        quality
                    ),

                "utility":
                    float(
                        utility
                    ),

                "final_state":
                    to_jsonable(
                        final_state
                    ),

                "geometry_files":
                    geometry_files,
            }
        )

        return result

    finally:

        env.close()


# ======================================================================
# Summary.
# ======================================================================

def summarize_records(
    records,
):

    records = list(
        records
    )

    outcomes = Counter(
        str(
            record[
                "outcome"
            ]
        )
        for record
        in records
    )

    completed_finalizations = [
        record
        for record
        in records
        if (
            record[
                "outcome"
            ]
            in
            {
                "FULL_HEX",
                "NON_FULL_HEX",
            }
        )
    ]

    d_c_values = [
        float(
            record[
                "quality"
            ][
                "d_c"
            ]
        )
        for record
        in completed_finalizations
    ]

    q_values = [
        float(
            record[
                "quality"
            ][
                "q_fidelity"
            ]
        )
        for record
        in completed_finalizations
    ]

    utility_values = [
        float(
            record[
                "utility"
            ]
        )
        for record
        in completed_finalizations
    ]

    total_polys = sum(
        int(
            record[
                "quality"
            ][
                "total_polys"
            ]
        )
        for record
        in completed_finalizations
    )

    nonhex = sum(
        int(
            record[
                "quality"
            ][
                "nonhex"
            ]
        )
        for record
        in completed_finalizations
    )

    def mean_or_none(
        values,
    ):
        if not values:
            return None

        return float(
            statistics.mean(
                values
            )
        )

    def median_or_none(
        values,
    ):
        if not values:
            return None

        return float(
            statistics.median(
                values
            )
        )

    return {
        "schema_version":
            SUMMARY_SCHEMA_VERSION,

        "seed":
            SEED,

        "expected_models":
            48,

        "completed_model_records":
            len(
                records
            ),

        "complete":
            (
                len(
                    records
                )
                ==
                48
            ),

        "outcomes":
            {
                key:
                    int(
                        value
                    )
                for key, value
                in sorted(
                    outcomes.items()
                )
            },

        "full_hex_count":
            int(
                outcomes.get(
                    "FULL_HEX",
                    0,
                )
            ),

        "full_hex_rate_over_operational_train48":
            (
                None
                if not records
                else
                float(
                    outcomes.get(
                        "FULL_HEX",
                        0,
                    )
                    /
                    48.0
                )
            ),

        "completed_finalization_count":
            len(
                completed_finalizations
            ),

        "aggregate_nonhex_fraction":
            (
                None
                if total_polys <= 0
                else
                float(
                    nonhex
                    /
                    total_polys
                )
            ),

        "mean_d_c":
            mean_or_none(
                d_c_values
            ),

        "median_d_c":
            median_or_none(
                d_c_values
            ),

        "mean_q_fidelity":
            mean_or_none(
                q_values
            ),

        "median_q_fidelity":
            median_or_none(
                q_values
            ),

        "mean_utility":
            mean_or_none(
                utility_values
            ),

        "median_utility":
            median_or_none(
                utility_values
            ),
    }


# ======================================================================
# Run manifest / provenance.
# ======================================================================

def verify_quality_ref_checksum_set():

    if not QUALITY_REF_SHA256SUMS.is_file():
        raise FileNotFoundError(
            QUALITY_REF_SHA256SUMS
        )

    identity = sha256_file(
        QUALITY_REF_SHA256SUMS
    )

    if (
        identity
        !=
        EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY
    ):
        raise RuntimeError(
            "Quality-ref SHA256SUMS "
            "identity mismatch"
        )

    checksums = {}

    for raw_line in (
        QUALITY_REF_SHA256SUMS
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    ):

        line = raw_line.strip()

        if not line:
            continue

        parts = line.split(
            None,
            1,
        )

        if len(
            parts
        ) != 2:
            raise RuntimeError(
                "Invalid SHA256SUMS line"
            )

        expected = (
            parts[
                0
            ]
            .strip()
        )

        relative = (
            parts[
                1
            ]
            .strip()
            .lstrip(
                "*"
            )
        )

        path = (
            QUALITY_REF_SET_ROOT
            /
            relative
        ).resolve()

        if not path.is_relative_to(
            QUALITY_REF_SET_ROOT.resolve()
        ):
            raise RuntimeError(
                "Quality-ref checksum path "
                "escapes frozen root"
            )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        actual = sha256_file(
            path
        )

        if actual != expected:
            raise RuntimeError(
                "Quality-ref SHA mismatch: "
                f"{path}"
            )

        checksums[
            str(
                path
            )
        ] = expected

    if not checksums:
        raise RuntimeError(
            "No frozen quality-ref "
            "checksums were validated"
        )

    return checksums


def build_run_manifest(
    *,
    runtime,
    models,
    runner_commit: str,
):

    if not EVALUATION_EXECUTABLE.is_file():
        raise FileNotFoundError(
            EVALUATION_EXECUTABLE
        )

    if not os.access(
        EVALUATION_EXECUTABLE,
        os.X_OK,
    ):
        raise RuntimeError(
            "Frozen evaluation executable "
            "is not executable"
        )

    if (
        sha256_file(
            EVALUATION_EXECUTABLE
        )
        !=
        EXPECTED_EVALUATION_EXECUTABLE_SHA256
    ):
        raise RuntimeError(
            "Frozen evaluation executable "
            "SHA256 mismatch"
        )

    if not EVALUATION_PROVENANCE.is_file():
        raise FileNotFoundError(
            EVALUATION_PROVENANCE
        )

    if (
        sha256_file(
            EVALUATION_PROVENANCE
        )
        !=
        EXPECTED_EVALUATION_PROVENANCE_SHA256
    ):
        raise RuntimeError(
            "Evaluation provenance "
            "SHA256 mismatch"
        )

    provenance = json.loads(
        EVALUATION_PROVENANCE.read_text(
            encoding="utf-8"
        )
    )

    if (
        provenance[
            "actor"
        ][
            "sha256"
        ]
        !=
        EXPECTED_ACTOR_SHA256
    ):
        raise RuntimeError(
            "Evaluation provenance actor mismatch"
        )

    if (
        provenance[
            "evaluation_software"
        ][
            "evaluation_executable_sha256"
        ]
        !=
        EXPECTED_EVALUATION_EXECUTABLE_SHA256
    ):
        raise RuntimeError(
            "Evaluation provenance "
            "executable mismatch"
        )

    (
        _train49_records,
        train49_aggregate,
    ) = compute_train49_inputs(
        dataset_manifest=
            DEFAULT_DATASET_MANIFEST
    )

    if (
        train49_aggregate
        !=
        EXPECTED_TRAIN49_INPUT_AGGREGATE_SHA256
    ):
        raise RuntimeError(
            "Frozen Train49 mesh/loop "
            "aggregate mismatch"
        )

    quality_checksums = (
        verify_quality_ref_checksum_set()
    )

    model_records = []

    for model in models:

        ref = Path(
            model.quality_ref_file
        ).resolve()

        ref_key = str(
            ref
        )

        if ref_key not in quality_checksums:
            raise RuntimeError(
                "Operational model quality ref "
                "is absent from frozen SHA256SUMS: "
                f"{model.model}"
            )

        model_records.append(
            {
                "model":
                    model.model,

                "complexity_stratum":
                    int(
                        model.complexity_stratum
                    ),

                "mesh_file":
                    str(
                        model.mesh_file
                    ),

                "mesh_sha256":
                    sha256_file(
                        model.mesh_file
                    ),

                "loop_file":
                    str(
                        model.loop_file
                    ),

                "loop_sha256":
                    sha256_file(
                        model.loop_file
                    ),

                "quality_ref_file":
                    ref_key,

                "quality_ref_sha256":
                    quality_checksums[
                        ref_key
                    ],
            }
        )

    return {
        "schema_version":
            RUN_MANIFEST_SCHEMA_VERSION,

        "evaluator_version":
            EVALUATOR_VERSION,

        "deterministic_actor_version":
            DETERMINISTIC_ACTOR_VERSION,

        "seed":
            SEED,

        "runner_git_commit":
            runner_commit,

        "runner_git_clean":
            True,

        "runtime":
            runtime,

        "actor": {
            "path":
                str(
                    ACTOR_PATH
                ),

            "sha256":
                EXPECTED_ACTOR_SHA256,

            "parameter_count":
                EXPECTED_ACTOR_PARAMETER_COUNT,

            "actor_only":
                True,

            "critic_loaded":
                False,

            "deterministic_eval":
                True,

            "exploration_epsilon":
                0.0,
        },

        "evaluation_executable": {
            "path":
                str(
                    EVALUATION_EXECUTABLE
                ),

            "sha256":
                EXPECTED_EVALUATION_EXECUTABLE_SHA256,
        },

        "evaluation_provenance": {
            "path":
                str(
                    EVALUATION_PROVENANCE
                ),

            "sha256":
                EXPECTED_EVALUATION_PROVENANCE_SHA256,
        },

        "dataset": {
            "training_protocol_dataset":
                TRAINING_DATASET_NAME,

            "training_protocol_model_count":
                49,

            "train49_input_aggregate_sha256":
                train49_aggregate,

            "operational_evaluation_dataset":
                OPERATIONAL_DATASET_NAME,

            "operational_model_count":
                48,

            "excluded_model":
                EXCLUDED_MODEL,

            "exclusion_scope":
                (
                    "post-training operational "
                    "evaluation only"
                ),
        },

        "quality_refs": {
            "set":
                "quality_refs_train49_v1",

            "sha256sums_path":
                str(
                    QUALITY_REF_SHA256SUMS
                ),

            "sha256sums_identity":
                EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY,
        },

        "resource_guard": {
            "step_warning_swap_gib":
                PROJECT_STAGE2_RESOURCE_GUARD_WARNING_SWAP_GIB,

            "step_abort_swap_gib":
                PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SWAP_GIB,

            "step_abort_hold_seconds":
                PROJECT_STAGE2_RESOURCE_GUARD_ABORT_HOLD_SECONDS,

            "step_emergency_swap_gib":
                PROJECT_STAGE2_RESOURCE_GUARD_EMERGENCY_SWAP_GIB,

            "preflight_rearm_swap_gib":
                PROJECT_STAGE2_RESOURCE_GUARD_REARM_SWAP_GIB,

            "finalize_swap_abort_gib":
                PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_QUALITY_SWAP_ABORT_GIB,
        },

        "models":
            model_records,
    }


def load_or_create_run_manifest(
    *,
    run_root: Path,
    candidate,
):

    path = (
        Path(
            run_root
        )
        /
        "run_manifest.json"
    )

    if path.is_file():

        existing = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if existing != candidate:
            raise RuntimeError(
                "Existing run_manifest.json "
                "does not match current frozen inputs"
            )

        return path

    write_once_json(
        path,
        candidate,
    )

    return path


def validate_existing_result(
    *,
    result,
    model,
    runner_commit,
):

    checks = {
        "schema_version":
            RESULT_SCHEMA_VERSION,

        "model":
            model.model,

        "runner_git_commit":
            runner_commit,

        "actor_sha256":
            EXPECTED_ACTOR_SHA256,

        "evaluation_executable_sha256":
            EXPECTED_EVALUATION_EXECUTABLE_SHA256,
    }

    for key, expected in checks.items():

        if (
            result.get(
                key
            )
            !=
            expected
        ):
            raise RuntimeError(
                "Existing model result identity "
                f"mismatch: model={model.model} "
                f"field={key}"
            )


def collect_existing_results(
    *,
    run_root: Path,
    models,
    runner_commit: str,
):

    records = []

    result_root = (
        Path(
            run_root
        )
        /
        "results"
    )

    for model in models:

        path = (
            result_root
            /
            (
                model.model
                +
                ".json"
            )
        )

        if not path.is_file():
            continue

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        validate_existing_result(
            result=payload,
            model=model,
            runner_commit=runner_commit,
        )

        records.append(
            payload
        )

    return records


def write_summary(
    *,
    run_root: Path,
    models,
    runner_commit: str,
):

    records = collect_existing_results(
        run_root=run_root,
        models=models,
        runner_commit=runner_commit,
    )

    summary = summarize_records(
        records
    )

    atomic_write_json(
        Path(
            run_root
        )
        /
        "summary.json",
        summary,
    )

    return summary


# ======================================================================
# Main formal evaluation runner.
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    run_root = (
        args.run_root
        .resolve()
    )

    if any(
        ch.isspace()
        for ch
        in str(
            run_root
        )
    ):
        raise RuntimeError(
            "Formal evaluation run_root "
            "must not contain whitespace"
        )


    print(
        "===== FORMAL EVALUATION SOFTWARE ====="
    )

    runner_commit = git_head(
        PROJECT_ROOT
    )

    runner_clean = git_is_clean(
        PROJECT_ROOT
    )

    print(
        "runner_git_commit =",
        runner_commit,
    )

    print(
        "runner_git_clean =",
        runner_clean,
    )

    if not runner_clean:
        raise RuntimeError(
            "Formal Train48 evaluation "
            "requires a clean RL repository"
        )


    print()
    print(
        "===== NUMERICAL RUNTIME ====="
    )

    runtime = (
        configure_formal_training_runtime()
    )

    set_formal_training_seed(
        SEED
    )

    print(
        runtime
    )


    print()
    print(
        "===== FROZEN ACTOR ====="
    )

    (
        actor,
        actor_artifact,
    ) = load_frozen_actor()

    print(
        "actor_sha256 =",
        EXPECTED_ACTOR_SHA256,
    )

    print(
        "actor_parameter_count =",
        count_trainable_parameters(
            actor
        ),
    )

    print(
        "critic_loaded = False"
    )

    print(
        "deterministic = True"
    )

    print(
        "epsilon = 0.0"
    )


    print()
    print(
        "===== OPERATIONAL DATASET ====="
    )

    train49_models = (
        load_formal_stage2_models(
            dataset_manifest=
                DEFAULT_DATASET_MANIFEST
        )
    )

    models = select_operational_models(
        train49_models
    )

    print(
        "training_protocol_models =",
        len(
            train49_models
        ),
    )

    print(
        "operational_models =",
        len(
            models
        ),
    )

    print(
        "excluded_model =",
        EXCLUDED_MODEL,
    )

    print(
        "models ="
    )

    for index, model in enumerate(
        models,
        start=1,
    ):
        print(
            f"{index:02d} "
            f"{model.model} "
            f"stratum="
            f"{model.complexity_stratum}"
        )


    print()
    print(
        "===== BUILD RUN MANIFEST ====="
    )

    candidate_manifest = (
        build_run_manifest(
            runtime=runtime,
            models=models,
            runner_commit=runner_commit,
        )
    )

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        load_or_create_run_manifest(
            run_root=run_root,
            candidate=candidate_manifest,
        )
    )

    print(
        "run_manifest =",
        manifest_path,
    )

    print(
        "run_manifest_sha256 =",
        sha256_file(
            manifest_path
        ),
    )


    if args.preflight_only:

        print()
        print(
            "===== PREFLIGHT PASS ====="
        )

        print(
            "No C++ evaluation episode "
            "was started."
        )

        return


    # ==============================================================
    # Exact one completed evaluation result per operational model.
    # ==============================================================

    result_root = (
        run_root
        /
        "results"
    )

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    for index, model in enumerate(
        models,
        start=1,
    ):

        result_path = (
            result_root
            /
            (
                model.model
                +
                ".json"
            )
        )

        print()
        print(
            "================================================"
        )

        print(
            f"MODEL {index:02d}/48: "
            f"{model.model}"
        )

        if result_path.is_file():

            existing = json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )

            validate_existing_result(
                result=existing,
                model=model,
                runner_commit=runner_commit,
            )

            print(
                "SKIP: durable result already exists; "
                f"outcome={existing['outcome']}"
            )

            continue


        # ----------------------------------------------------------
        # Same <=6 GiB global re-arm requirement used by formal
        # training before every new LoopyCuts model.
        # ----------------------------------------------------------

        preflight = (
            wait_for_formal_resource_rearm(
                emit_logs=True
            )
        )

        print(
            "resource_preflight =",
            preflight,
        )


        (
            attempt_directory,
            attempt_index,
        ) = next_attempt_directory(
            run_root,
            model.model,
        )

        print(
            "attempt =",
            attempt_index,
        )

        print(
            "attempt_directory =",
            attempt_directory,
        )


        try:

            result = run_one_model(
                actor=actor,
                model=model,
                runner_commit=runner_commit,
                attempt_index=attempt_index,
                attempt_directory=attempt_directory,
            )

        except Exception as exc:

            #
            # Unknown software/infrastructure failures are fatal.
            # Preserve their evidence but do NOT fabricate a model
            # evaluation result and do NOT silently continue.
            #
            fatal_record = {
                "schema_version":
                    "loopycuts_train48_fatal_attempt_v1",

                "model":
                    model.model,

                "attempt_index":
                    attempt_index,

                "runner_git_commit":
                    runner_commit,

                "exception_type":
                    type(
                        exc
                    ).__name__,

                "exception_message":
                    str(
                        exc
                    ),

                "traceback":
                    traceback.format_exc(),

                "geometry_files":
                    geometry_manifest(
                        attempt_directory
                        /
                        "geometry"
                    ),
            }

            write_once_json(
                attempt_directory
                /
                "fatal_error.json",
                fatal_record,
            )

            raise


        result[
            "resource_preflight"
        ] = preflight

        write_once_json(
            result_path,
            result,
        )

        print(
            "outcome =",
            result[
                "outcome"
            ],
        )

        print(
            "steps =",
            len(
                result[
                    "action_sequence"
                ]
            ),
        )

        if (
            result[
                "quality"
            ]
            is not None
        ):

            print(
                "D_C =",
                result[
                    "quality"
                ][
                    "d_c"
                ],
            )

            print(
                "q_fidelity =",
                result[
                    "quality"
                ][
                    "q_fidelity"
                ],
            )

            print(
                "utility =",
                result[
                    "utility"
                ],
            )

        print(
            "geometry_file_count =",
            len(
                result[
                    "geometry_files"
                ]
            ),
        )


        summary = write_summary(
            run_root=run_root,
            models=models,
            runner_commit=runner_commit,
        )

        print(
            "completed_model_records =",
            summary[
                "completed_model_records"
            ],
        )


    # ==============================================================
    # Final completeness audit.
    # ==============================================================

    summary = write_summary(
        run_root=run_root,
        models=models,
        runner_commit=runner_commit,
    )

    if (
        summary[
            "completed_model_records"
        ]
        !=
        48
    ):
        raise RuntimeError(
            "Train48 evaluation ended without "
            "48 durable model results"
        )

    if (
        summary[
            "complete"
        ]
        is not True
    ):
        raise RuntimeError(
            "Train48 summary completeness mismatch"
        )


    run_complete = {
        "schema_version":
            "loopycuts_seed42_train48_run_complete_v1",

        "evaluator_version":
            EVALUATOR_VERSION,

        "seed":
            SEED,

        "runner_git_commit":
            runner_commit,

        "run_manifest_sha256":
            sha256_file(
                manifest_path
            ),

        "summary":
            summary,
    }

    run_complete_path = (
        run_root
        /
        "RUN_COMPLETE.json"
    )

    if run_complete_path.is_file():

        existing = json.loads(
            run_complete_path.read_text(
                encoding="utf-8"
            )
        )

        if existing != run_complete:
            raise RuntimeError(
                "Existing RUN_COMPLETE.json "
                "does not match current complete run"
            )

    else:

        write_once_json(
            run_complete_path,
            run_complete,
        )


    print()
    print(
        "===== TRAIN48 EVALUATION COMPLETE ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "RUN_COMPLETE =",
        run_complete_path,
    )


if __name__ == "__main__":
    main()
