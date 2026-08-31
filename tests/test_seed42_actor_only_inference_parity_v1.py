from __future__ import annotations

import hashlib
import sys

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch


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


from envs.loopycuts_env import LoopyCutsEnv

from evaluation.deterministic_actor_v1 import (
    select_deterministic_actor_action,
)

from networks.loopycuts_actor_critic_v1 import (
    build_loopycuts_actor_critics_v1,
)

from observation.builder import MAX_LOOPS

from policies.masked_discrete_sac import (
    MaskedDiscreteSACPolicy,
)


ACTOR_PATH = Path(
    "/home/yjk/loopycuts_test/"
    "evaluation_v5_seed42/actor/"
    "seed42_actor_v5.pt"
)

EXE = Path(
    "/home/yjk/loopycuts_test/"
    "evaluation_v5_seed42/"
    "frozen_evaluation_inputs/"
    "volumetric_cutter_eval_"
    "53e4ec4c137e9c959abcb2ceac91b039"
    "02a17f9577457970de0e506150f34875"
)

MESH = Path(
    "/home/yjk/codes/LoopyCuts/test_data/"
    "mech10/mech10_rem_splitted.obj"
)

LOOPS = Path(
    "/home/yjk/codes/LoopyCuts/test_data/"
    "mech10/mech10_rem_loop.txt"
)

EXPECTED_ACTOR_SHA = (
    "6486910b923818c9197bf2a69a41d5f"
    "dbd3161470968467e6ec60711d9fac86a"
)

EXPECTED_EXE_SHA = (
    "53e4ec4c137e9c959abcb2ceac91b039"
    "02a17f9577457970de0e506150f34875"
)


def sha256_file(path: Path) -> str:

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


def main():

    print(
        "===== INPUT IDENTITY ====="
    )

    for path in (
        ACTOR_PATH,
        EXE,
        MESH,
        LOOPS,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    actor_sha = sha256_file(
        ACTOR_PATH
    )

    exe_sha = sha256_file(
        EXE
    )

    print(
        "actor_sha256 =",
        actor_sha,
    )

    print(
        "evaluation_exe_sha256 =",
        exe_sha,
    )

    assert (
        actor_sha
        ==
        EXPECTED_ACTOR_SHA
    )

    assert (
        exe_sha
        ==
        EXPECTED_EXE_SHA
    )


    print()
    print(
        "===== LOAD ACTOR-ONLY ====="
    )

    artifact = torch.load(
        str(
            ACTOR_PATH
        ),
        map_location="cpu",
        weights_only=False,
    )

    assert (
        artifact[
            "schema_version"
        ]
        ==
        "loopycuts_actor_only_v5_v1"
    )

    assert int(
        artifact[
            "seed"
        ]
    ) == 42

    assert (
        artifact[
            "deterministic_eval"
        ]
        is True
    )

    assert float(
        artifact[
            "exploration_epsilon"
        ]
    ) == 0.0

    (
        actor,
        _critic1,
        _critic2,
    ) = (
        build_loopycuts_actor_critics_v1(
            device="cpu"
        )
    )

    actor.load_state_dict(
        artifact[
            "actor_state_dict"
        ],
        strict=True,
    )

    actor.eval()

    print(
        "PASS: actor-only strict load"
    )


    print()
    print(
        "===== REAL MECH10 INITIAL STATE ====="
    )

    env = LoopyCutsEnv(
        executable=EXE,
        mesh_file=MESH,
        loop_file=LOOPS,
        echo_logs=False,

        #
        # No STEP is executed in this contract test.
        #
        resource_guard_policy=None,

        finalize_eval_swap_abort_bytes=None,
    )

    try:

        (
            observation,
            info,
        ) = env.reset(
            seed=42
        )

        legal_from_cpp = tuple(
            int(x)
            for x
            in env.legal_actions
        )

        legal_from_mask = tuple(
            int(x)
            for x
            in np.flatnonzero(
                np.asarray(
                    observation[
                        "mask"
                    ],
                    dtype=np.bool_,
                )
            ).tolist()
        )

        assert (
            legal_from_cpp
            ==
            legal_from_mask
        )

        print(
            "legal_actions =",
            legal_from_cpp,
        )

        print(
            "PASS: observation mask "
            "equals C++ ACTIONS exactly"
        )


        print()
        print(
            "===== DIRECT ACTOR-ONLY DECISION ====="
        )

        direct = (
            select_deterministic_actor_action(
                actor,
                observation,
            )
        )

        direct_action = int(
            direct[
                "action"
            ]
        )

        print(
            "direct_action =",
            direct_action,
        )

        print(
            "selected_logit =",
            direct[
                "selected_logit"
            ],
        )


        print()
        print(
            "===== OFFICIAL POLICY DECISION ====="
        )

        policy = (
            MaskedDiscreteSACPolicy(
                actor=actor,

                action_space=
                    gym.spaces.Discrete(
                        MAX_LOOPS
                    ),

                deterministic_eval=
                    True,

                exploration_epsilon=
                    0.0,
            )
        )

        policy.eval()

        batch = Batch(
            obs=Batch(
                obs=
                    observation[
                        "obs"
                    ],

                mask=
                    observation[
                        "mask"
                    ],
            ),

            info=[
                {},
            ],
        )

        with torch.inference_mode():
            policy_output = policy(
                batch
            )

        policy_action = int(
            np.asarray(
                policy_output.act
            ).reshape(
                -1
            )[0]
        )

        print(
            "policy_action =",
            policy_action,
        )

        assert (
            direct_action
            ==
            policy_action
        )

        assert (
            direct_action
            in
            legal_from_cpp
        )

        direct_masked = (
            direct[
                "masked_logits"
            ]
        )

        policy_masked = (
            policy_output[
                "logits"
            ][
                0
            ]
            .detach()
            .cpu()
        )

        assert torch.equal(
            direct_masked,
            policy_masked,
        )

        print(
            "PASS: direct action == "
            "MaskedDiscreteSACPolicy mode"
        )

        print(
            "PASS: masked logits exact"
        )

        print()
        print(
            "===== ACTOR-ONLY INFERENCE PARITY PASS ====="
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
