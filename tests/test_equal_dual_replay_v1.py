import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from tianshou.algorithm.optim import (
    AdamOptimizerFactory,
)

from tianshou.data import (
    ReplayBuffer,
)

from tianshou.utils.torch_utils import (
    policy_within_training_step,
)


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


from algorithms.demo_guided_discrete_sac_v1 import (
    LoopyCutsDemoGuidedDiscreteSACV1,
)

from imitation.demo_replay import (
    load_main_demo_replay,
)

from networks.loopycuts_actor_critic_v1 import (
    build_loopycuts_actor_critics_v1,
)

from observation.builder import (
    MAX_LOOPS,
)

from policies.masked_discrete_sac import (
    MaskedDiscreteSACPolicy,
)


RAW_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw/train"
)

QUALITY = Path(
    "data/manifests/"
    "demo_quality_v1.csv"
)


def main():
    torch.manual_seed(
        31
    )

    np.random.seed(
        31
    )

    # ==========================================================
    # Formal D_demo.
    # ==========================================================

    demo_buffer, records, provenance = (
        load_main_demo_replay(
            raw_root=
                RAW_ROOT,

            quality_manifest=
                QUALITY,

            random_seed=
                31,
        )
    )

    assert len(
        demo_buffer
    ) == 29

    # ==========================================================
    # Temporary synthetic D_expo.
    #
    # For this infrastructure regression we copy complete
    # transitions into an INDEPENDENT ReplayBuffer.
    #
    # This test does NOT claim these are real exploration data.
    # It only validates dual-buffer preprocessing/sampling/update.
    # ==========================================================

    expo_buffer = ReplayBuffer(
        size=64,
        random_seed=37,
    )

    expo_buffer.update(
        demo_buffer
    )

    assert len(
        expo_buffer
    ) == 29

    # The two buffers are distinct objects.
    assert (
        id(
            demo_buffer
        )
        !=
        id(
            expo_buffer
        )
    )

    # ==========================================================
    # Production networks / policy / algorithm.
    # ==========================================================

    actor, critic1, critic2 = (
        build_loopycuts_actor_critics_v1(
            device="cpu"
        )
    )

    policy = (
        MaskedDiscreteSACPolicy(
            actor=
                actor,

            action_space=
                gym.spaces.Discrete(
                    MAX_LOOPS
                ),

            deterministic_eval=
                False,
        )
    )

    algorithm = (
        LoopyCutsDemoGuidedDiscreteSACV1(
            policy=
                policy,

            policy_optim=
                AdamOptimizerFactory(
                    lr=1e-3
                ),

            critic=
                critic1,

            critic_optim=
                AdamOptimizerFactory(
                    lr=1e-3
                ),

            critic2=
                critic2,

            critic2_optim=
                AdamOptimizerFactory(
                    lr=1e-3
                ),

            tau=
                0.005,

            gamma=
                0.99,

            alpha=
                0.2,

            n_step_return_horizon=
                1,

            # Regression-only value.
            bc_weight=
                0.5,

            # Stage II semantics.
            bc_enabled=
                False,
        )
    )

    # ==========================================================
    # One exact 1:1 update:
    #
    # 6 D_demo + 6 D_expo = batch 12.
    # ==========================================================

    with policy_within_training_step(
        policy
    ):
        stats, mix = (
            algorithm.update_equal_replay(
                demo_buffer=
                    demo_buffer,

                expo_buffer=
                    expo_buffer,

                samples_per_buffer=
                    6,
            )
        )

    print("=" * 80)
    print("EQUAL DUAL REPLAY V1")
    print("=" * 80)

    print(
        "demo samples :",
        mix[
            "demo_samples"
        ],
    )

    print(
        "expo samples :",
        mix[
            "expo_samples"
        ],
    )

    print(
        "total        :",
        mix[
            "total_samples"
        ],
    )

    print(
        "actor loss   :",
        stats.actor_loss,
    )

    print(
        "critic1 loss :",
        stats.critic1_loss,
    )

    print(
        "critic2 loss :",
        stats.critic2_loss,
    )

    print(
        "BC loss      :",
        stats.bc_loss,
    )

    assert (
        mix[
            "demo_samples"
        ]
        ==
        6
    )

    assert (
        mix[
            "expo_samples"
        ]
        ==
        6
    )

    assert (
        mix[
            "total_samples"
        ]
        ==
        12
    )

    # Stage II must remain SAC-only.
    assert (
        stats.bc_loss
        ==
        0.0
    )

    assert (
        stats.bc_selected_count
        ==
        0
    )

    values = np.asarray(
        [
            stats.actor_loss,
            stats.critic1_loss,
            stats.critic2_loss,
        ],
        dtype=np.float64,
    )

    assert bool(
        np.isfinite(
            values
        ).all()
    )

    for name, module in [
        (
            "actor",
            actor,
        ),
        (
            "critic1",
            critic1,
        ),
        (
            "critic2",
            critic2,
        ),
    ]:
        for parameter in module.parameters():
            assert bool(
                torch.isfinite(
                    parameter
                ).all()
            ), (
                f"non-finite parameter "
                f"in {name}"
            )

    # ==========================================================
    # Guard:
    # equal-replay Stage-II update must refuse BC-enabled mode.
    # ==========================================================

    algorithm.set_bc_enabled(
        True
    )

    try:
        with policy_within_training_step(
            policy
        ):
            algorithm.update_equal_replay(
                demo_buffer=
                    demo_buffer,

                expo_buffer=
                    expo_buffer,

                samples_per_buffer=
                    2,
            )

    except RuntimeError:
        pass

    else:
        raise RuntimeError(
            "Stage-II dual replay incorrectly "
            "accepted bc_enabled=True"
        )

    print()
    print(
        "PASS: Stage-II equal dual replay "
        "uses exactly 1:1 D_demo/D_expo "
        "for one SAC-only optimizer update"
    )


if __name__ == "__main__":
    main()
