from __future__ import annotations

import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from tianshou.algorithm.optim import (
    AdamOptimizerFactory,
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

from training.masked_auto_alpha_v1 import (
    MASKED_AUTO_ALPHA_VERSION,
    MaskedAutoAlphaError,
    MaskedAutoAlphaV1,
)

from training.protocol_v1 import (
    PAPER_ENTROPY_TARGET_COEFFICIENT,
    PROJECT_MAIN_DEMO_EPISODES,
    PROJECT_MAIN_DEMO_TRANSITIONS,
)


RAW_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw/train"
)

QUALITY = Path(
    "data/manifests/"
    "demo_quality_v1.csv"
)


def deterministic_direction_test():
    # ==========================================================
    # Entropy BELOW target -> alpha must increase.
    # ==========================================================

    alpha_up = (
        MaskedAutoAlphaV1(
            target_coefficient=
                0.6,

            # Regression-only value.
            initial_alpha=
                0.2,

            optim=
                AdamOptimizerFactory(
                    lr=1e-2
                ),

            device=
                "cpu",
        )
    )

    mask_up = np.asarray(
        [
            [
                1, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            [
                1, 1, 1, 1,
                0, 0, 0, 0, 0,
            ],
            [
                1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ],
        ],
        dtype=np.bool_,
    )

    targets_up = torch.tensor(
        [
            0.6 * math.log(1.0),
            0.6 * math.log(4.0),
            0.6 * math.log(9.0),
        ],
        dtype=torch.float32,
    )

    entropy_below = torch.tensor(
        [
            0.0,
            float(
                targets_up[1]
                -
                0.2
            ),
            float(
                targets_up[2]
                -
                0.2
            ),
        ],
        dtype=torch.float32,
    )

    before_up = (
        alpha_up.value
    )

    result_up = (
        alpha_up.update_with_mask(
            entropy=
                entropy_below,

            mask=
                mask_up,
        )
    )

    assert (
        result_up.alpha_after
        >
        before_up
    )

    assert (
        result_up.min_legal_actions
        ==
        1
    )

    assert (
        result_up.max_legal_actions
        ==
        9
    )


    # ==========================================================
    # Entropy ABOVE target -> alpha must decrease.
    # ==========================================================

    alpha_down = (
        MaskedAutoAlphaV1(
            target_coefficient=
                0.6,

            initial_alpha=
                0.2,

            optim=
                AdamOptimizerFactory(
                    lr=1e-2
                ),

            device=
                "cpu",
        )
    )

    mask_down = np.asarray(
        [
            [
                1, 1, 1, 1,
                0, 0, 0, 0, 0,
            ],
            [
                1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ],
        ],
        dtype=np.bool_,
    )

    targets_down = torch.tensor(
        [
            0.6 * math.log(4.0),
            0.6 * math.log(9.0),
        ],
        dtype=torch.float32,
    )

    entropy_above = (
        targets_down
        +
        0.2
    )

    before_down = (
        alpha_down.value
    )

    result_down = (
        alpha_down.update_with_mask(
            entropy=
                entropy_above,

            mask=
                mask_down,
        )
    )

    assert (
        result_down.alpha_after
        <
        before_down
    )


    # ==========================================================
    # all-False CURRENT mask must fail.
    # ==========================================================

    bad_mask = np.zeros(
        (
            1,
            9,
        ),
        dtype=np.bool_,
    )

    try:
        alpha_down.update_with_mask(
            entropy=
                torch.tensor(
                    [
                        0.0
                    ],
                    dtype=torch.float32,
                ),

            mask=
                bad_mask,
        )

    except MaskedAutoAlphaError:
        pass

    else:
        raise AssertionError(
            "Masked auto-alpha accepted "
            "an all-False current mask"
        )


    # ==========================================================
    # Plain update(entropy) must never silently ignore the mask.
    # ==========================================================

    try:
        alpha_down.update(
            torch.tensor(
                [
                    0.5
                ],
                dtype=torch.float32,
            )
        )

    except RuntimeError:
        pass

    else:
        raise AssertionError(
            "MaskedAutoAlphaV1 unexpectedly "
            "accepted update(entropy) without mask"
        )


def production_demo_integration_test():
    torch.manual_seed(
        53
    )

    np.random.seed(
        53
    )

    demo_buffer, records, provenance = (
        load_main_demo_replay(
            raw_root=
                RAW_ROOT,

            quality_manifest=
                QUALITY,

            random_seed=
                53,
        )
    )

    assert len(
        demo_buffer
    ) == PROJECT_MAIN_DEMO_TRANSITIONS

    assert len(
        records
    ) == PROJECT_MAIN_DEMO_EPISODES

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

    auto_alpha = (
        MaskedAutoAlphaV1(
            target_coefficient=
                PAPER_ENTROPY_TARGET_COEFFICIENT,

            # Regression-only value.
            #
            # The formal initial-alpha contract is frozen
            # separately by the training-protocol tests.
            initial_alpha=
                0.2,

            optim=
                AdamOptimizerFactory(
                    lr=1e-3
                ),

            device=
                "cpu",
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
                0.95,

            alpha=
                auto_alpha,

            n_step_return_horizon=
                1,

            # Regression-only value.
            bc_weight=
                0.5,

            bc_enabled=
                True,
        )
    )

    alpha_before = (
        auto_alpha.value
    )

    with policy_within_training_step(
        policy
    ):
        stats = (
            algorithm.update(
                demo_buffer,
                sample_size=12,
            )
        )

    alpha_after = (
        auto_alpha.value
    )

    update = (
        auto_alpha.last_update
    )

    assert update is not None

    assert (
        update.batch_size
        ==
        12
    )

    assert (
        update.min_legal_actions
        >=
        1
    )

    assert (
        update.max_legal_actions
        <=
        MAX_LOOPS
    )

    assert np.isfinite(
        stats.alpha_loss
    )

    assert np.isfinite(
        alpha_after
    )

    assert (
        alpha_after
        >
        0.0
    )

    assert (
        alpha_after
        !=
        alpha_before
    )

    assert math.isclose(
        stats.alpha,
        alpha_after,
        rel_tol=1e-7,
        abs_tol=1e-7,
    )

    print("=" * 80)
    print("MASKED AUTO-ALPHA V1")
    print("=" * 80)

    print(
        "version             :",
        MASKED_AUTO_ALPHA_VERSION,
    )

    print(
        "batch size          :",
        update.batch_size,
    )

    print(
        "legal count range   :",
        (
            update.min_legal_actions,
            update.max_legal_actions,
        ),
    )

    print(
        "mean entropy        :",
        update.mean_entropy,
    )

    print(
        "mean target entropy :",
        update.mean_target_entropy,
    )

    print(
        "alpha before        :",
        alpha_before,
    )

    print(
        "alpha after         :",
        alpha_after,
    )

    print(
        "alpha loss          :",
        stats.alpha_loss,
    )

    print(
        "actor loss          :",
        stats.actor_loss,
    )

    print(
        "BC loss             :",
        stats.bc_loss,
    )

    print(
        "BC selected         :",
        stats.bc_selected_count,
    )


def main():
    deterministic_direction_test()

    production_demo_integration_test()

    print()
    print(
        "PASS: Masked Auto-Alpha V1 uses "
        "0.6*log(n_legal) per current state, "
        "updates in the correct direction, and "
        "integrates with production D_demo SAC+BC"
    )


if __name__ == "__main__":
    main()
