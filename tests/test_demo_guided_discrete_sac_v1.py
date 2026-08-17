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
    ALGORITHM_VERSION,
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


def clone_parameters(
    module,
):
    return [
        parameter
        .detach()
        .clone()
        for parameter
        in module.parameters()
    ]


def parameters_changed(
    before,
    module,
):
    after = list(
        module.parameters()
    )

    assert len(
        before
    ) == len(
        after
    )

    return any(
        not torch.equal(
            old,
            new.detach(),
        )
        for old, new
        in zip(
            before,
            after,
        )
    )


def assert_module_finite(
    module,
):
    for parameter in module.parameters():
        assert bool(
            torch.isfinite(
                parameter
            ).all()
        )


def main():
    torch.manual_seed(
        23
    )

    np.random.seed(
        23
    )


    demo_buffer, records, provenance = (
        load_main_demo_replay(
            raw_root=
                RAW_ROOT,

            quality_manifest=
                QUALITY,

            random_seed=
                23,
        )
    )


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

            # Regression value only.
            # Formal lambda is NOT frozen here.
            bc_weight=
                0.5,

            bc_enabled=
                True,
        )
    )


    actor_id = id(
        actor
    )

    critic1_id = id(
        critic1
    )

    critic2_id = id(
        critic2
    )


    # ==========================================================
    # Stage I:
    # SAC + Q-filtered BC.
    # ==========================================================

    actor_before = (
        clone_parameters(
            actor
        )
    )

    critic1_before = (
        clone_parameters(
            critic1
        )
    )

    critic2_before = (
        clone_parameters(
            critic2
        )
    )


    with policy_within_training_step(
        policy
    ):
        stage1 = (
            algorithm.update(
                demo_buffer,
                sample_size=12,
            )
        )


    print("=" * 80)
    print("STAGE I UPDATE")
    print("=" * 80)

    print(
        "algorithm version :",
        ALGORITHM_VERSION,
    )

    print(
        "sac actor loss    :",
        stage1.sac_actor_loss,
    )

    print(
        "BC loss           :",
        stage1.bc_loss,
    )

    print(
        "total actor loss  :",
        stage1.total_actor_loss,
    )

    print(
        "BC selected       :",
        stage1.bc_selected_count,
    )

    print(
        "BC fraction       :",
        stage1.bc_filter_fraction,
    )

    print(
        "critic1 loss      :",
        stage1.critic1_loss,
    )

    print(
        "critic2 loss      :",
        stage1.critic2_loss,
    )


    values = np.asarray(
        [
            stage1.sac_actor_loss,
            stage1.bc_loss,
            stage1.total_actor_loss,
            stage1.critic1_loss,
            stage1.critic2_loss,
        ],
        dtype=np.float64,
    )

    assert bool(
        np.isfinite(
            values
        ).all()
    )


    np.testing.assert_allclose(
        stage1.total_actor_loss,
        (
            stage1.sac_actor_loss
            +
            0.5
            *
            stage1.bc_loss
        ),
        atol=1e-6,
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        stage1.actor_loss,
        stage1.total_actor_loss,
        atol=1e-7,
        rtol=1e-7,
    )


    assert (
        0
        <=
        stage1.bc_selected_count
        <=
        12
    )

    assert (
        0.0
        <=
        stage1.bc_filter_fraction
        <=
        1.0
    )


    assert parameters_changed(
        actor_before,
        actor,
    )

    assert parameters_changed(
        critic1_before,
        critic1,
    )

    assert parameters_changed(
        critic2_before,
        critic2,
    )


    # ==========================================================
    # Switch to Stage II semantics.
    #
    # SAME algorithm / Actor / critics / optimizers.
    # Only the BC term is disabled.
    # ==========================================================

    algorithm.set_bc_enabled(
        False
    )


    assert id(
        actor
    ) == actor_id

    assert id(
        critic1
    ) == critic1_id

    assert id(
        critic2
    ) == critic2_id


    actor_before_stage2 = (
        clone_parameters(
            actor
        )
    )


    with policy_within_training_step(
        policy
    ):
        stage2 = (
            algorithm.update(
                demo_buffer,
                sample_size=12,
            )
        )


    print()
    print("=" * 80)
    print("STAGE II-SEMANTICS UPDATE")
    print("=" * 80)

    print(
        "sac actor loss    :",
        stage2.sac_actor_loss,
    )

    print(
        "BC loss           :",
        stage2.bc_loss,
    )

    print(
        "total actor loss  :",
        stage2.total_actor_loss,
    )

    print(
        "BC selected       :",
        stage2.bc_selected_count,
    )


    assert (
        stage2.bc_loss
        ==
        0.0
    )

    assert (
        stage2.bc_selected_count
        ==
        0
    )

    assert (
        stage2.bc_filter_fraction
        ==
        0.0
    )


    np.testing.assert_allclose(
        stage2.total_actor_loss,
        stage2.sac_actor_loss,
        atol=1e-7,
        rtol=1e-7,
    )

    np.testing.assert_allclose(
        stage2.actor_loss,
        stage2.sac_actor_loss,
        atol=1e-7,
        rtol=1e-7,
    )


    assert parameters_changed(
        actor_before_stage2,
        actor,
    )


    assert_module_finite(
        actor
    )

    assert_module_finite(
        critic1
    )

    assert_module_finite(
        critic2
    )


    print()
    print(
        "PASS: same Actor/Critics perform "
        "Stage-I SAC+Q-filtered-BC and then "
        "continue with Stage-II SAC-only "
        "without reinitialization"
    )


if __name__ == "__main__":
    main()
