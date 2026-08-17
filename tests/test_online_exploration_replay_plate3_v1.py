import csv
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from tianshou.algorithm.optim import (
    AdamOptimizerFactory,
)

from tianshou.data import (
    Collector,
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

from envs.final_reward_wrapper import (
    FinalRewardWrapper,
)

from envs.finalization_eval_wrapper import (
    FinalizationEvalWrapper,
)

from envs.loopycuts_env import (
    LoopyCutsEnv,
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


EXECUTABLE = Path(
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

MANIFEST = Path(
    "data/manifests/"
    "dataset_split_v2.csv"
)

RAW_DEMO_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw/train"
)

DEMO_QUALITY = Path(
    "data/manifests/"
    "demo_quality_v1.csv"
)


def load_train_model(
    model,
):
    with MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = [
            row
            for row
            in csv.DictReader(f)
            if row["model"] == model
        ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest "
            f"row for {model}; got {len(rows)}"
        )

    row = rows[0]

    if row["split"] != "train":
        raise RuntimeError(
            f"{model} is not in split=train"
        )

    return row


def main():
    torch.manual_seed(
        43
    )

    np.random.seed(
        43
    )

    # ==========================================================
    # Formal quality-filtered D_demo.
    # ==========================================================

    demo_buffer, demo_records, demo_provenance = (
        load_main_demo_replay(
            raw_root=
                RAW_DEMO_ROOT,

            quality_manifest=
                DEMO_QUALITY,

            random_seed=
                43,
        )
    )

    assert len(
        demo_buffer
    ) == 29

    # ==========================================================
    # Same production Actor / Q1 / Q2.
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

            # Smoke-test value ONLY.
            # Formal lambda is not frozen.
            bc_weight=
                0.5,

            bc_enabled=
                True,
        )
    )

    actor_id = id(
        actor
    )

    # ==========================================================
    # Exactly ONE Stage-I update.
    #
    # This is infrastructure smoke, NOT formal training.
    # ==========================================================

    with policy_within_training_step(
        policy
    ):
        stage1 = algorithm.update(
            demo_buffer,
            sample_size=12,
        )

    print("=" * 80)
    print("PRE-COLLECTION STAGE-I SMOKE")
    print("=" * 80)

    print(
        "actor loss  :",
        stage1.actor_loss,
    )

    print(
        "BC loss     :",
        stage1.bc_loss,
    )

    print(
        "BC selected :",
        stage1.bc_selected_count,
    )

    assert id(
        actor
    ) == actor_id

    # ==========================================================
    # Enter Stage-II semantics without reinitialization.
    # ==========================================================

    algorithm.set_bc_enabled(
        False
    )

    assert not algorithm.bc_enabled
    assert id(actor) == actor_id

    # ==========================================================
    # Real Plate3 environment.
    # ==========================================================

    row = load_train_model(
        "Plate3"
    )

    mesh_file = Path(
        row[
            "mesh_file"
        ]
    )

    loop_file = Path(
        row[
            "loop_file"
        ]
    )

    print()
    print("=" * 80)
    print("REAL ONLINE EXPLORATION")
    print("=" * 80)

    print(
        "model      :",
        row["model"],
    )

    print(
        "mesh       :",
        mesh_file,
    )

    print(
        "loop       :",
        loop_file,
    )

    assert EXECUTABLE.is_file()
    assert mesh_file.is_file()
    assert loop_file.is_file()

    env = (
        FinalRewardWrapper(
            FinalizationEvalWrapper(
                LoopyCutsEnv(
                    executable=
                        EXECUTABLE,

                    mesh_file=
                        mesh_file,

                    loop_file=
                        loop_file,

                    echo_logs=
                        False,
                )
            )
        )
    )

    # Independent online replay.
    expo_buffer = ReplayBuffer(
        size=128,
        random_seed=47,
    )

    collector = Collector(
        algorithm,
        env,
        expo_buffer,
    )

    try:
        collector.reset()

        collect_stats = (
            collector.collect(
                n_episode=1
            )
        )

    finally:
        env.close()

    # ==========================================================
    # Validate real D_expo episode.
    # ==========================================================

    assert len(
        expo_buffer
    ) > 0

    indices = (
        expo_buffer.sample_indices(
            0
        )
    )

    data = expo_buffer[
        indices
    ]

    actions = np.asarray(
        data.act,
        dtype=np.int64,
    ).reshape(-1)

    rewards = np.asarray(
        data.rew,
        dtype=np.float64,
    ).reshape(-1)

    terminated = np.asarray(
        data.terminated,
        dtype=np.bool_,
    ).reshape(-1)

    truncated = np.asarray(
        data.truncated,
        dtype=np.bool_,
    ).reshape(-1)

    current_masks = np.asarray(
        data.obs.mask,
        dtype=np.bool_,
    )

    next_masks = np.asarray(
        data.obs_next.mask,
        dtype=np.bool_,
    )

    rows_index = np.arange(
        len(
            actions
        )
    )

    assert bool(
        current_masks[
            rows_index,
            actions,
        ].all()
    )

    assert not bool(
        truncated.any()
    )

    assert int(
        terminated.sum()
    ) == 1

    terminal_indices = np.flatnonzero(
        terminated
    )

    assert len(
        terminal_indices
    ) == 1

    assert not bool(
        next_masks[
            terminal_indices
        ].any()
    )

    assert bool(
        np.isfinite(
            rewards
        ).all()
    )

    # Current transition states must always be actionable.
    assert bool(
        current_masks
        .any(
            axis=1
        )
        .all()
    )

    n_episodes = (
        collect_stats[
            "n_collected_episodes"
        ]
        if isinstance(
            collect_stats,
            dict,
        )
        else
        collect_stats.n_collected_episodes
    )

    n_steps = (
        collect_stats[
            "n_collected_steps"
        ]
        if isinstance(
            collect_stats,
            dict,
        )
        else
        collect_stats.n_collected_steps
    )

    print()
    print(
        "collected episodes :",
        n_episodes,
    )

    print(
        "collected steps    :",
        n_steps,
    )

    print(
        "D_expo size        :",
        len(
            expo_buffer
        ),
    )

    print(
        "actions            :",
        actions.tolist(),
    )

    print(
        "episode return     :",
        float(
            rewards.sum()
        ),
    )

    print(
        "terminal index     :",
        int(
            terminal_indices[
                0
            ]
        ),
    )

    print(
        "all actions legal  :",
        True,
    )

    print(
        "terminal next mask :",
        "all False",
    )

    assert int(
        n_episodes
    ) == 1

    assert int(
        n_steps
    ) == len(
        expo_buffer
    )

    # Same Actor object survived:
    #
    # Stage-I update
    #   ->
    # Stage-II switch
    #   ->
    # real online exploration.
    assert id(
        actor
    ) == actor_id

    # ==========================================================
    # Audit the collector-facing Reward V2 / finalization record.
    # ==========================================================

    if not hasattr(
        data.info,
        "reward_version",
    ):
        raise RuntimeError(
            "Collected D_expo is missing "
            "info.reward_version"
        )

    if not hasattr(
        data.info,
        "finalization_outcome",
    ):
        raise RuntimeError(
            "Collected D_expo is missing "
            "info.finalization_outcome"
        )

    reward_versions = np.asarray(
        data.info.reward_version
    ).reshape(
        -1
    )

    assert all(
        str(value)
        ==
        "final_v2"
        for value
        in reward_versions
    )

    outcome_records = (
        data
        .info
        .finalization_outcome
    )

    terminal_outcome = str(
        np.asarray(
            outcome_records.outcome
        ).reshape(
            -1
        )[
            terminal_indices[
                0
            ]
        ]
    )

    assert terminal_outcome in {
        "FULL_HEX",
        "NON_FULL_HEX",
        "FINALIZATION_CRASH",
    }

    nonterminal_indices = np.flatnonzero(
        ~terminated
    )

    nonterminal_outcomes = np.asarray(
        outcome_records.outcome
    ).reshape(
        -1
    )[
        nonterminal_indices
    ]

    assert all(
        str(value)
        ==
        "NONE"
        for value
        in nonterminal_outcomes
    )

    print(
        "reward version      :",
        "final_v2",
    )

    print(
        "finalization outcome:",
        terminal_outcome,
    )


    # ==========================================================
    # REAL Stage-II dual replay update.
    #
    # The D_expo below is the episode just generated by the SAME
    # Actor against the real Plate3 LoopyCuts environment.
    #
    # Exactly:
    #
    #     3 D_demo
    #     3 real D_expo
    #
    # are used in ONE SAC-only optimizer update.
    # ==========================================================

    actor_before_mix = [
        parameter
        .detach()
        .clone()
        for parameter
        in actor.parameters()
    ]

    with policy_within_training_step(
        policy
    ):
        mixed_stats, mix = (
            algorithm.update_equal_replay(
                demo_buffer=
                    demo_buffer,

                expo_buffer=
                    expo_buffer,

                samples_per_buffer=
                    3,
            )
        )


    print()
    print("=" * 80)
    print("REAL STAGE-II 1:1 UPDATE")
    print("=" * 80)

    print(
        "D_demo samples :",
        mix[
            "demo_samples"
        ],
    )

    print(
        "D_expo samples :",
        mix[
            "expo_samples"
        ],
    )

    print(
        "total samples  :",
        mix[
            "total_samples"
        ],
    )

    print(
        "actor loss     :",
        mixed_stats.actor_loss,
    )

    print(
        "critic1 loss   :",
        mixed_stats.critic1_loss,
    )

    print(
        "critic2 loss   :",
        mixed_stats.critic2_loss,
    )

    print(
        "BC loss        :",
        mixed_stats.bc_loss,
    )

    print(
        "BC selected    :",
        mixed_stats.bc_selected_count,
    )


    assert (
        mix[
            "demo_samples"
        ]
        ==
        3
    )

    assert (
        mix[
            "expo_samples"
        ]
        ==
        3
    )

    assert (
        mix[
            "total_samples"
        ]
        ==
        6
    )

    assert (
        mixed_stats.bc_loss
        ==
        0.0
    )

    assert (
        mixed_stats.bc_selected_count
        ==
        0
    )

    assert (
        mixed_stats.bc_filter_fraction
        ==
        0.0
    )

    losses = np.asarray(
        [
            mixed_stats.actor_loss,
            mixed_stats.critic1_loss,
            mixed_stats.critic2_loss,
        ],
        dtype=np.float64,
    )

    assert bool(
        np.isfinite(
            losses
        ).all()
    )


    actor_after_mix = list(
        actor.parameters()
    )

    assert len(
        actor_before_mix
    ) == len(
        actor_after_mix
    )

    actor_changed = any(
        not torch.equal(
            before,
            after.detach(),
        )
        for before, after
        in zip(
            actor_before_mix,
            actor_after_mix,
        )
    )

    assert actor_changed


    # SAME Actor object throughout:
    #
    # Stage-I update
    #   ->
    # online Plate3 exploration
    #   ->
    # real D_demo/D_expo mixed Stage-II update.
    assert id(
        actor
    ) == actor_id


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
                f"Non-finite parameter "
                f"in {name}"
            )


    print()
    print(
        "PASS: same Stage-I Actor generated "
        "real Plate3 D_expo and immediately "
        "continued with a real 1:1 "
        "D_demo/D_expo Stage-II SAC update"
    )


if __name__ == "__main__":
    main()
