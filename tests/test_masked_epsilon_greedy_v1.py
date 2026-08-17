from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
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
    MASKED_EPSILON_GREEDY_VERSION,
    MaskedDiscreteSACPolicy,
)

from training.protocol_v1 import (
    PAPER_TRAIN_EPSILON_GREEDY,
    PROJECT_EPSILON_RANDOM_SUPPORT,
    PROJECT_MASKED_EPSILON_GREEDY_VERSION,
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
        67
    )

    np.random.seed(
        67
    )

    demo_buffer, records, provenance = (
        load_main_demo_replay(
            raw_root=
                RAW_ROOT,

            quality_manifest=
                QUALITY,

            random_seed=
                67,
        )
    )

    indices = (
        demo_buffer.sample_indices(
            0
        )
    )

    batch = (
        demo_buffer[
            indices
        ]
    )

    assert len(
        batch
    ) == 29

    actor, _, _ = (
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

            exploration_epsilon=
                0.0,

            exploration_seed=
                71,
        )
    )

    # ==========================================================
    # Obtain normal masked policy actions.
    # ==========================================================

    with torch.no_grad():
        result = (
            policy(
                batch
            )
        )

    base_actions = (
        result.act
        .detach()
        .cpu()
        .numpy()
    )

    masks = np.asarray(
        batch.obs.mask,
        dtype=np.bool_,
    )

    rows = np.arange(
        len(
            base_actions
        )
    )

    assert bool(
        masks[
            rows,
            base_actions,
        ].all()
    )

    # ==========================================================
    # epsilon = 0:
    # action must remain exactly unchanged.
    # ==========================================================

    unchanged = (
        policy.add_exploration_noise(
            base_actions.copy(),
            batch,
        )
    )

    assert np.array_equal(
        unchanged,
        base_actions,
    )

    # ==========================================================
    # epsilon = 1:
    # every row enters random branch, but random support must be
    # exactly the CURRENT legal action set.
    # ==========================================================

    policy.set_exploration_epsilon(
        1.0
    )

    exploratory_actions = (
        policy.add_exploration_noise(
            base_actions.copy(),
            batch,
        )
    )

    assert exploratory_actions.shape == (
        base_actions.shape
    )

    assert bool(
        masks[
            rows,
            exploratory_actions,
        ].all()
    )

    # With 29 rows and deterministic RNG this regression must
    # actually exercise at least one replacement.
    assert bool(
        (
            exploratory_actions
            !=
            base_actions
        ).any()
    )

    # ==========================================================
    # Formal paper epsilon is accepted.
    # ==========================================================

    policy.set_exploration_epsilon(
        PAPER_TRAIN_EPSILON_GREEDY
    )

    assert np.isclose(
        policy.exploration_epsilon,
        0.05,
    )

    # ==========================================================
    # Current all-False mask must NEVER receive terminal fallback
    # here. Collector must not request another action after terminal.
    # ==========================================================

    bad_batch = Batch(
        obs=Batch(
            obs=np.zeros(
                (
                    1,
                    1,
                ),
                dtype=np.float32,
            ),

            mask=np.zeros(
                (
                    1,
                    MAX_LOOPS,
                ),
                dtype=np.bool_,
            ),
        ),

        info=Batch(),
    )

    try:
        policy.add_exploration_noise(
            np.asarray(
                [
                    0
                ],
                dtype=np.int64,
            ),
            bad_batch,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Masked epsilon-greedy accepted "
            "an all-False current mask"
        )

    # ==========================================================
    # Invalid epsilon must fail.
    # ==========================================================

    for bad_epsilon in [
        -0.01,
        1.01,
        float("nan"),
    ]:
        try:
            policy.set_exploration_epsilon(
                bad_epsilon
            )

        except ValueError:
            pass

        else:
            raise AssertionError(
                "Invalid epsilon was accepted: "
                f"{bad_epsilon}"
            )

    assert (
        PROJECT_EPSILON_RANDOM_SUPPORT
        ==
        "CURRENT_LEGAL_ACTIONS"
    )

    assert (
        PROJECT_MASKED_EPSILON_GREEDY_VERSION
        ==
        MASKED_EPSILON_GREEDY_VERSION
    )

    print("=" * 80)
    print("MASKED EPSILON-GREEDY V1")
    print("=" * 80)

    print(
        "version             :",
        MASKED_EPSILON_GREEDY_VERSION,
    )

    print(
        "transitions         :",
        len(
            base_actions
        ),
    )

    print(
        "formal epsilon      :",
        PAPER_TRAIN_EPSILON_GREEDY,
    )

    print(
        "epsilon=1 legal     :",
        True,
    )

    print(
        "replacement count   :",
        int(
            (
                exploratory_actions
                !=
                base_actions
            )
            .sum()
        ),
    )

    print()
    print(
        "PASS: Masked epsilon-greedy V1 "
        "preserves epsilon=0 actions, "
        "samples epsilon-random actions only "
        "from the current legal set, and "
        "rejects all-False current masks"
    )


if __name__ == "__main__":
    main()
