from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from gymnasium import spaces

from tianshou.data import (
    Batch,
    Collector,
    ReplayBuffer,
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


from policies.masked_discrete_sac import (
    MaskedDiscreteSACPolicy,
)


ACTION_DIM = 4


class FixedActor(
    torch.nn.Module
):
    """
    Raw logits:

        a0 = 0
        a1 = 5
        a2 = 0
        a3 = 1

    Current legal actions are {1, 3}.

    Therefore the deterministic masked policy action is a1.
    """

    def forward(
        self,
        obs,
        state=None,
        info=None,
    ):
        batch_size = int(
            np.asarray(
                obs
            ).shape[0]
        )

        logits = torch.tensor(
            [
                [
                    0.0,
                    5.0,
                    0.0,
                    1.0,
                ]
            ],
            dtype=torch.float32,
        ).repeat(
            batch_size,
            1,
        )

        return (
            logits,
            None,
        )


class OneStepMaskedEnv(
    gym.Env
):
    metadata = {}

    def __init__(
        self,
    ):
        super().__init__()

        self.action_space = (
            spaces.Discrete(
                ACTION_DIM
            )
        )

        self.observation_space = (
            spaces.Dict(
                {
                    "obs":
                        spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(1,),
                            dtype=np.float32,
                        ),

                    "mask":
                        spaces.MultiBinary(
                            ACTION_DIM
                        ),
                }
            )
        )

        self.executed_action = None

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        self.executed_action = None

        return (
            {
                "obs":
                    np.asarray(
                        [0.0],
                        dtype=np.float32,
                    ),

                "mask":
                    np.asarray(
                        [
                            0,
                            1,
                            0,
                            1,
                        ],
                        dtype=np.int8,
                    ),
            },
            {},
        )

    def step(
        self,
        action,
    ):
        action = int(
            action
        )

        if action not in {
            1,
            3,
        }:
            raise RuntimeError(
                f"Collector executed illegal "
                f"action {action}"
            )

        self.executed_action = (
            action
        )

        return (
            {
                "obs":
                    np.asarray(
                        [1.0],
                        dtype=np.float32,
                    ),

                # genuine terminal mask
                "mask":
                    np.zeros(
                        ACTION_DIM,
                        dtype=np.int8,
                    ),
            },
            1.0,
            True,
            False,
            {},
        )


def main():
    torch.manual_seed(
        0
    )

    np.random.seed(
        0
    )

    actor = FixedActor()

    policy = (
        MaskedDiscreteSACPolicy(
            actor=actor,

            action_space=
                spaces.Discrete(
                    ACTION_DIM
                ),

            deterministic_eval=True,

            # Force every Collector action through
            # the epsilon-random branch.
            exploration_epsilon=1.0,

            # With this deterministic RNG seed,
            # uniform choice from {1, 3}
            # on the first call is 3.
            exploration_seed=0,
        )
    )

    # ==========================================================
    # Verify the learned/masked policy itself chooses action 1.
    # ==========================================================

    policy_batch = Batch(
        obs=Batch(
            obs=np.asarray(
                [
                    [0.0]
                ],
                dtype=np.float32,
            ),

            mask=np.asarray(
                [
                    [
                        0,
                        1,
                        0,
                        1,
                    ]
                ],
                dtype=np.bool_,
            ),
        ),

        info=Batch(),
    )

    with torch.no_grad():
        policy_output = (
            policy(
                policy_batch
            )
        )

    base_action = int(
        policy_output.act[
            0
        ].item()
    )

    assert base_action == 1

    # ==========================================================
    # Collector must call add_exploration_noise().
    # ==========================================================

    env = OneStepMaskedEnv()

    buffer = ReplayBuffer(
        size=8
    )

    collector = Collector(
        policy,
        env,
        buffer,

        exploration_noise=True,
    )

    collector.reset()

    stats = collector.collect(
        n_episode=1
    )

    assert len(
        buffer
    ) == 1

    indices = (
        buffer.sample_indices(
            0
        )
    )

    data = buffer[
        indices
    ]

    stored_action = int(
        np.asarray(
            data.act
        ).reshape(
            -1
        )[
            0
        ]
    )

    # epsilon=1 + seed=0:
    #
    # policy mode = 1
    # random legal action = 3
    #
    # Therefore the stored action proves Collector used
    # the exploration hook.
    assert stored_action == 3

    assert (
        stored_action
        !=
        base_action
    )

    current_mask = np.asarray(
        data.obs.mask,
        dtype=np.bool_,
    )[0]

    assert bool(
        current_mask[
            stored_action
        ]
    )

    terminated = bool(
        np.asarray(
            data.terminated
        ).reshape(
            -1
        )[
            0
        ]
    )

    assert terminated

    terminal_mask = np.asarray(
        data.obs_next.mask,
        dtype=np.bool_,
    )[0]

    assert not bool(
        terminal_mask.any()
    )

    print("=" * 80)
    print("MASKED EPSILON-GREEDY COLLECTOR")
    print("=" * 80)

    print(
        "policy action       :",
        base_action,
    )

    print(
        "executed action     :",
        stored_action,
    )

    print(
        "executed action legal:",
        True,
    )

    print(
        "replay size         :",
        len(
            buffer
        ),
    )

    print()
    print(
        "PASS: Tianshou Collector with "
        "exploration_noise=True applies "
        "Masked epsilon-greedy before "
        "storing the executed legal action"
    )


if __name__ == "__main__":
    main()
