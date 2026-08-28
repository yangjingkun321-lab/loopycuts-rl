import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from gymnasium import spaces

from tianshou.algorithm.modelfree.discrete_sac import (
    DiscreteSAC,
)

from tianshou.algorithm.optim import (
    AdamOptimizerFactory,
)

from tianshou.data import (
    Batch,
    Collector,
    ReplayBuffer,
)

from tianshou.utils.net.common import (
    Net,
)

from tianshou.utils.net.discrete import (
    DiscreteActor,
    DiscreteCritic,
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


from policies.masked_discrete_sac import (
    MaskedDiscreteSACPolicy,
)


ACTION_DIM = 5
STATE_DIM = 3
EPISODE_LENGTH = 3


# ================================================================
# Tiny masked environment
# ================================================================


class TinyMaskedEnv(gym.Env):
    """
    Minimal deterministic environment used only to verify the
    Tianshou + dynamic action-mask data path.

    State 0:
        legal actions = {2, 4}

    State 1:
        legal actions = {1, 3}

    State 2:
        legal actions = {0, 4}

    Terminal observation:
        legal actions = {}

        mask = all False

    Every episode contains exactly three environment steps.
    """

    metadata = {}

    def __init__(self):
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
                            shape=(
                                STATE_DIM,
                            ),
                            dtype=np.float32,
                        ),

                    "mask":
                        spaces.MultiBinary(
                            ACTION_DIM
                        ),
                }
            )
        )

        self.state_index = 0

    # ------------------------------------------------------------

    @staticmethod
    def _mask_for_state(
        state_index,
    ):
        if state_index == 0:
            return np.asarray(
                [
                    0,
                    0,
                    1,
                    0,
                    1,
                ],
                dtype=np.int8,
            )

        if state_index == 1:
            return np.asarray(
                [
                    0,
                    1,
                    0,
                    1,
                    0,
                ],
                dtype=np.int8,
            )

        if state_index == 2:
            return np.asarray(
                [
                    1,
                    0,
                    0,
                    0,
                    1,
                ],
                dtype=np.int8,
            )

        #
        # Terminal observation.
        #
        return np.zeros(
            ACTION_DIM,
            dtype=np.int8,
        )

    # ------------------------------------------------------------

    def _observation(self):
        observation = {
            "obs":
                np.asarray(
                    [
                        float(
                            self.state_index
                        ),
                        1.0,
                        -1.0,
                    ],
                    dtype=np.float32,
                ),

            "mask":
                self._mask_for_state(
                    self.state_index
                ),
        }

        assert (
            self.observation_space.contains(
                observation
            )
        )

        return observation

    # ------------------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        self.state_index = 0

        return (
            self._observation(),
            {},
        )

    # ------------------------------------------------------------

    def step(
        self,
        action,
    ):
        action = int(action)

        current_mask = (
            self._mask_for_state(
                self.state_index
            )
        )

        if not (
            0
            <= action
            < ACTION_DIM
        ):
            raise RuntimeError(
                f"Action {action} is outside "
                f"Discrete({ACTION_DIM})"
            )

        #
        # This is deliberately strict.
        #
        # If Tianshou / the masked policy ever sends an illegal
        # action, the test must fail immediately.
        #
        if (
            current_mask[
                action
            ]
            == 0
        ):
            raise RuntimeError(
                "Masked SAC selected illegal "
                f"action {action} at state "
                f"{self.state_index}; "
                f"mask={current_mask.tolist()}"
            )

        #
        # Simple deterministic reward.
        #
        preferred_actions = [
            2,
            3,
            4,
        ]

        if (
            action
            ==
            preferred_actions[
                self.state_index
            ]
        ):
            reward = 1.0
        else:
            reward = -0.1

        self.state_index += 1

        terminated = bool(
            self.state_index
            >= EPISODE_LENGTH
        )

        truncated = False

        info = {
            "state_index":
                self.state_index,

            "executed_action":
                action,
        }

        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            info,
        )


# ================================================================
# Critic adapter
# ================================================================


class ObsOnlyDiscreteCritic(
    torch.nn.Module
):
    """
    Adapter around Tianshou DiscreteCritic.

    Tianshou DiscreteSAC sends the full Dict-observation Batch
    directly to the critic:

        Batch(
            obs=<state features>,
            mask=<action mask>,
        )

    The critic should learn Q(s, a) from the actual state features,
    not from the action-mask container itself.

    Therefore this adapter extracts observation.obs before calling
    the normal Tianshou DiscreteCritic.
    """

    def __init__(
        self,
        state_shape,
        action_dim,
    ):
        super().__init__()

        preprocess_net = Net(
            state_shape=
                state_shape,

            hidden_sizes=[
                32,
                32,
            ],
        )

        self.critic = (
            DiscreteCritic(
                preprocess_net=
                    preprocess_net,

                last_size=
                    action_dim,
            )
        )

    # ------------------------------------------------------------

    def forward(
        self,
        observation,
        state=None,
        info=None,
    ):
        if hasattr(
            observation,
            "obs",
        ):
            observation = (
                observation.obs
            )

        return self.critic(
            observation,
            state=state,
            info=info,
        )


# ================================================================
# Helpers
# ================================================================


def clone_parameters(
    module,
):
    return [
        parameter.detach().clone()
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

    assert len(before) == len(after)

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


# ================================================================
# Main test
# ================================================================


def main():

    torch.manual_seed(0)
    np.random.seed(0)

    env = TinyMaskedEnv()

    # ------------------------------------------------------------
    # Actor
    # ------------------------------------------------------------

    actor_preprocess = Net(
        state_shape=(
            STATE_DIM,
        ),

        hidden_sizes=[
            32,
            32,
        ],
    )

    actor = DiscreteActor(
        preprocess_net=
            actor_preprocess,

        action_shape=(
            ACTION_DIM,
        ),

        hidden_sizes=(),

        #
        # Required by Tianshou Discrete SAC:
        #
        # policy constructs Categorical(logits=...).
        #
        softmax_output=False,
    )

    actor_optim = (
        AdamOptimizerFactory(
            lr=1e-3
        )
    )

    # ------------------------------------------------------------
    # Critics
    # ------------------------------------------------------------

    critic1 = (
        ObsOnlyDiscreteCritic(
            state_shape=(
                STATE_DIM,
            ),

            action_dim=
                ACTION_DIM,
        )
    )

    critic2 = (
        ObsOnlyDiscreteCritic(
            state_shape=(
                STATE_DIM,
            ),

            action_dim=
                ACTION_DIM,
        )
    )

    critic1_optim = (
        AdamOptimizerFactory(
            lr=1e-3
        )
    )

    critic2_optim = (
        AdamOptimizerFactory(
            lr=1e-3
        )
    )

    # ------------------------------------------------------------
    # Masked policy
    # ------------------------------------------------------------

    policy = (
        MaskedDiscreteSACPolicy(
            actor=actor,

            action_space=
                env.action_space,

            observation_space=
                env.observation_space,

            #
            # For this infrastructure test always use categorical
            # sampling.
            #
            deterministic_eval=
                False,
        )
    )

    # ------------------------------------------------------------
    # Tianshou Discrete SAC
    # ------------------------------------------------------------

    algorithm = DiscreteSAC(
        policy=policy,

        policy_optim=
            actor_optim,

        critic=critic1,

        critic_optim=
            critic1_optim,

        critic2=critic2,

        critic2_optim=
            critic2_optim,

        tau=0.005,
        gamma=0.99,
        alpha=0.2,

        #
        # First integration test uses 1-step targets.
        #
        n_step_return_horizon=1,
    )

    # ------------------------------------------------------------
    # Collector + ReplayBuffer
    # ------------------------------------------------------------

    buffer = ReplayBuffer(
        size=128
    )

    collector = Collector(
        algorithm,
        env,
        buffer,
    )

    collector.reset()

    collect_stats = (
        collector.collect(
            n_episode=6
        )
    )

    print()
    print(
        "===================================="
    )

    print(
        "COLLECTION"
    )

    print(
        "===================================="
    )

    print(
        "episodes:",
        collect_stats[
            "n_collected_episodes"
        ]
        if isinstance(
            collect_stats,
            dict,
        )
        else
        collect_stats.n_collected_episodes,
    )

    print(
        "steps:",
        collect_stats[
            "n_collected_steps"
        ]
        if isinstance(
            collect_stats,
            dict,
        )
        else
        collect_stats.n_collected_steps,
    )

    print(
        "buffer size:",
        len(buffer),
    )

    assert len(buffer) == (
        6
        *
        EPISODE_LENGTH
    )

    # ------------------------------------------------------------
    # Inspect every transition stored by Tianshou.
    # ------------------------------------------------------------

    all_indices = (
        buffer.sample_indices(
            0
        )
    )

    data = buffer[
        all_indices
    ]

    actions = np.asarray(
        data.act,
        dtype=np.int64,
    ).reshape(-1)

    current_masks = np.asarray(
        data.obs.mask
    ).astype(bool)

    next_masks = np.asarray(
        data.obs_next.mask
    ).astype(bool)

    terminated = np.asarray(
        data.terminated
    ).astype(bool)

    print()
    print(
        "stored actions:",
        actions.tolist(),
    )

    print(
        "terminated count:",
        int(
            terminated.sum()
        ),
    )

    print(
        "terminal indices:",
        all_indices[
            terminated
        ].tolist(),
    )

    # ------------------------------------------------------------
    # Test 1:
    # Every action actually executed by Collector must have been
    # legal under the corresponding current-state mask.
    # ------------------------------------------------------------

    chosen_action_legal = (
        current_masks[
            np.arange(
                len(actions)
            ),
            actions,
        ]
    )

    assert bool(
        chosen_action_legal.all()
    )

    print(
        "all collected actions legal:",
        True,
    )

    # ------------------------------------------------------------
    # Test 2:
    # Exactly one terminal transition per episode.
    # ------------------------------------------------------------

    assert int(
        terminated.sum()
    ) == 6

    # ------------------------------------------------------------
    # Test 3:
    # Terminal obs_next must preserve the genuine all-False mask.
    #
    # We do NOT modify the environment observation to insert the
    # dummy fallback.
    # ------------------------------------------------------------

    terminal_next_masks = (
        next_masks[
            terminated
        ]
    )

    assert not bool(
        terminal_next_masks.any()
    )

    print(
        "terminal obs_next masks all False:",
        True,
    )

    # ------------------------------------------------------------
    # Test 4:
    # Non-terminal obs_next must always have at least one legal
    # action.
    # ------------------------------------------------------------

    nonterminal_next_masks = (
        next_masks[
            ~terminated
        ]
    )

    assert bool(
        nonterminal_next_masks
        .any(
            axis=1
        )
        .all()
    )

    # ------------------------------------------------------------
    # Test 5:
    # Feed the terminal obs_next observations directly through
    # MaskedDiscreteSACPolicy.
    #
    # The policy must activate its INTERNAL fallback only here.
    # ------------------------------------------------------------

    terminal_obs_next = (
        data.obs_next[
            terminated
        ]
    )

    terminal_policy_batch = (
        Batch(
            obs=
                terminal_obs_next,

            info=[
                {}
                for _ in range(
                    int(
                        terminated.sum()
                    )
                )
            ],
        )
    )

    terminal_policy_output = (
        policy(
            terminal_policy_batch
        )
    )

    assert bool(
        terminal_policy_output
        .mask_fallback
        .all()
    )

    terminal_probs = (
        terminal_policy_output
        .dist
        .probs
        .detach()
    )

    print()

    print(
        "terminal fallback probabilities:"
    )

    print(
        terminal_probs
    )

    expected_terminal_probs = (
        torch.zeros_like(
            terminal_probs
        )
    )

    expected_terminal_probs[
        :,
        0,
    ] = 1.0

    assert torch.equal(
        terminal_probs,
        expected_terminal_probs,
    )

    # ------------------------------------------------------------
    # Test 6:
    # Exercise Tianshou's exact terminal n-step preprocessing path.
    #
    # For a terminated transition and n_step=1:
    #
    #     return = immediate reward
    #
    # because the bootstrap target is multiplied by zero.
    #
    # This call internally performs:
    #
    #     terminal obs_next
    #       -> policy
    #       -> fallback distribution
    #       -> critic target
    #       -> terminated value mask
    #
    # ------------------------------------------------------------

    first_terminal_position = int(
        np.flatnonzero(
            terminated
        )[0]
    )

    first_terminal_index = int(
        all_indices[
            first_terminal_position
        ]
    )

    target_indices = np.asarray(
        [
            first_terminal_index,
        ],
        dtype=np.int64,
    )

    target_batch = buffer[
        target_indices
    ]

    immediate_reward = (
        np.asarray(
            target_batch.rew,
            dtype=np.float32,
        )
        .reshape(-1)
        .copy()
    )

    processed_batch = (
        algorithm._preprocess_batch(
            target_batch,
            buffer,
            target_indices,
        )
    )

    returns = (
        processed_batch
        .returns
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    print()
    print(
        "terminal immediate reward:",
        immediate_reward,
    )

    print(
        "terminal computed return:",
        returns,
    )

    assert np.isfinite(
        returns
    ).all()

    assert np.allclose(
        returns,
        immediate_reward,
        atol=1e-6,
        rtol=1e-6,
    )

    # ------------------------------------------------------------
    # Test 7:
    # Execute one real Discrete SAC optimizer update.
    #
    # This simultaneously tests:
    #
    #   ReplayBuffer sampling
    #   current-state actor masking
    #   terminal/nonterminal target policy
    #   both critics
    #   actor entropy
    #   critic loss
    #   backward()
    #   optimizer.step()
    #
    # ------------------------------------------------------------

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

    with policy_within_training_step(
        policy
    ):
        training_stats = (
            algorithm.update(
                buffer,
                sample_size=12,
            )
        )

    print()
    print(
        "===================================="
    )

    print(
        "ONE SAC UPDATE"
    )

    print(
        "===================================="
    )

    print(
        "actor_loss:",
        training_stats.actor_loss,
    )

    print(
        "critic1_loss:",
        training_stats.critic1_loss,
    )

    print(
        "critic2_loss:",
        training_stats.critic2_loss,
    )

    print(
        "alpha:",
        training_stats.alpha,
    )

    print(
        "alpha_loss:",
        training_stats.alpha_loss,
    )

    losses = np.asarray(
        [
            training_stats.actor_loss,
            training_stats.critic1_loss,
            training_stats.critic2_loss,
        ],
        dtype=np.float64,
    )

    assert np.isfinite(
        losses
    ).all()

    assert parameters_changed(
        actor_before,
        actor,
    )

    assert parameters_changed(
        critic1_before,
        critic1,
    )

    #
    # Every network parameter must remain finite after backward
    # and optimizer updates.
    #
    for name, module in (
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
    ):
        for parameter in (
            module.parameters()
        ):
            assert bool(
                torch.isfinite(
                    parameter
                ).all()
            ), (
                f"Non-finite parameter "
                f"detected in {name}"
            )

    print()
    print(
        "actor parameters changed:",
        True,
    )

    print(
        "critic1 parameters changed:",
        True,
    )

    print()
    print(
        "PASS: Collector -> ReplayBuffer -> "
        "masked terminal target -> "
        "DiscreteSAC optimizer update "
        "works end-to-end."
    )

    collector.close()


if __name__ == "__main__":
    main()