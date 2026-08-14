import numpy as np
import torch
import gymnasium as gym

from tianshou.algorithm.modelfree.discrete_sac import (
    DiscreteSAC,
    DiscreteSACPolicy,
)

from tianshou.algorithm.optim import (
    AdamOptimizerFactory,
)

from tianshou.data import (
    Batch,
)

from tianshou.utils.net.common import (
    Net,
)

from tianshou.utils.net.discrete import (
    DiscreteActor,
    DiscreteCritic,
)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    env = gym.make(
        "CartPole-v1"
    )

    assert isinstance(
        env.action_space,
        gym.spaces.Discrete,
    )

    observation, info = (
        env.reset(seed=0)
    )

    state_shape = (
        env.observation_space.shape
    )

    action_dim = int(
        env.action_space.n
    )

    print(
        "state_shape:",
        state_shape,
    )

    print(
        "action_dim:",
        action_dim,
    )

    # --------------------------------------------------------
    # Actor
    # --------------------------------------------------------

    actor_preprocess = Net(
        state_shape=state_shape,
        hidden_sizes=[32, 32],
    )

    actor = DiscreteActor(
        preprocess_net=
            actor_preprocess,

        action_shape=(
            action_dim,
        ),

        hidden_sizes=(),

        #
        # Tianshou DiscreteSACPolicy itself creates
        # Categorical(logits=...).
        #
        softmax_output=False,
    )

    actor_optim = (
        AdamOptimizerFactory(
            lr=1e-3
        )
    )

    # --------------------------------------------------------
    # Critic 1
    # --------------------------------------------------------

    critic1_preprocess = Net(
        state_shape=state_shape,
        hidden_sizes=[32, 32],
    )

    critic1 = DiscreteCritic(
        preprocess_net=
            critic1_preprocess,

        last_size=
            action_dim,
    )

    critic1_optim = (
        AdamOptimizerFactory(
            lr=1e-3
        )
    )

    # --------------------------------------------------------
    # Critic 2
    # --------------------------------------------------------

    critic2_preprocess = Net(
        state_shape=state_shape,
        hidden_sizes=[32, 32],
    )

    critic2 = DiscreteCritic(
        preprocess_net=
            critic2_preprocess,

        last_size=
            action_dim,
    )

    critic2_optim = (
        AdamOptimizerFactory(
            lr=1e-3
        )
    )

    # --------------------------------------------------------
    # Policy / Algorithm
    # --------------------------------------------------------

    policy = DiscreteSACPolicy(
        actor=actor,
        action_space=
            env.action_space,
        observation_space=
            env.observation_space,
    )

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
        n_step_return_horizon=1,
    )

    # --------------------------------------------------------
    # One policy forward pass
    # --------------------------------------------------------

    obs_batch = np.asarray(
        [observation],
        dtype=np.float32,
    )

    batch = Batch(
        obs=obs_batch,
        info=[
            {},
        ],
    )

    output = policy(
        batch
    )

    print(
        "actor logits shape:",
        tuple(
            output.logits.shape
        ),
    )

    print(
        "sampled action:",
        output.act,
    )

    print(
        "distribution probabilities:",
        output.dist.probs,
    )

    assert (
        tuple(
            output.logits.shape
        )
        ==
        (1, action_dim)
    )

    assert (
        output.dist.probs.shape
        ==
        (
            1,
            action_dim,
        )
    )

    probability_sum = (
        output.dist.probs.sum(
            dim=-1
        )
    )

    assert torch.allclose(
        probability_sum,
        torch.ones_like(
            probability_sum
        ),
    )

    print(
        "algorithm class:",
        type(algorithm).__name__,
    )

    print(
        "PASS: Tianshou Discrete SAC "
        "construction and policy "
        "forward pass work on CPU."
    )

    env.close()


if __name__ == "__main__":
    main()
