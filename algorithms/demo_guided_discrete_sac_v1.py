from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tianshou.algorithm.modelfree.discrete_sac import (
    DiscreteSAC,
    DiscreteSACTrainingStats,
)

from tianshou.data import (
    to_torch,
)

from tianshou.data.types import (
    RolloutBatchProtocol,
)

from imitation.masked_bc_v1 import (
    q_filtered_masked_behavior_cloning_loss,
)


ALGORITHM_VERSION = (
    "loopycuts_demo_guided_discrete_sac_v1"
)


@dataclass(
    kw_only=True
)
class DemoGuidedDiscreteSACTrainingStats(
    DiscreteSACTrainingStats
):
    sac_actor_loss: float
    bc_loss: float
    total_actor_loss: float

    bc_unfiltered_loss: float
    bc_selected_count: int
    bc_filter_fraction: float
    bc_mean_expert_probability: float
    bc_top1_accuracy: float
    bc_mean_q_margin: float

    bc_weight: float


class LoopyCutsDemoGuidedDiscreteSACV1(
    DiscreteSAC
):
    """
    LoopyCuts Stage-I / Stage-II Discrete SAC.

    Stage I:
        bc_enabled = True

        Actor objective:

            J_actor =
                J_SAC
                +
                lambda_BC * L_BC_QF

    Stage II:
        bc_enabled = False

        Actor objective:

            J_actor = J_SAC

    The SAME:
        Actor
        Critic-1
        Critic-2
        target critics
        alpha
        optimizer states

    remain alive when switching stages.
    """

    def __init__(
        self,
        *,
        bc_weight: float,
        bc_enabled: bool = True,
        **kwargs,
    ):
        bc_weight = float(
            bc_weight
        )

        if (
            not np.isfinite(
                bc_weight
            )
            or
            bc_weight
            <
            0.0
        ):
            raise ValueError(
                "bc_weight must be a "
                "finite non-negative scalar"
            )

        super().__init__(
            **kwargs
        )

        self.bc_weight = (
            bc_weight
        )

        self.bc_enabled = bool(
            bc_enabled
        )

    def set_bc_enabled(
        self,
        enabled: bool,
    ):
        """
        Switch between:

            Stage I:
                SAC + Q-filtered BC

            Stage II:
                SAC only

        No network or optimizer is re-created.
        """

        self.bc_enabled = bool(
            enabled
        )

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> DemoGuidedDiscreteSACTrainingStats:

        # ======================================================
        # Critic 1 / Critic 2.
        #
        # Match the pinned Tianshou DiscreteSAC semantics.
        # ======================================================

        weight = batch.pop(
            "weight",
            1.0,
        )

        target_q = (
            batch
            .returns
            .flatten()
        )

        action = to_torch(
            batch.act[
                :,
                np.newaxis,
            ],
            device=
                target_q.device,
            dtype=
                torch.long,
        )

        current_q1 = (
            self.critic(
                batch.obs
            )
            .gather(
                1,
                action,
            )
            .flatten()
        )

        td1 = (
            current_q1
            -
            target_q
        )

        critic1_loss = (
            (
                td1.pow(
                    2
                )
                *
                weight
            )
            .mean()
        )

        self.critic_optim.step(
            critic1_loss
        )


        current_q2 = (
            self.critic2(
                batch.obs
            )
            .gather(
                1,
                action,
            )
            .flatten()
        )

        td2 = (
            current_q2
            -
            target_q
        )

        critic2_loss = (
            (
                td2.pow(
                    2
                )
                *
                weight
            )
            .mean()
        )

        self.critic2_optim.step(
            critic2_loss
        )


        batch.weight = (
            td1
            +
            td2
        ) / 2.0


        # ======================================================
        # Standard masked Discrete-SAC Actor objective.
        # ======================================================

        dist = (
            self.policy(
                batch
            )
            .dist
        )

        entropy = (
            dist.entropy()
        )

        with torch.no_grad():
            q1_all = (
                self.critic(
                    batch.obs
                )
            )

            q2_all = (
                self.critic2(
                    batch.obs
                )
            )

            q_min = torch.minimum(
                q1_all,
                q2_all,
            )

        sac_actor_loss = (
            -(
                self.alpha.value
                *
                entropy
                +
                (
                    dist.probs
                    *
                    q_min
                )
                .sum(
                    dim=-1
                )
            )
            .mean()
        )


        # ======================================================
        # Stage-I Q-filtered BC auxiliary objective.
        # ======================================================

        if self.bc_enabled:
            bc_output = (
                q_filtered_masked_behavior_cloning_loss(
                    actor=
                        self.policy.actor,

                    critic1=
                        self.critic,

                    critic2=
                        self.critic2,

                    batch=
                        batch,
                )
            )

            bc_loss = (
                bc_output.loss
            )

            bc_unfiltered_loss = (
                bc_output
                .unfiltered_loss
            )

            bc_selected_count = (
                bc_output
                .selected_count
            )

            bc_filter_fraction = (
                bc_output
                .filter_fraction
            )

            bc_mean_expert_probability = (
                bc_output
                .mean_expert_probability
            )

            bc_top1_accuracy = (
                bc_output
                .top1_accuracy
            )

            bc_mean_q_margin = (
                bc_output
                .mean_q_margin
            )

        else:
            bc_loss = (
                sac_actor_loss
                .new_zeros(
                    ()
                )
            )

            bc_unfiltered_loss = 0.0
            bc_selected_count = 0
            bc_filter_fraction = 0.0
            bc_mean_expert_probability = 0.0
            bc_top1_accuracy = 0.0
            bc_mean_q_margin = 0.0


        total_actor_loss = (
            sac_actor_loss
            +
            self.bc_weight
            *
            bc_loss
        )

        self.policy_optim.step(
            total_actor_loss
        )


        # ======================================================
        # Entropy temperature and target critics.
        # ======================================================

        alpha_loss = (
            self.alpha.update(
                entropy.detach()
            )
        )

        self._update_lagged_network_weights()


        return (
            DemoGuidedDiscreteSACTrainingStats(
                actor_loss=
                    float(
                        total_actor_loss
                        .detach()
                        .item()
                    ),

                critic1_loss=
                    float(
                        critic1_loss
                        .detach()
                        .item()
                    ),

                critic2_loss=
                    float(
                        critic2_loss
                        .detach()
                        .item()
                    ),

                alpha=
                    self.alpha.value,

                alpha_loss=
                    alpha_loss,

                sac_actor_loss=
                    float(
                        sac_actor_loss
                        .detach()
                        .item()
                    ),

                bc_loss=
                    float(
                        bc_loss
                        .detach()
                        .item()
                    ),

                total_actor_loss=
                    float(
                        total_actor_loss
                        .detach()
                        .item()
                    ),

                bc_unfiltered_loss=
                    float(
                        bc_unfiltered_loss
                    ),

                bc_selected_count=
                    int(
                        bc_selected_count
                    ),

                bc_filter_fraction=
                    float(
                        bc_filter_fraction
                    ),

                bc_mean_expert_probability=
                    float(
                        bc_mean_expert_probability
                    ),

                bc_top1_accuracy=
                    float(
                        bc_top1_accuracy
                    ),

                bc_mean_q_margin=
                    float(
                        bc_mean_q_margin
                    ),

                bc_weight=
                    self.bc_weight,
            )
        )
