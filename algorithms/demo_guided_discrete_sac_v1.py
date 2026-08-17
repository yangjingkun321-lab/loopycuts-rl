from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tianshou.algorithm.modelfree.discrete_sac import (
    DiscreteSAC,
    DiscreteSACTrainingStats,
)

from tianshou.data import (
    Batch,
    ReplayBuffer,
    to_torch,
)

from tianshou.utils.torch_utils import (
    torch_train_mode,
)

from tianshou.data.types import (
    RolloutBatchProtocol,
)

from imitation.masked_bc_v1 import (
    q_filtered_masked_behavior_cloning_loss,
)

from training.masked_auto_alpha_v1 import (
    MaskedAutoAlphaV1,
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

    def update_equal_replay(
        self,
        *,
        demo_buffer: ReplayBuffer,
        expo_buffer: ReplayBuffer,
        samples_per_buffer: int,
    ):
        """
        Stage-II equal replay update.

        Exactly:

            N transitions from D_demo
            N transitions from D_expo

        are sampled for ONE optimizer update.

        Each source batch is preprocessed against its OWN replay
        buffer before concatenation, preserving correct n-step
        episode semantics.

        Stage-II must be SAC-only, so BC must already be disabled.
        """

        samples_per_buffer = int(
            samples_per_buffer
        )

        if samples_per_buffer <= 0:
            raise ValueError(
                "samples_per_buffer must "
                "be positive"
            )

        if self.bc_enabled:
            raise RuntimeError(
                "update_equal_replay() is a "
                "Stage-II operation and requires "
                "bc_enabled=False"
            )

        if len(
            demo_buffer
        ) <= 0:
            raise ValueError(
                "D_demo is empty"
            )

        if len(
            expo_buffer
        ) <= 0:
            raise ValueError(
                "D_expo is empty"
            )

        if not self.policy.is_within_training_step:
            raise RuntimeError(
                "update_equal_replay() must be "
                "called within a Tianshou "
                "training step"
            )

        # ======================================================
        # Sample independently.
        # ReplayBuffer sampling is with replacement, matching
        # normal Tianshou ReplayBuffer.sample() behavior.
        # ======================================================

        demo_batch, demo_indices = (
            demo_buffer.sample(
                samples_per_buffer
            )
        )

        expo_batch, expo_indices = (
            expo_buffer.sample(
                samples_per_buffer
            )
        )

        # ======================================================
        # CRITICAL:
        # preprocess each source with its own replay topology.
        # ======================================================

        demo_batch = (
            self._preprocess_batch(
                demo_batch,
                demo_buffer,
                demo_indices,
            )
        )

        expo_batch = (
            self._preprocess_batch(
                expo_batch,
                expo_buffer,
                expo_indices,
            )
        )

        # ------------------------------------------------------
        # info / policy are not learning inputs for our current
        # LoopyCuts Actor/Critic.
        #
        # D_expo may contain runtime info while D_demo does not.
        # Normalize them before Batch.cat().
        # ------------------------------------------------------

        demo_batch.info = Batch()
        expo_batch.info = Batch()

        demo_batch.policy = Batch()
        expo_batch.policy = Batch()

        # Source marker is diagnostic only.
        demo_batch.replay_source = np.zeros(
            samples_per_buffer,
            dtype=np.int8,
        )

        expo_batch.replay_source = np.ones(
            samples_per_buffer,
            dtype=np.int8,
        )

        # ------------------------------------------------------
        # Do NOT shuffle here.
        #
        # Sampling within each buffer is already random.
        # Keeping:
        #
        #   [demo rows][expo rows]
        #
        # lets us map TD weights back to the two original buffers
        # if prioritized replay is introduced later.
        # ------------------------------------------------------

        mixed_batch = Batch.cat(
            [
                demo_batch,
                expo_batch,
            ]
        )

        expected_size = (
            2
            *
            samples_per_buffer
        )

        if len(
            mixed_batch
        ) != expected_size:
            raise RuntimeError(
                "Mixed replay batch has "
                "unexpected size"
            )

        source = np.asarray(
            mixed_batch.replay_source,
            dtype=np.int8,
        )

        if (
            int(
                np.sum(
                    source == 0
                )
            )
            !=
            samples_per_buffer
        ):
            raise RuntimeError(
                "D_demo sample count is not 1:1"
            )

        if (
            int(
                np.sum(
                    source == 1
                )
            )
            !=
            samples_per_buffer
        ):
            raise RuntimeError(
                "D_expo sample count is not 1:1"
            )

        # ======================================================
        # ONE optimizer update on the combined 1:1 minibatch.
        # ======================================================

        with torch_train_mode(
            self
        ):
            stats = (
                self._update_with_batch(
                    mixed_batch
                )
            )

        # ======================================================
        # Preserve Tianshou post-processing semantics separately
        # for each source buffer.
        #
        # _update_with_batch writes TD error into batch.weight.
        # ======================================================

        if hasattr(
            mixed_batch,
            "weight",
        ):
            demo_batch.weight = (
                mixed_batch.weight[
                    :samples_per_buffer
                ]
            )

            expo_batch.weight = (
                mixed_batch.weight[
                    samples_per_buffer:
                ]
            )

        self._postprocess_batch(
            demo_batch,
            demo_buffer,
            demo_indices,
        )

        self._postprocess_batch(
            expo_batch,
            expo_buffer,
            expo_indices,
        )

        for scheduler in self.lr_schedulers:
            scheduler.step()

        return (
            stats,
            {
                "demo_samples":
                    samples_per_buffer,

                "expo_samples":
                    samples_per_buffer,

                "total_samples":
                    expected_size,
            },
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

        if isinstance(
            self.alpha,
            MaskedAutoAlphaV1,
        ):
            if not hasattr(
                batch.obs,
                "mask",
            ):
                raise RuntimeError(
                    "MaskedAutoAlphaV1 requires "
                    "batch.obs.mask"
                )

            alpha_update = (
                self.alpha.update_with_mask(
                    entropy=
                        entropy.detach(),

                    mask=
                        batch.obs.mask,
                )
            )

            alpha_loss = (
                alpha_update.loss
            )

        else:
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
