from typing import Any

import numpy as np
import torch

from torch.distributions import (
    Categorical,
)

from tianshou.algorithm.modelfree.discrete_sac import (
    DiscreteSACPolicy,
)

from tianshou.data import (
    Batch,
)

from tianshou.data.batch import (
    BatchProtocol,
)

from tianshou.data.types import (
    ObsBatchProtocol,
)


MASKED_EPSILON_GREEDY_VERSION = (
    "loopycuts_masked_epsilon_greedy_v1"
)


class MaskedDiscreteSACPolicy(
    DiscreteSACPolicy
):
    """
    Discrete SAC policy with explicit dynamic action masking.

    Expected observation structure:

        batch.obs.obs
            Actual neural-network observation.

        batch.obs.mask
            Boolean action mask of shape:

                (batch_size, action_dim)

            True:
                action is legal

            False:
                action is illegal

    Important terminal-state behavior
    ---------------------------------

    LoopyCuts terminal states have no legal actions, therefore their
    mask is all False.

    Tianshou still evaluates the target policy on obs_next before
    applying its terminated-state value mask. Therefore an all-False
    mask cannot simply become all -inf logits, because that would not
    define a valid Categorical distribution.

    For an all-False row, action 0 is temporarily enabled internally
    as a numerical fallback.

    This fallback does NOT make action 0 legal in the LoopyCuts MDP.
    For a truly terminated transition, Tianshou later multiplies the
    resulting bootstrap value by zero.
    """

    def __init__(
        self,
        *,
        actor,
        action_space,
        deterministic_eval: bool = True,
        observation_space=None,
        exploration_epsilon: float = 0.0,
        exploration_seed: int | None = None,
    ):
        super().__init__(
            actor=actor,
            action_space=action_space,
            deterministic_eval=deterministic_eval,
            observation_space=observation_space,
        )

        self._exploration_rng = (
            np.random.default_rng(
                exploration_seed
            )
        )

        self.set_exploration_epsilon(
            exploration_epsilon
        )

    # --------------------------------------------------------------

    def set_exploration_epsilon(
        self,
        epsilon: float,
    ):
        epsilon = float(
            epsilon
        )

        if (
            not np.isfinite(
                epsilon
            )
            or
            epsilon < 0.0
            or
            epsilon > 1.0
        ):
            raise ValueError(
                "exploration_epsilon must "
                "be finite and in [0, 1]"
            )

        self.exploration_epsilon = (
            epsilon
        )

    # --------------------------------------------------------------

    def add_exploration_noise(
        self,
        act,
        batch: ObsBatchProtocol,
    ):
        """
        Apply masked epsilon-greedy exploration to Collector actions.

        With probability epsilon independently for each current state:

            replace the policy action by a uniformly sampled
            CURRENTLY LEGAL action.

        Otherwise preserve the policy action.

        This method affects behavior-policy collection only.
        It does NOT modify the SAC policy distribution used for:

            entropy
            actor loss
            target-Q expectation.
        """

        if not isinstance(
            act,
            np.ndarray,
        ):
            raise TypeError(
                "Masked epsilon-greedy expects "
                "Collector actions as np.ndarray"
            )

        action_array = np.asarray(
            act
        )

        original_shape = (
            action_array.shape
        )

        flat_actions = (
            action_array
            .reshape(
                -1
            )
            .copy()
        )

        (
            _,
            mask,
        ) = (
            self
            ._extract_actor_obs_and_mask(
                batch.obs
            )
        )

        mask_array = np.asarray(
            mask,
            dtype=np.bool_,
        )

        if (
            mask_array.ndim == 1
            and
            flat_actions.size == 1
        ):
            mask_array = (
                mask_array.reshape(
                    1,
                    -1,
                )
            )

        if mask_array.ndim != 2:
            raise ValueError(
                "Current action mask must have "
                "shape [batch_size, action_dim]"
            )

        if (
            int(
                mask_array.shape[
                    0
                ]
            )
            !=
            int(
                flat_actions.size
            )
        ):
            raise ValueError(
                "Action batch size does not "
                "match mask batch size"
            )

        legal_counts = (
            mask_array.sum(
                axis=1
            )
        )

        if bool(
            (
                legal_counts
                <=
                0
            ).any()
        ):
            bad_row = int(
                np.flatnonzero(
                    legal_counts
                    <=
                    0
                )[
                    0
                ]
            )

            raise ValueError(
                "Masked epsilon-greedy current "
                "state has no legal actions at "
                f"batch row {bad_row}"
            )

        # The incoming action came from the already-masked policy.
        # Verify that invariant before adding exploration.
        integer_actions = (
            flat_actions.astype(
                np.int64,
                copy=False,
            )
        )

        if bool(
            (
                integer_actions
                <
                0
            ).any()
        ) or bool(
            (
                integer_actions
                >=
                mask_array.shape[
                    1
                ]
            ).any()
        ):
            raise ValueError(
                "Policy action is outside "
                "the mask action dimension"
            )

        row_ids = np.arange(
            integer_actions.size
        )

        if not bool(
            mask_array[
                row_ids,
                integer_actions,
            ].all()
        ):
            raise RuntimeError(
                "Incoming policy action is illegal "
                "under the current mask"
            )

        if (
            self.exploration_epsilon
            ==
            0.0
        ):
            return act

        explore_rows = (
            self
            ._exploration_rng
            .random(
                integer_actions.size
            )
            <
            self.exploration_epsilon
        )

        result = (
            integer_actions.copy()
        )

        for row in np.flatnonzero(
            explore_rows
        ):
            legal_actions = (
                np.flatnonzero(
                    mask_array[
                        row
                    ]
                )
            )

            result[
                row
            ] = int(
                self
                ._exploration_rng
                .choice(
                    legal_actions
                )
            )

        # Final hard invariant:
        # every action actually returned to Collector is legal.
        if not bool(
            mask_array[
                row_ids,
                result,
            ].all()
        ):
            raise RuntimeError(
                "Masked epsilon-greedy produced "
                "an illegal action"
            )

        return (
            result
            .reshape(
                original_shape
            )
            .astype(
                action_array.dtype,
                copy=False,
            )
        )

    # --------------------------------------------------------------

    @staticmethod
    def _extract_actor_obs_and_mask(
        observation,
    ):
        """
        Extract:

            actual observation
            action mask

        from the Tianshou Batch representation of a Gymnasium
        Dict observation.
        """

        if not hasattr(
            observation,
            "obs",
        ):
            raise ValueError(
                "MaskedDiscreteSACPolicy expects "
                "batch.obs.obs"
            )

        if not hasattr(
            observation,
            "mask",
        ):
            raise ValueError(
                "MaskedDiscreteSACPolicy expects "
                "batch.obs.mask"
            )

        actor_obs = observation.obs
        mask = observation.mask

        return (
            actor_obs,
            mask,
        )

    # --------------------------------------------------------------

    @staticmethod
    def _apply_action_mask(
        logits,
        mask,
    ):
        """
        Apply a boolean action mask to actor logits.

        Illegal actions receive -inf logits.

        If a row contains no legal action at all, action 0 is enabled
        only as an internal terminal-state fallback so that PyTorch can
        construct a valid Categorical distribution.
        """

        if not isinstance(
            logits,
            torch.Tensor,
        ):
            raise TypeError(
                "Actor logits must be "
                "a torch.Tensor"
            )

        mask_tensor = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=logits.device,
        )

        #
        # A single environment observation can occasionally arrive
        # as shape (A,) while the actor output is (1, A).
        #
        if (
            mask_tensor.ndim == 1
            and logits.ndim == 2
            and logits.shape[0] == 1
        ):
            mask_tensor = (
                mask_tensor.unsqueeze(0)
            )

        if (
            mask_tensor.shape
            != logits.shape
        ):
            raise ValueError(
                "Action mask shape does not "
                "match actor logits: "
                f"mask={tuple(mask_tensor.shape)}, "
                f"logits={tuple(logits.shape)}"
            )

        #
        # Do not modify the original mask object.
        #
        safe_mask = (
            mask_tensor.clone()
        )

        #
        # terminal rows:
        #
        #     False False ... False
        #
        # Tianshou may still evaluate the target policy on those
        # obs_next states before multiplying the bootstrap value
        # by zero.
        #
        no_legal_action = (
            ~safe_mask.any(
                dim=-1
            )
        )

        if bool(
            no_legal_action.any()
        ):
            safe_mask[
                no_legal_action,
                0,
            ] = True

        masked_logits = (
            logits.masked_fill(
                ~safe_mask,
                float("-inf"),
            )
        )

        return (
            masked_logits,
            no_legal_action,
        )

    # --------------------------------------------------------------

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: (
            dict
            | BatchProtocol
            | np.ndarray
            | None
        ) = None,
        **kwargs: Any,
    ) -> Batch:

        (
            actor_obs,
            mask,
        ) = (
            self
            ._extract_actor_obs_and_mask(
                batch.obs
            )
        )

        #
        # The actor sees only the actual state features.
        #
        # It does NOT see the action mask as an ordinary numerical
        # feature here. The mask is applied explicitly below.
        #
        (
            raw_logits,
            hidden_state,
        ) = self.actor(
            actor_obs,
            state=state,
            info=batch.info,
        )

        (
            masked_logits,
            no_legal_action,
        ) = (
            self._apply_action_mask(
                raw_logits,
                mask,
            )
        )

        #
        # Important:
        #
        # Construct the Categorical distribution from MASKED logits.
        #
        # Therefore:
        #
        #   sampling
        #   entropy
        #   dist.probs
        #   SAC target expectation
        #   SAC actor objective
        #
        # all use the same legal-action distribution.
        #
        dist = Categorical(
            logits=masked_logits
        )

        if (
            self.deterministic_eval
            and not
            self.is_within_training_step
        ):
            action = dist.mode
        else:
            action = dist.sample()

        return Batch(
            logits=masked_logits,
            act=action,
            state=hidden_state,
            dist=dist,

            #
            # Diagnostic only.
            #
            # True means this row had no actual legal action and the
            # internal terminal fallback was used.
            #
            mask_fallback=
                no_legal_action,
        )
