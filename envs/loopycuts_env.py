from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bridge.cpp_client import LoopyCutsClient
from dataset_tools.loop_metadata import parse_loop_metadata
from observation.builder import (
    GLOBAL_DIM,
    LOOP_FEATURE_DIM,
    MAX_LOOPS,
    LoopyCutsObservationBuilder,
)
from rewards.reward_v1 import (
    compute_reward_v1,
)
from rewards.transition_metrics import (
    extract_transition_metrics,
)


class LoopyCutsEnv(gym.Env):
    """
    Gymnasium wrapper for LoopyCuts Stage-2 loop selection.

    V1 responsibility:
        - own one persistent C++ RL-server process per episode;
        - preserve original LoopyCuts loop IDs as actions;
        - expose the frozen Observation V1;
        - enforce C++ ACTIONS as authoritative legality;
        - maintain Python-side executed_loop_ids;
        - terminate when the C++ Stage-2 selection state is terminal.

    Reward:
        Selection Reward V1.

        The reward is computed from the real C++ transition using:
            state_before
            -> STEP_RESULT
            -> state_after
            -> TransitionMetrics
            -> Reward V1

        FINALIZE/full-hex outcome is deliberately not part of
        Selection Reward V1.
    """

    metadata = {
        "render_modes": [],
    }

    reward_range = (
        -np.inf,
        np.inf,
    )

    def __init__(
        self,
        *,
        executable: str | Path,
        mesh_file: str | Path,
        loop_file: str | Path,
        echo_logs: bool = False,
    ):
        super().__init__()

        self.executable = Path(
            executable
        )

        self.mesh_file = Path(
            mesh_file
        )

        self.loop_file = Path(
            loop_file
        )

        self.echo_logs = bool(
            echo_logs
        )

        # ------------------------------------------------------------
        # Fail early on missing input paths.
        # ------------------------------------------------------------

        if not self.executable.is_file():
            raise FileNotFoundError(
                f"LoopyCuts executable not found: "
                f"{self.executable}"
            )

        if not self.mesh_file.is_file():
            raise FileNotFoundError(
                f"Mesh file not found: "
                f"{self.mesh_file}"
            )

        if not self.loop_file.is_file():
            raise FileNotFoundError(
                f"Loop file not found: "
                f"{self.loop_file}"
            )

        # ------------------------------------------------------------
        # Static Stage-1 metadata is immutable across episodes.
        # Parse once.
        # ------------------------------------------------------------

        self.loop_metadata = (
            parse_loop_metadata(
                self.loop_file
            )
        )

        self.num_loops = len(
            self.loop_metadata
        )

        if self.num_loops > MAX_LOOPS:
            raise ValueError(
                f"Model has {self.num_loops} serialized loops, "
                f"but MAX_LOOPS={MAX_LOOPS}. "
                "V1 must not truncate or remap loop IDs."
            )

        # ------------------------------------------------------------
        # Action ID == original LoopyCuts loop ID.
        #
        # Discrete(331) contains the entire fixed author-corpus
        # namespace. Dynamic legality is supplied ONLY by the
        # observation mask emitted from authoritative C++ ACTIONS.
        # ------------------------------------------------------------

        self.action_space = spaces.Discrete(
            MAX_LOOPS
        )

        # ------------------------------------------------------------
        # Observation V1.
        #
        # MultiBinary is used for exists/mask because they are
        # semantically binary arrays. The Observation Builder itself
        # deliberately returns np.bool_ arrays.
        # ------------------------------------------------------------

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Dict(
                    {
                        "global": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                GLOBAL_DIM,
                            ),
                            dtype=np.float32,
                        ),

                        "loops": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                MAX_LOOPS,
                                LOOP_FEATURE_DIM,
                            ),
                            dtype=np.float32,
                        ),

                        "exists": spaces.MultiBinary(
                            MAX_LOOPS
                        ),
                    }
                ),

                "mask": spaces.MultiBinary(
                    MAX_LOOPS
                ),
            }
        )

        # ------------------------------------------------------------
        # Episode-owned objects.
        # ------------------------------------------------------------

        self._client: (
            LoopyCutsClient
            | None
        ) = None

        self._builder: (
            LoopyCutsObservationBuilder
            | None
        ) = None

        self.executed_loop_ids: set[int] = set()

    # ------------------------------------------------------------------

    @property
    def client(
        self,
    ) -> LoopyCutsClient:
        if self._client is None:
            raise RuntimeError(
                "Environment has not been reset"
            )

        return self._client

    # ------------------------------------------------------------------

    @property
    def legal_actions(
        self,
    ) -> tuple[int, ...]:
        if self._client is None:
            return ()

        return tuple(
            self._client.actions
        )

    # ------------------------------------------------------------------

    @property
    def current_state(
        self,
    ) -> dict[str, Any] | None:
        if (
            self._client is None
            or
            self._client.state is None
        ):
            return None

        return dict(
            self._client.state
        )

    # ------------------------------------------------------------------

    def _build_observation(
        self,
    ) -> dict:
        if self._builder is None:
            raise RuntimeError(
                "Observation builder is not initialized"
            )

        client = self.client

        return self._builder.build(
            state=client.state,
            actions=client.actions,
            used=client.used,
            reverted=client.reverted,
            nico_bug=client.nico_bug,
            top_relevant=client.top_relevant,
            executed=self.executed_loop_ids,
        )

    # ------------------------------------------------------------------

    def _make_info(
        self,
        *,
        step_result: dict | None = None,
    ) -> dict[str, Any]:
        """
        Auxiliary information only.

        None of these fields are required for the policy observation.
        """

        client = self.client

        info: dict[str, Any] = {
            "state":
                dict(
                    client.state
                ),

            "num_legal_actions":
                len(
                    client.actions
                ),

            "num_executed":
                len(
                    self.executed_loop_ids
                ),

            "reward_is_placeholder":
                False,

            "reward_version":
                "selection_v1",
        }

        if step_result is not None:
            info[
                "step_result"
            ] = dict(
                step_result
            )

        return info

    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        #
        # Gymnasium seeding contract.
        #
        super().reset(
            seed=seed
        )

        #
        # V1 fixed-model environment currently has no reset options.
        # Accept None or an empty dictionary only.
        #
        if (
            options is not None
            and len(options) != 0
        ):
            raise ValueError(
                "LoopyCutsEnv V1 does not support "
                "non-empty reset options"
            )

        # ------------------------------------------------------------
        # One fresh persistent C++ GlobalState per episode.
        #
        # If reset() is called for a second episode, close the previous
        # child process first.
        # ------------------------------------------------------------

        self.close()

        self.executed_loop_ids = set()

        client = LoopyCutsClient(
            executable=self.executable,
            mesh_file=self.mesh_file,
            loop_file=self.loop_file,
            echo_logs=self.echo_logs,
        )

        try:
            builder = (
                LoopyCutsObservationBuilder(
                    metadata=self.loop_metadata,
                    initial_state=client.state,
                    initial_actions=client.actions,
                )
            )

            self._client = client
            self._builder = builder

            observation = (
                self._build_observation()
            )

            if not self.observation_space.contains(
                observation
            ):
                raise RuntimeError(
                    "Initial observation is not contained "
                    "in observation_space"
                )

            info = self._make_info()

        except Exception:
            client.close()

            self._client = None
            self._builder = None

            raise

        return (
            observation,
            info,
        )

    # ------------------------------------------------------------------

    def step(
        self,
        action,
    ):
        client = self.client

        # ------------------------------------------------------------
        # Stepping after terminal is a caller error.
        # ------------------------------------------------------------

        if int(
            client.state[
                "terminal"
            ]
        ):
            raise RuntimeError(
                "Cannot call step() after terminal state; "
                "call reset() first"
            )

        # ------------------------------------------------------------
        # Do not silently coerce floats or booleans into loop IDs.
        # ------------------------------------------------------------

        if isinstance(
            action,
            (
                bool,
                np.bool_,
            ),
        ):
            raise TypeError(
                "Action must be an integer loop ID, not bool"
            )

        if not isinstance(
            action,
            (
                int,
                np.integer,
            ),
        ):
            raise TypeError(
                "Action must be an integer loop ID"
            )

        action = int(
            action
        )

        if not self.action_space.contains(
            action
        ):
            raise ValueError(
                f"Action {action} is outside "
                f"Discrete({MAX_LOOPS})"
            )

        # ------------------------------------------------------------
        # C++ ACTIONS is the one and only legality authority.
        #
        # Do NOT reconstruct legality from:
        #
        #     loop type,
        #     used/reverted,
        #     convergence,
        #     regular_phase_closed,
        #     or Python metadata.
        # ------------------------------------------------------------

        if action not in client.actions:
            raise ValueError(
                f"Illegal LoopyCuts action {action}. "
                f"Current legal actions: "
                f"{client.actions}"
            )

        # ------------------------------------------------------------
        # Execute one real Stage-2 action.
        #
        # Reward V1 must be derived from the actual transition, so
        # preserve the pre-action C++ STATE before STEP mutates it.
        # ------------------------------------------------------------

        state_before = dict(
            client.state
        )

        (
            step_result,
            _,
            _,
        ) = client.step(
            action
        )

        #
        # executed means the agent actually selected this action.
        #
        # This is deliberately distinct from C++ Loop.used because
        # find_mates() may consume loops that the agent never selected.
        #
        self.executed_loop_ids.add(
            action
        )

        observation = (
            self._build_observation()
        )

        if not self.observation_space.contains(
            observation
        ):
            raise RuntimeError(
                "Post-step observation is not contained "
                "in observation_space"
            )

        # ------------------------------------------------------------
        # Selection Reward V1.
        # ------------------------------------------------------------

        state_after = dict(
            client.state
        )

        transition_metrics = (
            extract_transition_metrics(
                state_before=state_before,
                step_result=step_result,
                state_after=state_after,
            )
        )

        if self._builder is None:
            raise RuntimeError(
                "Observation builder is not initialized"
            )

        reward_breakdown = (
            compute_reward_v1(
                metrics=transition_metrics,
                initial_actionable_count=(
                    self._builder
                    .initial_actionable_count
                ),
            )
        )

        reward = float(
            reward_breakdown.total
        )

        terminated = bool(
            client.state[
                "terminal"
            ]
        )

        #
        # No artificial time limit exists in V1.
        # Stage-2 terminates through the C++ legal-action semantics.
        #
        truncated = False

        info = self._make_info(
            step_result=step_result
        )

        info[
            "transition_metrics"
        ] = transition_metrics.to_dict()

        info[
            "reward_breakdown"
        ] = {
            "step":
                reward_breakdown.step,

            "tet_growth":
                reward_breakdown.tet_growth,

            "revert":
                reward_breakdown.revert,

            "convergence":
                reward_breakdown.convergence,

            "terminal":
                reward_breakdown.terminal,

            "total":
                reward_breakdown.total,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------------------

    def render(
        self,
    ):
        return None

    # ------------------------------------------------------------------

    def close(
        self,
    ):
        if self._client is not None:
            try:
                self._client.close()

            finally:
                self._client = None
                self._builder = None
