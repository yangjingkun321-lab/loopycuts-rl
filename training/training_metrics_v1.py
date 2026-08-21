from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any


TRAINING_METRICS_VERSION = (
    "loopycuts_training_metrics_v1"
)

TRAINING_METRICS_FILENAME = (
    "training_metrics_v1.jsonl"
)


class TrainingMetricsError(
    RuntimeError
):
    pass


def _canonical_json(
    payload: dict[str, Any],
) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TrainingMetricsError(
            "Training metric is not canonical "
            "finite JSON"
        ) from exc


def _validate_record(
    record: dict[str, Any],
):
    if (
        record.get(
            "schema_version"
        )
        !=
        TRAINING_METRICS_VERSION
    ):
        raise TrainingMetricsError(
            "Training metric schema mismatch"
        )

    seed = int(
        record[
            "seed"
        ]
    )

    stage = str(
        record[
            "stage"
        ]
    )

    if stage not in {
        "STAGE_I",
        "STAGE_II",
    }:
        raise TrainingMetricsError(
            f"Unknown training metric stage: "
            f"{stage!r}"
        )

    gradient_update = int(
        record[
            "gradient_update"
        ]
    )

    if gradient_update <= 0:
        raise TrainingMetricsError(
            "gradient_update must be positive"
        )

    stats = record.get(
        "stats"
    )

    if not isinstance(
        stats,
        dict,
    ):
        raise TrainingMetricsError(
            "Training metric stats must be a dict"
        )

    return (
        seed,
        stage,
        gradient_update,
    )


class TrainingMetricsWriterV1:
    """
    Append-only observability sidecar.

    This class:
      * performs no model forward pass;
      * performs no replay sampling;
      * consumes no RNG;
      * mutates no optimizer/model/replay state.

    Existing identical records are treated as deterministic replay.
    Conflicting records for the same update key are fatal.
    """

    def __init__(
        self,
        *,
        path: Path,
    ):
        self.path = Path(
            path
        ).resolve()

        self._records = {}

        if self.path.is_file():
            raw = self.path.read_bytes()

            # A process may be killed between writing the JSON payload
            # and writing its trailing newline. Such a final partial
            # record cannot have been followed by a durable formal
            # checkpoint, so it is safe to discard and deterministically
            # replay from the last checkpoint.
            if (
                raw
                and
                not raw.endswith(
                    b"\n"
                )
            ):
                last_newline = raw.rfind(
                    b"\n"
                )

                recovered_size = (
                    0
                    if last_newline < 0
                    else last_newline + 1
                )

                # Preserve every already-complete record byte-for-byte.
                # Only remove the torn final suffix; never rewrite the
                # durable prefix during recovery.
                with self.path.open(
                    "r+b"
                ) as f:
                    f.truncate(
                        recovered_size
                    )

                    f.flush()

                    os.fsync(
                        f.fileno()
                    )

                raw = raw[
                    :recovered_size
                ]

            if raw:
                try:
                    decoded = raw.decode(
                        "utf-8"
                    )

                except UnicodeDecodeError as exc:
                    raise TrainingMetricsError(
                        "Training metrics file contains "
                        "invalid UTF-8 in a complete record"
                    ) from exc

                for line_number, raw_line in enumerate(
                    decoded.splitlines(),
                    start=1,
                ):
                    if not raw_line:
                        raise TrainingMetricsError(
                            "Training metrics file contains "
                            f"an empty line at {line_number}"
                        )

                    try:
                        record = json.loads(
                            raw_line
                        )

                    except json.JSONDecodeError as exc:
                        raise TrainingMetricsError(
                            "Invalid complete training metrics JSON "
                            f"at line {line_number}"
                        ) from exc

                    key = _validate_record(
                        record
                    )

                    canonical = _canonical_json(
                        record
                    )

                    existing = self._records.get(
                        key
                    )

                    if (
                        existing is not None
                        and
                        existing != canonical
                    ):
                        raise TrainingMetricsError(
                            "Conflicting duplicate training "
                            f"metric at key={key}"
                        )

                    self._records[
                        key
                    ] = canonical


    def append(
        self,
        *,
        seed: int,
        stage: str,
        gradient_update: int,
        stats: dict[str, Any],
        **metadata,
    ):
        record = {
            "schema_version":
                TRAINING_METRICS_VERSION,

            "seed":
                int(
                    seed
                ),

            "stage":
                str(
                    stage
                ),

            "gradient_update":
                int(
                    gradient_update
                ),

            "stats":
                dict(
                    stats
                ),
        }

        for key, value in metadata.items():
            record[
                str(
                    key
                )
            ] = value

        key = _validate_record(
            record
        )

        canonical = _canonical_json(
            record
        )

        existing = self._records.get(
            key
        )

        if existing is not None:
            if existing != canonical:
                raise TrainingMetricsError(
                    "Deterministic replay metric "
                    f"mismatch at key={key}"
                )

            return {
                "replayed":
                    True,

                "key":
                    key,
            }

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                canonical
            )

            f.write(
                "\n"
            )

            f.flush()

        self._records[
            key
        ] = canonical

        return {
            "replayed":
                False,

            "key":
                key,
        }


    def sync(
        self,
    ):
        """
        Make all currently appended telemetry durable.

        Production policy:
          * append() flushes userspace buffers per row;
          * sync() is called only at durable checkpoint boundaries.

        Therefore formal training does NOT fsync all 25,782 updates.
        """

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.flush()

            os.fsync(
                f.fileno()
            )


    def stage_record_count(
        self,
        *,
        seed: int,
        stage: str,
    ) -> int:
        seed = int(
            seed
        )

        stage = str(
            stage
        )

        if stage not in {
            "STAGE_I",
            "STAGE_II",
        }:
            raise TrainingMetricsError(
                f"Unknown metrics stage: {stage!r}"
            )

        return sum(
            1
            for (
                record_seed,
                record_stage,
                _,
            )
            in self._records
            if (
                record_seed
                ==
                seed
                and
                record_stage
                ==
                stage
            )
        )


    def assert_complete_prefix(
        self,
        *,
        seed: int,
        stage: str,
        gradient_updates: int,
    ):
        """
        Validate the telemetry prefix represented by a checkpoint.

        Extra suffix rows are deliberately allowed.

        Example:
          checkpoint says Stage-II update=2500
          metrics contains rows 1..2520

        This is legal after interruption because rows 2501..2520 may
        have been written after the last durable checkpoint. Resume
        deterministically replays them; append() requires an exact
        content match.
        """

        seed = int(
            seed
        )

        stage = str(
            stage
        )

        gradient_updates = int(
            gradient_updates
        )

        if stage not in {
            "STAGE_I",
            "STAGE_II",
        }:
            raise TrainingMetricsError(
                f"Unknown metrics stage: {stage!r}"
            )

        if gradient_updates < 0:
            raise TrainingMetricsError(
                "gradient_updates cannot be negative"
            )

        for update_index in range(
            1,
            gradient_updates + 1,
        ):
            key = (
                seed,
                stage,
                update_index,
            )

            if key not in self._records:
                raise TrainingMetricsError(
                    "Training metrics checkpoint prefix "
                    "is incomplete: "
                    f"seed={seed}, "
                    f"stage={stage}, "
                    f"missing_gradient_update={update_index}"
                )

        return {
            "seed":
                seed,

            "stage":
                stage,

            "gradient_updates":
                gradient_updates,
        }


    def assert_exact_stage(
        self,
        *,
        seed: int,
        stage: str,
        gradient_updates: int,
    ):
        """
        Final-run validation: exactly rows 1..N must exist.
        """

        self.assert_complete_prefix(
            seed=
                seed,

            stage=
                stage,

            gradient_updates=
                gradient_updates,
        )

        observed = sorted(
            update_index
            for (
                record_seed,
                record_stage,
                update_index,
            )
            in self._records
            if (
                record_seed
                ==
                int(seed)
                and
                record_stage
                ==
                str(stage)
            )
        )

        expected = list(
            range(
                1,
                int(gradient_updates) + 1,
            )
        )

        if observed != expected:
            raise TrainingMetricsError(
                "Training metrics final stage range mismatch: "
                f"seed={int(seed)}, "
                f"stage={str(stage)}, "
                f"expected_rows={int(gradient_updates)}, "
                f"observed_rows={len(observed)}"
            )

        return {
            "seed":
                int(seed),

            "stage":
                str(stage),

            "gradient_updates":
                int(gradient_updates),

            "record_count":
                len(
                    observed
                ),
        }


    def record_count(
        self,
    ) -> int:
        return len(
            self._records
        )
