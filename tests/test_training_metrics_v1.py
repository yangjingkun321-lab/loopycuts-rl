import sys

from pathlib import Path
from tempfile import TemporaryDirectory


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


from training.training_metrics_v1 import (
    TRAINING_METRICS_FILENAME,
    TRAINING_METRICS_VERSION,
    TrainingMetricsError,
    TrainingMetricsWriterV1,
)


def main():
    with TemporaryDirectory() as tmp:
        path = (
            Path(tmp)
            /
            TRAINING_METRICS_FILENAME
        )

        writer = TrainingMetricsWriterV1(
            path=path
        )

        result = writer.append(
            seed=42,
            stage="STAGE_I",
            gradient_update=1,
            sampled_demo_transitions=64,
            stats={
                "actor_loss": 1.25,
                "critic1_loss": 0.5,
                "critic2_loss": 0.6,
                "bc_loss": 0.2,
                "alpha": 1.0,
            },
        )

        assert (
            result[
                "replayed"
            ]
            is False
        )

        assert writer.record_count() == 1

        # Exact deterministic replay must not append
        # a duplicate physical line.
        result = writer.append(
            seed=42,
            stage="STAGE_I",
            gradient_update=1,
            sampled_demo_transitions=64,
            stats={
                "actor_loss": 1.25,
                "critic1_loss": 0.5,
                "critic2_loss": 0.6,
                "bc_loss": 0.2,
                "alpha": 1.0,
            },
        )

        assert (
            result[
                "replayed"
            ]
            is True
        )

        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()

        assert len(lines) == 1

        # Reopen from disk and verify replay semantics
        # survive process restart.
        writer2 = TrainingMetricsWriterV1(
            path=path
        )

        assert writer2.record_count() == 1

        assert (
            writer2.stage_record_count(
                seed=42,
                stage="STAGE_I",
            )
            ==
            1
        )

        writer2.assert_complete_prefix(
            seed=42,
            stage="STAGE_I",
            gradient_updates=1,
        )

        writer2.assert_exact_stage(
            seed=42,
            stage="STAGE_I",
            gradient_updates=1,
        )

        writer2.sync()

        # A checkpoint claiming update 2 must fail because only
        # update 1 is durable in the sidecar.
        try:
            writer2.assert_complete_prefix(
                seed=42,
                stage="STAGE_I",
                gradient_updates=2,
            )

        except TrainingMetricsError:
            pass

        else:
            raise RuntimeError(
                "Incomplete metrics checkpoint prefix was accepted"
            )

        result = writer2.append(
            seed=42,
            stage="STAGE_I",
            gradient_update=1,
            sampled_demo_transitions=64,
            stats={
                "actor_loss": 1.25,
                "critic1_loss": 0.5,
                "critic2_loss": 0.6,
                "bc_loss": 0.2,
                "alpha": 1.0,
            },
        )

        assert (
            result[
                "replayed"
            ]
            is True
        )

        # Same formal update key with different
        # numerical training data must fail closed.
        try:
            writer2.append(
                seed=42,
                stage="STAGE_I",
                gradient_update=1,
                sampled_demo_transitions=64,
                stats={
                    "actor_loss": 999.0,
                },
            )

        except TrainingMetricsError:
            pass

        else:
            raise RuntimeError(
                "Conflicting metric replay was accepted"
            )

        # NaN/Inf are forbidden.
        try:
            writer2.append(
                seed=42,
                stage="STAGE_I",
                gradient_update=2,
                sampled_demo_transitions=128,
                stats={
                    "actor_loss": float("nan"),
                },
            )

        except TrainingMetricsError:
            pass

        else:
            raise RuntimeError(
                "Non-finite metric was accepted"
            )

        text = path.read_text(
            encoding="utf-8"
        )

        assert (
            TRAINING_METRICS_VERSION
            in
            text
        )

        # --------------------------------------------------------
        # Simulate interruption in the middle of the next JSONL
        # append. Only the incomplete final physical line may be
        # discarded. The previous complete record must survive.
        # --------------------------------------------------------

        with path.open(
            "ab"
        ) as f:
            f.write(
                b'{"schema_version":"incomplete'
            )

            f.flush()

        recovered = TrainingMetricsWriterV1(
            path=path
        )

        assert recovered.record_count() == 1

        recovered.assert_exact_stage(
            seed=42,
            stage="STAGE_I",
            gradient_updates=1,
        )

        recovered_text = path.read_bytes()

        assert recovered_text.endswith(
            b"\n"
        )

        assert (
            b"incomplete"
            not in
            recovered_text
        )

    print(
        "PASS: Training Metrics V1 append/replay "
        "contract"
    )


if __name__ == "__main__":
    main()
