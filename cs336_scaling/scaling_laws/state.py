"""Persistent record of planned / submitted / finished experiments for the scaling-law study.

The state file is the single source of truth for the pipeline: ``plan`` appends planned
experiments, ``run`` submits them and fills in results, ``fit`` reads the finished ones. It is
plain JSON so it can be inspected and committed alongside the write-up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cs336_scaling.scaling_laws.model_shapes import (
    count_parameters,
    flops_per_token,
    training_flops,
)
from cs336_scaling.scaling_laws.throughput import ThroughputObservation
from cs336_scaling.schemas import ExperimentResponse
from cs336_scaling.training.training_config import TrainingConfig

Stage = Literal["probe", "lr_sweep", "iso_time", "extra"]
RecordStatus = Literal["planned", "queued", "running", "completed", "failed"]


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    stage: Stage
    training_config: TrainingConfig
    ladder_k: int | None = None
    time_budget_seconds: float | None = None
    planned_at_round: int = 0
    experiment_id: int | None = None
    status: RecordStatus = "planned"
    used_runtime_seconds: float | None = None
    val_losses: list[float] = Field(default_factory=list)
    failure: str | None = None

    @property
    def unique_id(self) -> str:
        return self.training_config.unique_id

    @property
    def is_finished(self) -> bool:
        return self.status in ("completed", "failed")

    @property
    def final_loss(self) -> float | None:
        if self.status != "completed" or not self.val_losses:
            return None
        return self.val_losses[-1]

    @property
    def tokens(self) -> int:
        return self.training_config.total_train_tokens

    @property
    def non_embedding_params(self) -> int:
        return count_parameters(self.training_config.architecture_config).non_embedding

    @property
    def total_params(self) -> int:
        return count_parameters(self.training_config.architecture_config).total

    @property
    def flops(self) -> float:
        return training_flops(self.training_config.architecture_config, self.tokens)

    @property
    def peak_lr(self) -> float:
        return self.training_config.optimizer_config.lr_scheduler.peak_value

    @property
    def batch_tokens(self) -> int:
        return self.training_config.tokens_per_optimizer_step

    def throughput_observation(self) -> ThroughputObservation | None:
        """Tokens/sec from a finished run.

        Completed runs give ``tokens / used_runtime``. Timed-out runs still tell us how many
        evaluation chunks finished before the deadline, which gives a (slightly pessimistic,
        since the final partial chunk is not counted) throughput estimate.
        """
        if not self.used_runtime_seconds or self.used_runtime_seconds <= 0:
            return None
        if self.status == "completed":
            tokens = self.tokens
        elif self.status == "failed" and self.failure == "timeout" and self.val_losses:
            tokens = len(self.val_losses) * self.training_config.eval_every_tokens
        else:
            return None
        return ThroughputObservation(
            name=self.name,
            flops_per_token=flops_per_token(self.training_config.architecture_config),
            tokens_per_second=tokens / self.used_runtime_seconds,
            batch_tokens=self.batch_tokens,
        )

    def apply_response(self, response: ExperimentResponse) -> None:
        """Copy the API's view of this experiment into the record."""
        self.experiment_id = response.experiment_id
        status = response.status
        self.status = status.status_type
        match status.status_type:
            case "queued":
                pass
            case "running":
                self.val_losses = list(status.val_losses)
            case "completed":
                self.val_losses = list(status.val_losses)
                self.used_runtime_seconds = status.used_runtime_seconds
            case "failed":
                self.used_runtime_seconds = status.used_runtime_seconds
                if status.reason.reason == "timeout":
                    self.val_losses = list(status.reason.partial_val_losses)
                    self.failure = "timeout"
                else:
                    self.failure = status.reason.failure


class PipelineState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str
    experiments: list[ExperimentRecord] = Field(default_factory=list)
    planning_rounds: int = 0

    @classmethod
    def load(cls, path: Path) -> PipelineState:
        return cls.model_validate_json(Path(path).read_text())

    @classmethod
    def load_or_create(cls, path: Path, backend: str) -> PipelineState:
        path = Path(path)
        if path.exists():
            state = cls.load(path)
            if state.backend != backend:
                raise ValueError(
                    f"state file {path} was created for backend {state.backend!r}, "
                    f"refusing to reuse it for {backend!r}"
                )
            return state
        return cls(backend=backend)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        tmp.replace(path)

    def by_unique_id(self) -> dict[str, ExperimentRecord]:
        return {record.unique_id: record for record in self.experiments}

    def has_config(self, config: TrainingConfig) -> bool:
        return config.unique_id in self.by_unique_id()

    def add(self, record: ExperimentRecord) -> bool:
        """Append ``record`` unless an identical training config is already tracked."""
        if self.has_config(record.training_config):
            return False
        self.experiments.append(record)
        return True

    def stage(self, stage: Stage) -> list[ExperimentRecord]:
        return [r for r in self.experiments if r.stage == stage]

    def completed(self, stage: Stage | None = None) -> list[ExperimentRecord]:
        return [
            r
            for r in self.experiments
            if r.status == "completed" and (stage is None or r.stage == stage)
        ]

    def pending(self) -> list[ExperimentRecord]:
        return [r for r in self.experiments if not r.is_finished]

    def reserved_seconds(self) -> float:
        """Budget the API would charge for everything tracked so far (worst case)."""
        total = 0.0
        for r in self.experiments:
            max_runtime = r.training_config.max_runtime_seconds
            if r.is_finished and r.used_runtime_seconds is not None:
                total += min(max(1.0, r.used_runtime_seconds), max_runtime)
            else:
                total += max_runtime
        return total
