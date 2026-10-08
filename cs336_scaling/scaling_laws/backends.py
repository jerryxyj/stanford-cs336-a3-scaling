"""Experiment backends: the real training API and an offline simulator with the same interface."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Protocol

from cs336_scaling.scaling_laws.simulate import SimulatorTruth
from cs336_scaling.schemas import (
    BudgetSummary,
    ExperimentResponse,
    FinalSubmissionResponse,
    SubmitResponse,
)
from cs336_scaling.schemas.experiment import (
    CompletedExperimentStatus,
    FailedExperimentStatus,
    QueuedExperimentStatus,
    TimeoutReason,
)
from cs336_scaling.training.training_config import TrainingConfig


class DuplicateExperimentError(Exception):
    """The API already has an experiment with this training config (HTTP 409)."""


class InsufficientBudgetError(Exception):
    """``max_runtime_seconds`` exceeds the remaining budget (HTTP 400)."""


class ExperimentBackend(Protocol):
    name: str

    def get_budget(self) -> BudgetSummary: ...

    def submit(self, training_config: TrainingConfig) -> SubmitResponse: ...

    def list_experiments(self) -> list[ExperimentResponse]: ...

    def get_experiment(self, experiment_id: int) -> ExperimentResponse: ...

    def save_final_submission(
        self, training_config: TrainingConfig, predicted_final_loss: float
    ) -> FinalSubmissionResponse: ...


class RemoteBackend:
    """Thin wrapper over :mod:`cs336_scaling.client` that maps HTTP errors to exceptions."""

    name = "api"

    def get_budget(self) -> BudgetSummary:
        from cs336_scaling import client

        return client.get_budget()

    def submit(self, training_config: TrainingConfig) -> SubmitResponse:
        from cs336_scaling import client

        try:
            return client.submit_experiment(training_config)
        except RuntimeError as exc:
            message = str(exc)
            if "409" in message:
                raise DuplicateExperimentError(message) from exc
            if "400" in message and "insufficient budget" in message:
                raise InsufficientBudgetError(message) from exc
            raise

    def list_experiments(self) -> list[ExperimentResponse]:
        from cs336_scaling import client

        return client.list_experiments()

    def get_experiment(self, experiment_id: int) -> ExperimentResponse:
        from cs336_scaling import client

        return client.get_experiment(experiment_id)

    def save_final_submission(
        self, training_config: TrainingConfig, predicted_final_loss: float
    ) -> FinalSubmissionResponse:
        from cs336_scaling import client

        return client.save_final_submission(training_config, predicted_final_loss)


class SimulatedBackend:
    """In-process replica of the API's queue/budget semantics driven by :class:`SimulatorTruth`.

    Experiments are "queued" when submitted and finish on the next status query, so the
    runner's submit/poll loop is exercised. Budget accounting matches the real API: active
    experiments reserve ``max_runtime_seconds``; finished ones are charged
    ``clip(used_runtime, 1, max_runtime)``. If ``persist_path`` is given the backend's
    database survives across processes, like the real server's.
    """

    name = "simulated"

    def __init__(
        self,
        truth: SimulatorTruth | None = None,
        *,
        total_budget_seconds: float = 12 * 3600.0,
        persist_path: Path | None = None,
    ):
        self.truth = truth or SimulatorTruth()
        self.total_budget_seconds = total_budget_seconds
        self.persist_path = persist_path
        self._experiments: dict[int, ExperimentResponse] = {}
        self._final_submission: FinalSubmissionResponse | None = None
        if persist_path is not None and persist_path.exists():
            self._load()

    # -- persistence -------------------------------------------------------------------
    def _load(self) -> None:
        assert self.persist_path is not None
        raw = json.loads(self.persist_path.read_text())
        self._experiments = {
            int(k): ExperimentResponse.model_validate(v)
            for k, v in raw["experiments"].items()
        }
        if raw.get("final_submission") is not None:
            self._final_submission = FinalSubmissionResponse.model_validate(
                raw["final_submission"]
            )

    def _save(self) -> None:
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiments": {
                str(k): v.model_dump(mode="json") for k, v in self._experiments.items()
            },
            "final_submission": (
                self._final_submission.model_dump(mode="json")
                if self._final_submission
                else None
            ),
        }
        self.persist_path.write_text(json.dumps(payload, indent=2) + "\n")

    # -- budget ------------------------------------------------------------------------
    def _charged_seconds(self, experiment: ExperimentResponse) -> float:
        max_runtime = experiment.training_config.max_runtime_seconds
        status = experiment.status
        if status.status_type in ("completed", "failed"):
            return min(max(1.0, status.used_runtime_seconds), max_runtime)
        return max_runtime

    def get_budget(self) -> BudgetSummary:
        used = sum(self._charged_seconds(e) for e in self._experiments.values())
        return BudgetSummary(
            used_seconds=used,
            remaining_seconds=self.total_budget_seconds - used,
            total_budget_seconds=self.total_budget_seconds,
        )

    # -- API surface -------------------------------------------------------------------
    def submit(self, training_config: TrainingConfig) -> SubmitResponse:
        for experiment in self._experiments.values():
            if experiment.training_config.unique_id == training_config.unique_id:
                raise DuplicateExperimentError(
                    f"409 Conflict: experiment already exists for this training config "
                    f"(experiment_id={experiment.experiment_id})"
                )
        budget = self.get_budget()
        if training_config.max_runtime_seconds > budget.remaining_seconds:
            raise InsufficientBudgetError(
                f"400 Bad Request: insufficient budget ({budget.model_dump()})"
            )
        experiment_id = max(self._experiments, default=0) + 1
        self._experiments[experiment_id] = ExperimentResponse(
            experiment_id=experiment_id,
            training_config=training_config,
            status=QueuedExperimentStatus(queued_at=_now()),
        )
        self._save()
        return SubmitResponse(
            experiment_id=experiment_id,
            budget_summary=budget.with_reserved_runtime_seconds(
                training_config.max_runtime_seconds
            ),
        )

    def _finish(self, experiment: ExperimentResponse) -> ExperimentResponse:
        if experiment.status.status_type != "queued":
            return experiment
        outcome = self.truth.simulate(experiment.training_config)
        queued_at = experiment.status.queued_at
        now = _now()
        if outcome.completed:
            status = CompletedExperimentStatus(
                queued_at=queued_at,
                dispatched_at=queued_at,
                run_id=f"sim-{experiment.experiment_id}",
                used_runtime_seconds=outcome.used_runtime_seconds,
                val_losses=outcome.val_losses,
                completed_at=now,
            )
        else:
            status = FailedExperimentStatus(
                queued_at=queued_at,
                dispatched_at=queued_at,
                run_id=f"sim-{experiment.experiment_id}",
                used_runtime_seconds=outcome.used_runtime_seconds,
                reason=TimeoutReason(partial_val_losses=outcome.val_losses),
                failed_at=now,
            )
        finished = ExperimentResponse(
            experiment_id=experiment.experiment_id,
            training_config=experiment.training_config,
            status=status,
        )
        self._experiments[experiment.experiment_id] = finished
        return finished

    def list_experiments(self) -> list[ExperimentResponse]:
        result = [self._finish(e) for e in self._experiments.values()]
        self._save()
        return result

    def get_experiment(self, experiment_id: int) -> ExperimentResponse:
        if experiment_id not in self._experiments:
            raise KeyError(f"404: experiment {experiment_id} not found")
        result = self._finish(self._experiments[experiment_id])
        self._save()
        return result

    def save_final_submission(
        self, training_config: TrainingConfig, predicted_final_loss: float
    ) -> FinalSubmissionResponse:
        self._final_submission = FinalSubmissionResponse(
            training_config=training_config,
            predicted_final_loss=predicted_final_loss,
            submitted_at=_now(),
        )
        self._save()
        return self._final_submission

    def get_final_submission(self) -> FinalSubmissionResponse | None:
        return self._final_submission


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def make_backend(name: str, *, workdir: Path) -> ExperimentBackend:
    if name == "api":
        return RemoteBackend()
    if name == "simulated":
        return SimulatedBackend(persist_path=workdir / "simulated_server.json")
    raise ValueError(f"unknown backend {name!r}; expected 'api' or 'simulated'")
