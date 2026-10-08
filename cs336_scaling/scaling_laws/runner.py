"""Submit planned experiments to a backend, poll them, and persist the results.

The runner is idempotent and resumable: it first reconciles the local state with whatever
the backend already knows (matching on the training config's ``unique_id``, which is also
what the API uses to detect duplicates), then submits the remaining planned runs in order,
waiting for budget refunds when a reservation does not fit, and finally polls until every
submitted run has finished.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from cs336_scaling.scaling_laws.backends import (
    DuplicateExperimentError,
    ExperimentBackend,
    InsufficientBudgetError,
)
from cs336_scaling.scaling_laws.state import ExperimentRecord, PipelineState

console = Console()


def reconcile(state: PipelineState, backend: ExperimentBackend) -> int:
    """Pull the backend's status for every tracked config; returns the number updated."""
    remote = {e.training_config.unique_id: e for e in backend.list_experiments()}
    updated = 0
    for record in state.experiments:
        response = remote.get(record.unique_id)
        if response is None:
            continue
        before = (record.status, len(record.val_losses))
        record.apply_response(response)
        if before != (record.status, len(record.val_losses)):
            updated += 1
    return updated


def _format_record(record: ExperimentRecord) -> str:
    cfg = record.training_config
    base = (
        f"{record.name:<22} N={record.non_embedding_params / 1e6:7.1f}M "
        f"D={record.tokens / 1e6:8.1f}M B={cfg.train_batch_size:4d} "
        f"lr={record.peak_lr:.2e} max={cfg.max_runtime_seconds:7.0f}s"
    )
    if record.status == "completed":
        return f"{base} -> loss {record.final_loss:.4f} in {record.used_runtime_seconds:7.1f}s"
    if record.status == "failed":
        return (
            f"{base} -> FAILED ({record.failure}) after {record.used_runtime_seconds}s"
        )
    return f"{base} [{record.status}]"


def run_pending(
    state: PipelineState,
    backend: ExperimentBackend,
    state_path: Path,
    *,
    poll_interval_seconds: float = 30.0,
    budget_wait_seconds: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    quiet: bool = False,
) -> PipelineState:
    log = (lambda *_: None) if quiet else console.print

    def save() -> None:
        state.save(state_path)

    reconciled = reconcile(state, backend)
    if reconciled:
        log(f"[dim]reconciled {reconciled} record(s) with the backend[/dim]")
    save()

    # -- submit --------------------------------------------------------------------------
    for record in [r for r in state.experiments if r.status == "planned"]:
        while True:
            try:
                response = backend.submit(record.training_config)
            except DuplicateExperimentError:
                reconcile(state, backend)
                if record.status == "planned":
                    raise RuntimeError(
                        f"backend reports {record.name} as a duplicate but it is not in "
                        "the experiment list"
                    )
                log(f"[yellow]already submitted[/yellow] {_format_record(record)}")
                save()
                break
            except InsufficientBudgetError:
                in_flight = [
                    r for r in state.experiments if r.status in ("queued", "running")
                ]
                if not in_flight:
                    budget = backend.get_budget()
                    raise RuntimeError(
                        f"insufficient budget for {record.name} "
                        f"(needs {record.training_config.max_runtime_seconds}s, remaining "
                        f"{budget.remaining_seconds:.0f}s) and nothing is running to refund"
                    )
                log(
                    f"[dim]budget exhausted by reservations; waiting for {len(in_flight)} "
                    f"in-flight run(s) before submitting {record.name}[/dim]"
                )
                sleep(budget_wait_seconds)
                reconcile(state, backend)
                save()
                continue
            record.experiment_id = response.experiment_id
            record.status = "queued"
            log(
                f"[green]submitted[/green] #{response.experiment_id:<4} {_format_record(record)}"
                f"  (remaining budget {response.budget_summary.remaining_seconds / 3600:.2f}h)"
            )
            save()
            break

    # -- poll ----------------------------------------------------------------------------
    announced = {r.name for r in state.experiments if r.is_finished}
    while state.pending():
        reconcile(state, backend)
        save()
        for record in state.experiments:
            if record.is_finished and record.name not in announced:
                announced.add(record.name)
                colour = "green" if record.status == "completed" else "red"
                log(f"[{colour}]{record.status}[/{colour}] {_format_record(record)}")
        if state.pending():
            sleep(poll_interval_seconds)

    budget = backend.get_budget()
    log(
        f"[bold]all runs finished.[/bold] budget used {budget.used_seconds / 3600:.2f}h of "
        f"{budget.total_budget_seconds / 3600:.2f}h "
        f"({budget.remaining_seconds / 3600:.2f}h remaining)"
    )
    save()
    return state
