from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid


# <YYYYMMDD_HHMMSS>_<8 hex>; group 1 is the creation second
RUN_ID_PATTERN = re.compile(r"^(\d{8}_\d{6})_[0-9a-f]{8}$")


def new_run_id() -> str:
    """
    Example: 20260205_142240_0f347ac5
    (time + short uuid)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}"


@dataclass(frozen=True)
class RunContext:
    """
    A small object that always knows:
      - where the run folder is
      - what 'task' produced it (quantitative_weights, train, gui, etc.)
      - run_id
    """
    task: str
    run_id: str
    run_dir: Path

    def ensure(self) -> "RunContext":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self


def create_run_dir(
    task: str,
    base_runs_dir: str | Path = "outputs/runs",
    run_id: str | None = None,
) -> RunContext:
    """
    Creates:
      outputs/runs/<task>/<run_id>/

    If run_id is None => auto-generate.
    """
    task = task.strip().replace(" ", "_")
    rid = run_id or new_run_id()
    run_dir = Path(base_runs_dir) / task / rid
    return RunContext(task=task, run_id=rid, run_dir=run_dir).ensure()
