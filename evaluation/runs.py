"""Creation of one evaluation run directory.

Every artefact this tool writes goes under evaluation/runs/<timestamp>/, which
../GOALS.md requires so that input/ and output/ are never touched. Two entry points: create_run_directory is strict and refuses any name that
already exists, and allocate_run_directory is what callers use - it asks for the
timestamp, then -2, -3 and so on until the strict call succeeds. The strictness
of the first is what guarantees the second can never write into an existing run.
"""

from datetime import datetime, timezone
from pathlib import Path

RUNS_ROOT = Path(__file__).resolve().parent / "runs"
RUN_SUBDIRECTORIES = ("hr", "lr", "bicubic", "sr")
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def create_run_directory(
    root: Path | None = None, moment: datetime | None = None, name: str | None = None,
    *, subdirectories=RUN_SUBDIRECTORIES,
) -> Path:
    """Create <root>/<UTC timestamp>/ with the run's subdirectories.

    Raises FileExistsError rather than writing into an existing directory: a
    previous run's record must never be overwritten.
    """
    root = RUNS_ROOT if root is None else root
    moment = datetime.now(timezone.utc) if moment is None else moment
    run = root / (moment.strftime(TIMESTAMP_FORMAT) if name is None else name)
    run.mkdir(parents=True, exist_ok=False)
    for name in subdirectories:
        (run / name).mkdir()
    return run


def allocate_run_directory(
    root: Path | None = None, moment: datetime | None = None, *, subdirectories=RUN_SUBDIRECTORIES
) -> Path:
    """A fresh run directory, suffixed if the timestamp is already taken.

    Two runs started in the same second both get a directory of their own, which
    the acceptance conditions require, and neither can touch the other's files -
    every attempt goes through create_run_directory's exist_ok=False.
    """
    base = (datetime.now(timezone.utc) if moment is None else moment).strftime(TIMESTAMP_FORMAT)
    attempt = 1
    while True:
        name = base if attempt == 1 else f"{base}-{attempt}"
        try:
            return create_run_directory(root, name=name, subdirectories=subdirectories)
        except FileExistsError:
            attempt += 1
