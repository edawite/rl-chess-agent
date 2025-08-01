"""Simple CSV logger for training and evaluation metrics.

This utility writes tabular data to a CSV file.  It is intentionally
lightweight and does not depend on external logging frameworks.  Each
instance of :class:`CSVLogger` opens its own file handle and writes a
header row on initialisation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Any, Optional


class CSVLogger:
    """Write rows of data to a CSV file with a predetermined header.

    Parameters
    ----------
    path : str or Path
        The location of the CSV file.  Parent directories are created
        automatically.
    headers : Iterable[str]
        Column names to write as the first row of the file.
    newline : str, optional
        Newline character passed to :func:`open`.  Defaults to ``""`` to
        ensure platform independent behaviour.
    """

    def __init__(self, path: Path | str, headers: Iterable[str], newline: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline=newline)
        self._writer = csv.writer(self._file)
        self._writer.writerow(list(headers))

    def log(self, row: Iterable[Any]) -> None:
        """Append a row of values to the CSV file.

        The length and order of values must match the header provided at
        initialisation.
        """
        self._writer.writerow(list(row))
        # Flush to disk to avoid data loss on unexpected termination.
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._file.close()

    def __enter__(self) -> "CSVLogger":  # type: ignore[override]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.close()
