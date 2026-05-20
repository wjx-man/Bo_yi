"""CSV metric logging."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


class CSVLogger:
    """Append dictionaries to a CSV file."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if self.path.exists():
            with self.path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                existing_header = next(reader, None)
            if existing_header != fieldnames:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = self.path.with_suffix(f".{timestamp}.bak.csv")
                self.path.replace(backup)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    def log(self, row: dict[str, Any]) -> None:
        """Append one metric row."""
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
            writer.writerow({name: row.get(name, "") for name in self.fieldnames})
